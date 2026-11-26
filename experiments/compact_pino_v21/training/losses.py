"""Independent denominators for M/J and fixed-reference physical vector errors."""
from __future__ import annotations

import math
import torch

from ..p0_diagnostics.baseline_cases import moment_scale
from .residual_matching import residual_match_terms, validate_residual_target


def physical_loss(predictions, records, *, jacobian_loss_weight=1.,
                  residual_loss_weight=0., constraint_loss_weight=0.):
    """Return differentiable loss and the exact final membership/weights.

    Per-example M error averages ten squared errors scaled by m*L**degree.
    Per-example J error averages 10*D squared errors, also multiplied by the
    explicit chart parameter scales. Each objective divides by its own sum of
    admitted positive example weights, NEVER by the batch's missing J labels.
    Audit weights are final coefficients multiplying those per-example means.

    Residual means stationarity-vector DIFFERENCE at fixed p/lambda/mu, with
    native nonzero residuals preserved. Constraints likewise match the native
    vector, not zero. These two optional objectives have separate denominators
    and use record.residual_weight (default moment_weight). A batch without
    valid physical labels remains M/J-only; dataset admission and exposure
    accounting must check that an enabled objective was actually exercised.
    """
    if len(predictions) != len(records) or not records:
        raise ValueError('predictions and nonempty record batch must align')
    for name, value in (('Jacobian',jacobian_loss_weight),('residual',residual_loss_weight),
                        ('constraint',constraint_loss_weight)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'finite nonnegative {name} loss coefficient required')
    moment_terms, jacobian_terms, moment_weights, jacobian_weights = [], [], [], []
    residual_terms, constraint_terms, residual_weights, constraint_weights = [], [], [], []
    residual_reasons, residual_profiles, residual_approximations = [], [], []
    reasons = []
    for prediction, record in zip(predictions,records):
        case = record['case']; meta = case['meta']
        scale = prediction.moments.new_tensor(moment_scale(meta['editable_mass'],meta['length_scale']))
        truth = prediction.moments.new_tensor(case['native_moments'])
        if prediction.moments.shape != (10,) or not torch.isfinite(prediction.moments).all():
            raise ValueError('nonfinite or malformed predicted Moment')
        moment_terms.append(((prediction.moments-truth)/scale).square().mean())
        moment_weights.append(float(record['moment_weight']))
        valid = (meta['derivative_valid'] and prediction.jacobian is not None
                 and record['jacobian_weight'] > 0 and jacobian_loss_weight > 0)
        if valid:
            expected = prediction.jacobian.new_tensor(case['native_jacobian'])
            if prediction.jacobian.shape != expected.shape or not torch.isfinite(prediction.jacobian).all():
                raise ValueError('nonfinite or malformed predicted Jacobian')
            parameters = prediction.jacobian.new_tensor(case['parameter_scales'])
            error = (prediction.jacobian-expected)*parameters[None,:]/scale[:,None]
            jacobian_terms.append(error.square().mean())
            jacobian_weights.append(float(record['jacobian_weight']))
            reasons.append('QUALIFIED_AND_RESPONSE_VALID')
        else:
            jacobian_terms.append(None); jacobian_weights.append(0.)
            reasons.append('LABEL_UNQUALIFIED' if not meta['derivative_valid'] else
                           'RESPONSE_J_UNAVAILABLE' if prediction.jacobian is None else 'ZERO_LOSS_WEIGHT')
        target = record.get('residual_target')
        enabled = residual_loss_weight > 0 or constraint_loss_weight > 0
        rterm = cterm = None
        rweight = cweight = 0.
        profile = approximation = None
        reason = 'ZERO_LOSS_WEIGHT'
        if enabled:
            sample_weight = float(record.get('residual_weight',record['moment_weight']))
            if not math.isfinite(sample_weight) or sample_weight < 0:
                raise ValueError('finite nonnegative residual_weight required')
            if target is None:
                reason = 'NO_FIXED_RESIDUAL_TARGET'
            elif sample_weight == 0:
                reason = 'ZERO_EXAMPLE_WEIGHT'
            else:
                qualification = validate_residual_target(target,case['native_moments'],
                                                         coefficients=case.get('coefficients'))
                if qualification['availability'] != 'AVAILABLE':
                    reason = 'TARGET_UNAVAILABLE: ' + qualification['reason']
                else:
                    if 'coefficients' not in case:
                        raise ValueError('fixed residual matching requires case query coefficients')
                    terms = residual_match_terms(prediction.moments,case['native_moments'],target,
                                                 coefficients=case['coefficients'])
                    if residual_loss_weight > 0:
                        rterm = terms['stationarity_scaled_delta'].square().mean()
                        rweight = sample_weight
                    if constraint_loss_weight > 0:
                        cterm = terms['constraint_scaled_delta'].square().mean()
                        cweight = sample_weight
                    profile, approximation = terms['profile'], terms['approximation']
                    reason = 'FIXED_REFERENCE_VALID'
        residual_terms.append(rterm); constraint_terms.append(cterm)
        residual_weights.append(rweight); constraint_weights.append(cweight)
        residual_reasons.append(reason); residual_profiles.append(profile)
        residual_approximations.append(approximation)
    mden, jden = sum(moment_weights), sum(jacobian_weights)
    rden, cden = sum(residual_weights), sum(constraint_weights)
    if mden == 0 and jden == 0 and rden == 0 and cden == 0:
        raise ValueError('batch has no effective loss membership')
    zero = predictions[0].moments.sum()*0.
    mloss = sum((term*weight for term,weight in zip(moment_terms,moment_weights)),zero)/mden if mden else zero
    jloss = (sum((term*weight for term,weight in zip(jacobian_terms,jacobian_weights) if term is not None),zero)/jden
             if jden else zero)
    rloss = (sum((term*weight for term,weight in zip(residual_terms,residual_weights) if term is not None),zero)/rden
             if rden else zero)
    closs = (sum((term*weight for term,weight in zip(constraint_terms,constraint_weights) if term is not None),zero)/cden
             if cden else zero)
    mfinal = [weight/mden if mden else 0. for weight in moment_weights]
    jfinal = [jacobian_loss_weight*weight/jden if jden else 0. for weight in jacobian_weights]
    rfinal = [residual_loss_weight*weight/rden if rden else 0. for weight in residual_weights]
    cfinal = [constraint_loss_weight*weight/cden if cden else 0. for weight in constraint_weights]
    info = dict(moment_mask=[w>0 for w in mfinal],moment_weight=mfinal,
                jacobian_mask=[w>0 for w in jfinal],jacobian_weight=jfinal,
                moment_denominator=mden,jacobian_denominator=jden,
                moment_loss=float(mloss.detach()),jacobian_loss=float(jloss.detach()),
                jacobian_reasons=reasons,
                normalization='M: mL^degree; J: chart parameter_scales / (mL^degree)')
    info.update(residual_mask=[w>0 for w in rfinal],residual_weight=rfinal,
                constraint_mask=[w>0 for w in cfinal],constraint_weight=cfinal,
                residual_denominator=rden,constraint_denominator=cden,
                residual_loss=float(rloss.detach()),constraint_loss=float(closs.detach()),
                residual_reasons=residual_reasons,residual_profiles=residual_profiles,
                residual_approximations=residual_approximations,
                residual_normalization='fixed positive teacher scales; vector differences, not zero targets')
    return mloss+jacobian_loss_weight*jloss+residual_loss_weight*rloss+constraint_loss_weight*closs, info
