"""Independent denominators for Moment and admitted response-valid Jacobians."""
from __future__ import annotations

import torch

from ..p0_diagnostics.baseline_cases import moment_scale


def physical_loss(predictions, records, *, jacobian_loss_weight=1.):
    """Return differentiable loss and the exact final membership/weights.

    Per-example M error averages ten squared errors scaled by m*L**degree.
    Per-example J error averages 10*D squared errors, also multiplied by the
    explicit chart parameter scales. Each objective divides by its own sum of
    admitted positive example weights, NEVER by the batch's missing J labels.
    Audit weights are final coefficients multiplying those per-example means.
    """
    if len(predictions) != len(records) or not records:
        raise ValueError('predictions and nonempty record batch must align')
    if jacobian_loss_weight < 0:
        raise ValueError('nonnegative Jacobian loss coefficient required')
    moment_terms, jacobian_terms, moment_weights, jacobian_weights = [], [], [], []
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
    mden, jden = sum(moment_weights), sum(jacobian_weights)
    if mden == 0 and jden == 0:
        raise ValueError('batch has no effective loss membership')
    zero = predictions[0].moments.sum()*0.
    mloss = sum((term*weight for term,weight in zip(moment_terms,moment_weights)),zero)/mden if mden else zero
    jloss = (sum((term*weight for term,weight in zip(jacobian_terms,jacobian_weights) if term is not None),zero)/jden
             if jden else zero)
    mfinal = [weight/mden if mden else 0. for weight in moment_weights]
    jfinal = [jacobian_loss_weight*weight/jden if jden else 0. for weight in jacobian_weights]
    info = dict(moment_mask=[w>0 for w in mfinal],moment_weight=mfinal,
                jacobian_mask=[w>0 for w in jfinal],jacobian_weight=jfinal,
                moment_denominator=mden,jacobian_denominator=jden,
                moment_loss=float(mloss.detach()),jacobian_loss=float(jloss.detach()),
                jacobian_reasons=reasons,
                normalization='M: mL^degree; J: chart parameter_scales / (mL^degree)')
    return mloss+jacobian_loss_weight*jloss, info
