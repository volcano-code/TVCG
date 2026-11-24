"""Event reporting for sampled probes. Whole-neighbourhood proofs have a separate entry."""
def assess_event(*, scope, center_signature=None, probe_signatures=(), finite_difference_passed=None):
    """Report probes at their actual scale. Observed changes do not localize an event to the centre."""
    if scope not in ('R3','R8','full'):
        raise ValueError('unknown derivative scope')
    if finite_difference_passed is not None and not isinstance(finite_difference_passed, bool):
        raise ValueError('finite difference status must be True, False or None')
    probes = tuple(probe_signatures)
    known = [p for p in probes if p is not None]
    if center_signature is None or not known:
        status = 'unmeasured'
    elif any(p != center_signature for p in known):
        status = 'observed_change'
    elif len(known) != len(probes):
        status = 'incomplete'
    else:
        status = 'all_observed_equal'
    equality={'observed_change':0,'all_observed_equal':1}.get(status,-1)
    return dict(scope=scope,event_valid=-1,sample_status=status,sampled_signature_equal=equality,measured_probes=len(known),
                requested_probes=len(probes), finite_difference_passed=finite_difference_passed,
                proof='unmeasured coefficient neighbourhood')


def match_center_qualification(record, center, *, proof_dir=None):
    """Join a verified certificate ledger to a physical center; identity mismatch is an error.

    Use the ledger read from the published checksum-bound file. The numerical
    row cannot sign its own event by providing an event flag in its packet.
    """
    import hashlib
    import numpy as np

    result=dict(ambient_event_valid=-1,restricted_event_valid=-1,native_output_triangulation_event_valid=-1)
    if record is None:
        return result
    def fingerprint(x):
        return hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()
    native=center.get('native_value') or {}
    primal=center.get('primal') or {}
    expected=dict(row_id=center['row_id'],geometry_id=center['geometry_id'],query_id=center['source_record']['query_id'],
                  parent_bank_sha256=center['source_record']['sha256'],scope='R'+str(center['dimension']),
                  raw_query_sha256=fingerprint(center['query']),raw_basis_sha256=fingerprint(center['raw_basis']),
                  evaluation_map_identity=center['context'])
    if 'target_faces_sha256' in record:
        expected.update(target_faces_sha256=primal.get('faces_sha256'),cap_mask_sha256=primal.get('cap_sha256'))
    if 'target_vertices_sha256' in record:
        expected.update(target_vertices_sha256=primal.get('vertices_sha256'),
                        operation=primal.get('operation'),triangulation_hash=native.get('triangulation_hash'),
                        source_map=native.get('source_map_version'))
    for name,value in expected.items():
        if value is None or value!=record.get(name):
            raise ValueError(f'qualification record differs in {name}')
    flag=record['restricted_event_valid']
    if flag not in (1,-1):
        raise ValueError('unrecognized certificate disposition')
    if flag==1:
        radius=record.get('certified_raw_linf_radius',0.)
        geometry_radius=radius
        if expected['scope']=='R8':
            try:
                from .admission import verify_joined
            except ImportError:
                from admission import verify_joined
            if proof_dir is None:
                verify_joined(record)
            else:
                verify_joined(record,proof_dir)
            geometry_radius=record['checked_source_raw_box_linf_radius']
        verified=any(t.get('raw_box_linf_radius')==geometry_radius and t.get('mixed_incidence',{}).get('certified') is True
                     and t.get('closure',{}).get('slope_lower',0.)>0
                     and 0<=t.get('embedding',{}).get('embedding_contraction_upper',float('inf'))<1
                     for t in record.get('trials',[]))
        if not (np.isfinite(radius) and 0<radius<=geometry_radius and verified and expected['scope'] in ('R3','R8')
                and 'target_vertices_sha256' in expected):
            raise ValueError('incomplete positive neighbourhood certificate')
        result.update(restricted_event_valid=1,certified_raw_linf_radius=radius,proof_scope=record['proof_scope'])
    return result
