"""Serialize the registered strict native event payloads in their original C++ byte format."""
import hashlib
import json
import math


def hexfloat(value):
    mantissa,exp=float(value).hex().split('p')
    if '.' in mantissa:
        mantissa=mantissa.rstrip('0').rstrip('.')
    return mantissa+'p'+exp


def raw_events_bytes(events):
    def string(s):
        return str(len(s.encode()))+':'+s
    def number(v):
        return 'none' if v is None else 'some:'+hexfloat(v)
    def optional_string(s):
        return 'none' if s is None else 'some:'+string(s)
    if not events or any(e['equation_profile']!='DUAL_MONOMIAL_RAW_V1' for e in events):
        raise ValueError('only the registered strict event stream is supported')
    capture=any('evaluation_payload' in e or 'newton_system_payload' in e for e in events)
    out=['SpinSolverEvents-v6-capture-v1' if capture else 'SpinSolverEvents-v2',events[0]['equation_profile'],
         events[0]['solver_coordinate_identity']]
    for e in events:
        out.extend([number(e['raw_equation_residual']),string(e['type']),str(e['restart']),str(e['iteration']),
                    str(e['evaluation_index']),str(e['restart_evaluation_index']),number(e['step']),
                    string(e['outcome']),string(e['reason']),string(e['detail'])])
        q=e['quadric_coefficients']
        if q is None:
            out.append('none')
        else:
            out.extend(['some',*(hexfloat(v) for v in q)])
        out.extend([string(e['quadric_hash']),number(e['normalized_equation_residual']),
                    number(e['physical_relative_residual']),number(e['objective']),number(e['multiplier_mu']),
                    optional_string(e['material_geometry_hash']),optional_string(e['material_topology_hash'])])
        if not capture:
            continue
        p=e.get('evaluation_payload')
        if p is None:
            out.append('payload:none')
        else:
            out.append('payload:'+p['schema'])
            for k in ('solver_coordinates','solver_raw_moments'):
                out.extend(number(v) for v in p[k])
            variable=p.get('variable_raw_moments')
            out.append('variable:some' if variable is not None else 'variable:none')
            if variable is not None:
                out.extend(number(v) for v in variable)
            jac=p.get('solver_moment_jacobian')
            out.append('jacobian:some' if jac is not None else 'jacobian:none')
            if jac is not None:
                out.extend(number(v) for row in jac for v in row)
            out.append('defined' if p['derivative_defined'] else 'undefined')
            out.extend(optional_string(p[k]) for k in ('jacobian_coordinate_identity','evaluation_map_identity',
                       'boolean_backend_identity','derivative_identity','derivative_diagnostic','geometry_hash','topology_hash'))
            out.extend(number(p[k]) for k in ('evaluator_wall_seconds','residual_evaluation_seconds','derivative_evaluation_seconds'))
        system=e.get('newton_system_payload')
        if system is None:
            out.append('newton-system:none')
        else:
            m=system['matrix']
            out.extend(['newton-system:'+system['schema'],str(len(m)),str(len(m[0])),str(system['computed_rank']),
                        str(system['equilibrated_rank'])])
            out.extend(number(v) for row in m for v in row)
            rhs=system['right_hand_side']
            out.extend([str(len(rhs)),*(number(v) for v in rhs),number(system['multiplier_mu'])])
            for k in ('constraint_multipliers','row_scales','column_scales'):
                x=system[k]
                out.extend([str(len(x)),*(number(v) for v in x)])
            out.extend(number(system[k]) for k in ('linear_residual_infinity_norm','linear_backward_error'))
    return (';'.join(out)+';').encode()


def raw_event_digest(events):
    return hashlib.sha256(raw_events_bytes(events)).hexdigest()


def scientific_event(event):
    """Normalize documented stopwatch fields and the Python stack of the registered cone rejection."""
    item = dict(event)
    if 'evaluation_payload' in item:
        payload=dict(item['evaluation_payload'])
        for k in ('evaluator_wall_seconds','residual_evaluation_seconds','derivative_evaluation_seconds'):
            if not math.isfinite(payload[k]) or payload[k]<0:
                raise ValueError('invalid stopwatch in native ledger')
            payload[k]=0.
        item['evaluation_payload']=payload
    message, separator, stack = item['detail'].partition('\n\nAt:\n')
    if separator and 'UNSUPPORTED_PAPER_IDEAL_FACE: no negative cone reference' in message:
        if not all('.py(' in line and '): ' in line for line in stack.splitlines() if line.strip()):
            raise ValueError('unrecognized traceback in native event ledger')
        item['detail']=message
    return item


def compare(direct, versioned):
    if direct['entry']!='legacy_direct' or versioned['entry']!='versioned':
        raise ValueError('expected the two actual native entry paths')
    left,right=direct['events'],versioned['events']
    differences=[]
    for index,(a,b) in enumerate(zip(left,right,strict=True)):
        if a!=b:
            keys=sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
            diff=dict(index=index,keys=keys)
            if 'evaluation_payload' in keys:
                pa,pb=a['evaluation_payload'],b['evaluation_payload']
                diff['payload_fields']=sorted(k for k in set(pa)|set(pb) if pa.get(k)!=pb.get(k))
            if 'detail' in keys:
                diff.update(direct_detail=a['detail'],versioned_detail=b['detail'])
            differences.append(diff)
    summaries=[]
    for run in (direct,versioned):
        if raw_event_digest(run['events'])!=run['event_hash']:
            raise ValueError('native event byte stream differs from its hash')
        normalized=[scientific_event(e) for e in run['events']]
        encoded=json.dumps(normalized,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
        summaries.append(dict(entry=run['entry'],events=len(normalized), event_hash=run['event_hash'],
                              scientific_event_sha256=hashlib.sha256(encoded).hexdigest(),
                              same_archived_event_hash=run['same_archived_event_hash']))
    same=summaries[0]['scientific_event_sha256']==summaries[1]['scientific_event_sha256']
    physical_fields=('objective','mesh_geometry_sha256','mesh_topology_sha256','raw_residual','mu',
                     'physical_relative_residual','constraints','exact_ten_moments')
    same_acceptance=all(direct['physical_check'][k]==versioned['physical_check'][k] for k in physical_fields)
    valid=(all(x['accepted'] and x['same_trace_hash'] and x['all_rows_match'] for x in (direct,versioned))
           and same and same_acceptance)
    return dict(entries=summaries,passed=valid,all_numerical_events_match=same,
                       exclusions='Three recorded wall-time fields and the Python stack text of the identical third-call '
                       'cone refusal only. All numerical payloads, Newton systems, sides, material hashes and decisions are compared.',
                       same_physical_acceptance=same_acceptance,raw_event_differences=differences)
