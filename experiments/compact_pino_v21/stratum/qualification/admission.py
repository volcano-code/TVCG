"""Check a frozen exhaustive R8 admission receipt and compose it with its original geometry proof."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
PROOF_DIR = HERE / 'native_admission' / 'evidence'
SCHEMA = 'tvcg.stratum.exhaustive-native-admission.v1'
MODULE = 'a2bc6ae680b50680930697136561ba9303a51bd8c0a3fc4118dc31617f99d9af'
IDENTITY = ('row_id', 'query_id', 'geometry_id', 'parent_bank_sha256', 'raw_query_sha256',
            'raw_basis_sha256', 'evaluation_map_identity', 'source_map', 'scope',
            'target_vertices_sha256', 'target_faces_sha256', 'cap_mask_sha256',
            'operation', 'triangulation_hash')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def raw_bytes(record):
    # Exact publisher serializer for a recomputed mixed-incidence row.
    return (json.dumps(record, indent=1, allow_nan=False) + '\n').encode()


def axis_values(bounds, count):
    left, right = map(float.fromhex, bounds)
    require(math.isfinite(left) and math.isfinite(right) and left * right > 0 and
            left < right and isinstance(count, int) and count >= 5, 'invalid raw axis')
    values = [left]
    while values[-1] < right and len(values) <= count:
        values.append(math.nextafter(values[-1], math.inf))
    require(len(values) == count and values[-1] == right, 'incomplete represented raw axis')
    return values


def check_cover(a, raw):
    """Recompute raw coordinate and normalization product counts without rerunning Eigen."""
    require(a['schema'] == SCHEMA and a['uniform_native_value_admission'] is True and
            a['native_module_sha256'] == raw['native_backend_identity']['module_sha256'] == MODULE,
            'unrecognized exhaustive native value certificate')
    require(all(raw[k] == a[k] for k in IDENTITY) and raw['scope'] == 'R8',
            'native cover belongs to a different finite operand or chart')
    require(sha(raw_bytes(raw)) == a['conditional_geometry_record_sha256'], 'different original geometry proof')
    radius = a['certified_raw_linf_radius']
    require(radius == float.fromhex(a['raw_radius_hex']) and math.isfinite(radius) and
            0 < radius <= a['conditional_geometry_raw_linf_radius'] == raw['checked_source_raw_box_linf_radius'],
            'native admission is outside the proved geometry box')
    require(len(a['raw_axes_hex']) == len(a['raw_axis_counts']) == 10, 'invalid raw box dimension')
    axes = [axis_values(pair, n) for pair, n in zip(a['raw_axes_hex'], a['raw_axis_counts'])]
    centers = []
    for axis in axes:
        lo, hi = Fraction(axis[0]), Fraction(axis[-1])
        mid = (lo + hi) / 2
        require(Fraction(float(mid)) == mid and mid - lo == hi - mid == Fraction(radius),
                'represented input box has different endpoints')
        centers.append(float(mid))
    require(sha(struct.pack('<10d', *centers)) == raw['raw_query_sha256'] and
            max(map(math.ulp, centers)) * a['radius_in_largest_raw_coefficient_ulps'] == radius,
            'represented input box differs from the registered center')
    normalizers = list(map(float.fromhex, a['normalizers_hex']))
    endpoints = list(map(float.fromhex, a['two_endpoint_native_normalizer_enclosures_hex']))
    require(normalizers and all(math.isfinite(n) and n > 0 for n in normalizers) and
            len(endpoints) == 4 and endpoints == sorted(endpoints) and
            normalizers[0] == endpoints[0] and normalizers[-1] == endpoints[-1] and
            all(y == math.nextafter(x, math.inf) for x, y in zip(normalizers, normalizers[1:])),
            'incomplete normalizer cover')
    shapes = [[len({v / n for v in axis}) for axis in axes[4:]] for n in normalizers]
    c = a['cover']
    require(shapes == a['quadratic_product_shape_per_normalizer'] and
            sum(math.prod(s) for s in shapes) == c['pair_count'] ==
            c['distinct_matrices'] + c['overlap_duplicates'] and
            c['distinct_matrices'] > 0 and c['overlap_duplicates'] >= 0, 'incomplete quadratic classifier cover')
    unit_lo, unit_hi = a['normalized_guard_interval']
    require(1 - 1.e-12 < unit_lo <= unit_hi < 1 + 1.e-12 and a['signed_representative'] == 1 and
            a['coefficient_pivot_gap_lower'] > 1.e-9 and 0 <= c['maximum_native_leaf_fraction'] < 1 and
            c['minimum_native_leaf_margin'] > 0 and c['minimum_transverse_abs'] > 4.e-12 and
            c['maximum_abs_null_eigenvalue'] < 1.e-12 and c['minimum_eigengap'] > 1.e-10 and
            c['minimum_eigenvector_pivot_gap'] > 1.e-10 and
            0 <= c['conservative_residual_upper'] < 2.**-42 and
            0 <= c['conservative_orthogonality_upper'] < 2.**-42, 'native admission branch lacks strict bounds')
    d = a['chart_resolution']
    require(d['two_sided_chord_rank'] == len(d['perturbation_steps_hex']) ==
            len(d['changed_coordinates_per_chord']) == 8 and
            all(0 < float.fromhex(t) <= radius for t in d['perturbation_steps_hex']) and
            min(d['changed_coordinates_per_chord']) > 0, 'missing resolved eight-dimensional chart')
    probe = a['compiled_probe']
    source = HERE / 'native_admission'
    require(probe['verified_source_module_sha256'] == MODULE and
            sha((source / 'build_probe.py').read_bytes()) == probe['probe_build_sha256'] and
            sha((source / 'probe.cpp').read_bytes()) == probe['probe_cpp_sha256'] and
            sha((source / 'certify.py').read_bytes()) == a['checker_sha256'], 'different native proof program')
    return radius


def combine(raw, path):
    """Create an identifiable, reproducible supplement for a still-conditional geometry row."""
    payload = path.read_bytes()
    a = json.loads(payload)
    radius = check_cover(a, raw)
    return dict(receipt_sha256=sha(payload), filename=path.name, original_geometry_record=raw,
                certified_raw_linf_radius=radius)


def verify_joined(record, proof_dir=PROOF_DIR):
    link = record.get('uniform_native_admission')
    require(isinstance(link, dict) and link['filename'] == record['row_id'] + '.json',
            'missing linked exhaustive native admission')
    data = (proof_dir / link['filename']).read_bytes()
    require(sha(data) == link['receipt_sha256'], 'changed exhaustive native admission file')
    a = json.loads(data)
    original = link['original_geometry_record']
    require(all(original[k] == a[k] == record[k] for k in IDENTITY) and
            original['native_backend_identity']['module_sha256'] == record['native_module_sha256'] and
            original['trials'] == record['trials'] and original['material_mesh_vertex_sha256'] ==
            record['material_mesh_vertex_sha256'] and original['material_mesh_faces_sha256'] ==
            record['material_mesh_faces_sha256'] and original['checked_source_envelope_incidence'] is True and
            original['restricted_event_valid'] == -1, 'conditional material record changed')
    radius = check_cover(a, original)
    require(radius == record['certified_raw_linf_radius'] == link['certified_raw_linf_radius'] and
            a['conditional_geometry_raw_linf_radius'] == record['checked_source_raw_box_linf_radius'],
            'different combined proof radius')
    return radius
