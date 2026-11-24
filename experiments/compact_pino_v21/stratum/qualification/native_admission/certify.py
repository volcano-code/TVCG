"""Exhaustive finite-machine source admission on a registered represented raw box."""
import argparse
from fractions import Fraction as F
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parents[1]/'native_backend'
PHYSICAL=HERE.parents[1]/'physical_replay'
sys.path.insert(0,str(PHYSICAL))
from checks import retracted


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def array_digest(x):
    return sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()


def floats(lo,hi):
    result=[]
    while lo<=hi:
        result.append(lo)
        lo=math.nextafter(lo,math.inf)
    return result


def ceil_machine(x):
    f=float(x)
    return math.nextafter(f,math.inf) if F(f)<x else f


def floor_machine(x):
    f=float(x)
    return math.nextafter(f,-math.inf) if F(f)>x else f


def possible_norms(rawlo,rawhi,probe):
    # The frozen native norm consists of squares followed by positive additions
    # and square root, monotonic in the magnitude of every input. Exact frozen
    # normalization at the two extremal magnitude vectors bounds every norm.
    lo=np.minimum(abs(rawlo),abs(rawhi));hi=np.maximum(abs(rawlo),abs(rawhi))
    lo=np.where((rawlo<=0)&(rawhi>=0),0.,lo)
    def denominator_range(x):
        values=probe.normalize(np.asarray(x))
        bounds=[]
        for a,y in zip(x,values):
            if a==0:continue
            assert a>0 and y>0
            lower=(F(math.nextafter(y,-math.inf))+F(y))/2
            upper=(F(math.nextafter(y,math.inf))+F(y))/2
            bounds.append((F(a)/upper,F(a)/lower))
        # N is binary64; use inclusive midpoint intervals regardless of tie parity.
        bottom=ceil_machine(max(t[0] for t in bounds));top=floor_machine(min(t[1] for t in bounds))
        if not bottom<=top:raise ValueError('unresolved native direct division rounding cell')
        return bottom,top
    nlo=denominator_range(lo);nhi=denominator_range(hi)
    return floats(nlo[0],nhi[1]),[float(t).hex() for t in (*nlo,*nhi)]


def full_rank_exact(columns):
    a=[[F(x) for x in row] for row in np.asarray(columns,dtype=object).T]
    pivot=0
    for j in range(len(a[0])):
        k=next((i for i in range(pivot,len(a)) if a[i][j]),None)
        if k is None:continue
        a[pivot],a[k]=a[k],a[pivot]
        v=a[pivot][j];a[pivot]=[x/v for x in a[pivot]]
        for i in range(len(a)):
            if i!=pivot:
                d=a[i][j];a[i]=[x-d*y for x,y in zip(a[i],a[pivot])]
        pivot+=1
    return pivot


def chart_witness(q,b,radius,rawlo,rawhi):
    # These reachable, linearly independent changes establish nontrivial
    # represented motion. Universality comes from the exhaustive cover below.
    chord=[];selected=[];counts=[]
    for d in b.T:
        for scale in [1.,.875,.75,.625,.5,.375,.25,.125]:
            t=radius*scale
            l=retracted(q,d,-t,'PARABOLOID');h=retracted(q,d,t,'PARABOLOID')
            if np.all(rawlo<=l) and np.all(l<=rawhi) and np.all(rawlo<=h) and np.all(h<=rawhi) and np.any(l!=h):
                chord.append([F(hi)-F(lo) for hi,lo in zip(h,l)])
                selected.append(t.hex());counts.append(int(np.count_nonzero(h!=l)));break
        else:raise ValueError('one supplied raw chart column has no resolved two-sided perturbation')
    rank=full_rank_exact(chord)
    if rank!=8:raise ValueError('represented chart chords do not span eight independent directions')
    return dict(two_sided_chord_rank=rank,perturbation_steps_hex=selected,
                changed_coordinates_per_chord=counts,chords_sha256=array_digest(chord))


def run(center,condition,probe,receipt,ulps):
    evidence=json.loads(Path(center).read_text()); geom=json.loads(Path(condition).read_text())
    row=evidence['row_id'];q=np.asarray(evidence['query'],dtype='f8');b=np.asarray(evidence['raw_basis'],dtype='f8')
    witness=evidence['native_value']['construction_witness']
    module=geom['native_backend_identity']['module_sha256']
    if (geom['row_id']!=row or geom['raw_query_sha256']!=array_digest(q) or
        geom['raw_basis_sha256']!=array_digest(b) or geom['evaluation_map_identity']!=evidence['context'] or
        geom['scope']!='R8' or geom['source_map']!=evidence['native_value']['source_map_version'] or
        module!=receipt['verified_source_module_sha256']):
        raise ValueError('mismatched registered source/geometry qualification')
    source=evidence['primal']
    for key, bound in [('target_vertices_sha256','vertices_sha256'),('target_faces_sha256','faces_sha256'),
                       ('cap_mask_sha256','cap_sha256'),('operation','operation')]:
        if geom[key]!=source[bound]:
            raise ValueError('conditional material geometry refers to another native operand')
    if (geom['triangulation_hash']!=evidence['native_value']['triangulation_hash'] or
        not np.array_equal(probe.normalize(q),witness['unit_coefficients'])):
        raise ValueError('native classifier evidence differs from acquired center')
    radius=max(math.ulp(x) for x in q)*ulps
    rawlo=np.asarray([ceil_machine(F(x)-F(radius)) for x in q])
    rawhi=np.asarray([floor_machine(F(x)+F(radius)) for x in q])
    if not all(F(lo)==F(x)-F(radius) and F(hi)==F(x)+F(radius) for lo,x,hi in zip(rawlo,q,rawhi)):
        raise ValueError('raw box endpoints must be exactly representable so real rounding remains inside')
    if np.any(rawlo==q) or np.any(rawhi==q):raise ValueError('raw radius is below input resolution')
    coeff_axes=[floats(l,h) for l,h in zip(rawlo,rawhi)]
    if min(map(len,coeff_axes))<5:raise ValueError('one raw axis has fewer than five represented query values')
    conditional=next((t for t in geom['trials'] if t.get('mixed_incidence',{}).get('certified') and
                      t.get('embedding',{}).get('embedding_contraction_upper',math.inf)<1 and
                      t.get('closure',{}).get('slope_lower',0)>0 and t['raw_box_linf_radius']>=radius),None)
    if conditional is None:raise ValueError('uniform conditional geometric proof does not contain the input box')
    norms,norm_endpoints=possible_norms(rawlo,rawhi,probe)
    if len(norms)>256 or min(norms)<=1e-15:raise ValueError('unsupported or unresolved normalization range')
    pivot=witness['coefficient_pivot'];sign=witness['coefficient_sign']
    if sign!=1:raise ValueError('audited entry covers the canonical-positive representative only')
    # Positive denominators preserve ordering. The margin also covers native
    # rounded division, the sign seam, and fromNormalized norm validation.
    axes=[sorted({sign*x/n for x in col for n in (norms[0],norms[-1])}) for col in coeff_axes]
    interval=np.array([[a[0],a[-1]] for a in axes])
    gap=min(abs(interval[pivot]))-max(abs(t) for i,r in enumerate(interval) if i!=pivot for t in r)
    if not (np.min(interval[pivot])>0 and gap>1e-9):raise ValueError('coefficient pivot or sign unstable')
    upper=np.max(abs(interval),axis=1);lower=np.maximum(np.maximum(interval[:,0],-interval[:,1]),0.)
    unit_range=[float(np.linalg.norm(lower))-2.e-14,float(np.linalg.norm(upper))+2.e-14]
    if unit_range[0]<1-9.e-13 or unit_range[1]>1+9.e-13:raise ValueError('native normalized-record guard unresolved')
    products=[[sorted({sign*x/n for x in col}) for col in coeff_axes[4:]] for n in norms]
    sizes=[math.prod(map(len,p)) for p in products];times=time.perf_counter()
    native=probe.cover(products,probe.normalize(q),np.ascontiguousarray(interval[1:4]),int(witness['least_eigenvalue_index']))
    if native['pair_count']!=sum(sizes) or native['distinct_matrices']+native['overlap_duplicates']!=sum(sizes):
        raise RuntimeError('incomplete enumeration')
    # Explicit comparison proving which two non-null eigenvalues and which
    # classifier tolerance apply to ALL unenumerated transverse coefficients.
    lo_ev=np.asarray(native['eigenvalue_bounds']);k=witness['least_eigenvalue_index']
    other=np.delete(lo_ev,k,axis=0)
    nonzero=np.maximum(other[:,0],-other[:,1])
    if (any(nonzero<=4.e-12) or native['maximum_abs_null_eigenvalue']>=1e-12 or
        native['minimum_transverse_abs']<=4.e-12):
        raise ValueError('rank classifier is not uniformly determined')
    diagram=chart_witness(q,b,radius,rawlo,rawhi)
    return dict(schema='tvcg.stratum.exhaustive-native-admission.v1',row_id=row,query_id=evidence['source_record']['query_id'],
        geometry_id=evidence['geometry_id'],parent_bank_sha256=evidence['source_record']['sha256'],
        raw_query_sha256=array_digest(q),raw_basis_sha256=array_digest(b),evaluation_map_identity=evidence['context'],
        native_module_sha256=module,source_map=geom['source_map'],scope='R8',
        uniform_native_value_admission=True,certified_raw_linf_radius=radius,raw_radius_hex=radius.hex(),
        radius_in_largest_raw_coefficient_ulps=ulps,raw_axes_hex=[[v.hex() for v in ab] for ab in zip(rawlo,rawhi)],
        exact_dyadic_endpoints=True,raw_axis_counts=list(map(len,coeff_axes)),normalizers_hex=[n.hex() for n in norms],
        two_endpoint_native_normalizer_enclosures_hex=norm_endpoints,normalized_guard_interval=unit_range,
        quadratic_product_shape_per_normalizer=[list(map(len,p)) for p in products],
        coefficient_pivot_gap_lower=math.nextafter(gap,-math.inf),signed_representative=sign,
        conditional_geometry_raw_linf_radius=conditional['raw_box_linf_radius'],
        conditional_geometry_record_sha256=digest(condition), center_record_sha256=digest(center),
        operation=geom['operation'],triangulation_hash=geom['triangulation_hash'],
        target_vertices_sha256=geom['target_vertices_sha256'],target_faces_sha256=geom['target_faces_sha256'],
        cap_mask_sha256=geom['cap_mask_sha256'],compiled_probe=receipt,checker_sha256=digest(__file__),cover=native,chart_resolution=diagram,
        numerical_domain='Every represented raw query in the stated box; in particular all matched chart inputs that lie in it.',
        timing_seconds=time.perf_counter()-times)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--center',type=Path,required=True)
    p.add_argument('--conditional',type=Path,required=True);p.add_argument('--ulps',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();receipt=json.loads((a.probe/'probe_receipt.json').read_text())
    sys.path.insert(0,str(a.probe)); import _rank_admission_probe as probe
    if (digest(probe.__file__)!=receipt['probe_sha256'] or digest(HERE/'probe.cpp')!=receipt['probe_cpp_sha256']
            or digest(HERE/'build_probe.py')!=receipt['probe_build_sha256']):
        raise ValueError('probe identity mismatch')
    doc=run(a.center,a.conditional,probe,receipt,a.ulps)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
    print(doc['row_id'],doc['raw_radius_hex'],doc['cover'],flush=True)
