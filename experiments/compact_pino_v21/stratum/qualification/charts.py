"""Enclose the versioned graph, its branch decisions and cap geometry on a raw R3 chart."""
import itertools
import numpy as np
if __package__:
    from .bounds import Interval as I, cross, dot, norm, stack, lower, upper
else:
    from bounds import Interval as I, cross, dot, norm, stack, lower, upper


MAP = 'ALGEBRAIC_FINITE_GRAPH_EXTRUSION_R3_R8_V1'
SIDE = 'canonical-graph-positive-extrusion-exact-oriented-boundary-v1'


def require(condition, reason):
    if not np.all(condition):
        raise ValueError(reason)


def _rounded_norm(v):
    # Eigen may sum squares in a different order. For nonnegative products an
    # n-term binary64 reduction has relative error <= n*u/(1-n*u).
    squares=v.square()
    total=squares[0]
    for i in range(1,len(v.lo)):
        total=total+squares[i]
    slack=32*np.finfo('f8').eps
    t=total*I(1-slack,1+slack)+I(-1.e-300,1.e-300)
    return I(max(float(t.lo),0.),t.hi).sqrt()


def canonical(q, radius, plane):
    require(q.shape==(10,) and np.isfinite(q).all(),'invalid raw coefficient center')
    if plane:
        require(np.all(q[4:]==0),'center is outside the plane chart')
        linear = I.ball(q[:4],radius)
        raw = I(np.r_[linear.lo,np.zeros(6)],np.r_[linear.hi,np.zeros(6)])
    else:
        raw=I.ball(q,radius)
    n = _rounded_norm(raw)
    require(n.lo>1.e-15,'possible normalization branch')
    unit = raw/n
    pivot = int(np.argmax(abs(q)))
    other = np.delete(unit.abs_upper(),pivot).max()
    gap = (I.point(unit.abs_lower()[pivot])-other).lo
    require(gap>1.e-10,'canonical coefficient pivot may switch')
    sign=int(unit[pivot].sign())
    require(sign!=0,'canonical coefficient sign may switch')
    return unit*sign,dict(pivot=pivot,sign=sign,pivot_gap_lower=float(gap))


def enclose_plane(q, raw_radius, patch_radius):
    q, facts = canonical(np.asarray(q),raw_radius,True)
    b=q[1:4]
    nlen = _rounded_norm(b)
    require(nlen.lo>1.e-12,'plane classification may be singular')
    axis=b/nlen
    if axis.abs_upper()[0] < .9-1.e-10:
        k=0
    elif axis.abs_lower()[0] > .9+1.e-10:
        k=1
    else:
        raise ValueError('plane reference axis may change')
    first=cross(axis,np.eye(3)[k])
    first=first/_rounded_norm(first)
    second=cross(axis,first)
    bn=dot(b,axis)
    require(bn.lo>1.e-12,'plane graph has no increasing transverse axis')
    s=np.asarray([-patch_radius,patch_radius,patch_radius,-patch_radius])
    t=np.asarray([-patch_radius,-patch_radius,patch_radius,patch_radius])
    y=first*s[:,None]+second*t[:,None]
    w=-(q[0]+dot(b,y))/bn
    # The native graph rounds each exact algebraic coordinate only after
    # composing the graph expression. Outward intervals include this rounding.
    points=y+axis*w[:,None]
    facts.update(axis=axis,first=first,second=second,bn=bn,an=I.point(np.zeros(3)),
                 coefficient_pivot=facts['pivot'],frame_reference_axis=k,
                 s=s,t=t,axial_lower=float(bn.lo))
    return points,facts


def target_box_corners(design):
    xyz=np.asarray(design,dtype=np.float64)
    require(xyz.ndim==2 and xyz.shape[1]==3 and len(xyz)>0 and np.isfinite(xyz).all(),'invalid target design domain')
    return np.asarray(list(itertools.product(*zip(xyz.min(axis=0),xyz.max(axis=0),strict=True))))


def _support(axis, corners):
    vals=dot(axis,corners)
    return I(vals.lo.min(),vals.hi.min()),I(vals.lo.max(),vals.hi.max())


def _layers(near,axis,depth):
    far=near + axis*depth
    return I(np.concatenate([near.lo,far.lo]),np.concatenate([near.hi,far.hi]))


def projected_rim_guard(near,frame,corners,margin):
    """Certify simple projected rim and strict in-plane separation of its edges from the design hull."""
    n=frame['axis']
    u,v=frame['first'],frame['second']
    m=int(round(np.sqrt(len(near.lo))))-1
    ids=(np.array([[0,1],[3,2]]) if len(near.lo)==4 else np.arange(len(near.lo)).reshape(m+1,m+1))
    rim=np.r_[ids[0],ids[1:,-1],ids[-1,-2::-1],ids[-2:0:-1,0]]
    a,b=near[rim],near[np.roll(rim,-1)]
    e=b-a
    normal=cross(n,e)
    side=dot(normal[:,None],I.point(corners)-a[:,None])
    signs=side.sign()
    require(np.all(signs==1) or np.all(signs==-1),'projected design hull may leave the rim kernel')
    dist=I.point(side.abs_lower())/norm(normal)[:,None]
    require(dist.lo.min()>margin,'projected hull-to-rim separation unresolved')
    nsq=dot(n,n)
    U=u-n*(dot(n,u)/nsq)
    V=v-n*(dot(n,v)/nsq)
    pa,pb=stack([dot(a,U),dot(a,V)]),stack([dot(b,U),dot(b,V)])
    def area(x,y):
        return x[...,0]*y[...,1]-x[...,1]*y[...,0]
    d=pb-pa
    for i in range(len(rim)):
        j=(i+1)%len(rim)
        # Consecutive segments meet only at their shared endpoint.
        forward=d[i]*d[j]
        require((forward[0]+forward[1]).lo>0 or area(d[i],d[j]).sign()!=0,
                'adjacent rim edges may overlap')
        idx=np.arange(i+2,len(rim)-(1 if i==0 else 0))
        if not len(idx):
            continue
        lo=np.minimum(pa.lo,pb.lo)
        hi=np.maximum(pa.hi,pb.hi)
        candidates=idx[np.all((lo[idx]<=hi[i])&(hi[idx]>=lo[i]),axis=1)]
        if len(candidates):
            j=candidates
            s1=area(d[i],pa[j]-pa[i]).sign()*area(d[i],pb[j]-pa[i]).sign()
            s2=area(d[j],pa[i]-pa[j]).sign()*area(d[j],pb[i]-pa[j]).sign()
            require((s1>0)|(s2>0),'nonadjacent projected rim segments may touch')
    return float(np.min(dist.lo))


def finite_guards(near,frame,radius,doublings,design,faces,interface_faces):
    """Verify the exact native support, corner, orientation and radius-selection inequalities."""
    axis=frame['axis']
    corners=target_box_corners(design)
    margin=1.e-10*max(1.,float(np.max(abs(corners))))
    projected_clearance=projected_rim_guard(near,frame,corners,margin)
    _,axmax=_support(axis,corners)
    slope=frame['bn']+2.*_support(frame['an'],corners)[0]
    require(slope.lo>0.,'axial selection may reverse inside design bounds')
    require(type(doublings) is int and 0<=doublings<=8,'cap growth outside pinned settings')
    growth=[]
    for k in range(doublings+1):
        depth=np.ldexp(4.*radius,k)
        vertices=_layers(near,axis,depth)
        far_gap=dot(axis,vertices[len(near.lo):])-axmax
        growth.append(dict(depth=float(depth),minimum_far_gap_lower=float(far_gap.lo.min()),
                           minimum_far_gap_upper=float(far_gap.hi.min())))
        if k<doublings:
            require(np.min(far_gap.hi)<margin,'an earlier cap-growth choice may be accepted')
        else:
            require(np.min(far_gap.lo)>margin,'far cap separation unresolved')
    slopes=frame['bn']+2.*dot(frame['an'],vertices)
    require(np.min(slopes.lo)>1.e-12,'source slope or motion guard unresolved')
    front=vertices[np.asarray(faces)[:interface_faces]]
    winding=dot(cross(front[:,1]-front[:,0],front[:,2]-front[:,0]),axis)
    require(np.max(winding.hi)<0,'interface winding unresolved')
    laterals=vertices[np.asarray(faces)[2*interface_faces:]]
    require(len(laterals.lo)>0,'missing lateral cap faces')
    def gap(direction):
        v=dot(direction,laterals)
        lo,hi=_support(direction,corners)
        return np.maximum((lo-I.point(v.hi.max(axis=1))).lo,(I.point(v.lo.min(axis=1))-hi).lo)
    clearance=float(np.min(np.maximum(gap(frame['first']),gap(frame['second']))))
    require(clearance>margin,'lateral cap separation unresolved')
    facts=dict(growth=growth,contact_clearance_threshold=margin,projected_rim_clearance_lower=projected_clearance,far_cap_clearance_lower=float(np.min(far_gap.lo)),lateral_cap_clearance_lower=clearance,
               slope_lower=float(np.min(slopes.lo)),front_winding_abs_lower=float(-np.max(winding.hi)),
               cap_doublings=doublings,initial_radius=float(radius))
    return facts,vertices


def physical_box(vertices,axes,offset,length,normalization_origin):
    # The four registered development frames have exact identity target axes.
    # A dot with these axes has one nonzero term; the remaining products are zero.
    require(np.array_equal(axes,np.eye(3)),'unverified target-axis arithmetic')
    require(np.shape(offset)==(3,) and np.shape(normalization_origin)==(3,) and length>0,'invalid physical transform')
    return (vertices+offset)*length+normalization_origin


def verify_prism_embedding(vertices, faces, caps, frame, center, depth, radius):
    """Bound a piecewise-affine extension on the entire box reference prism."""
    n=len(center)//2
    n_grid=int(round(np.sqrt(n)))-1
    require(n_grid>=1 and (n_grid+1)**2==n and len(center)==2*n,'unrecognized graph lattice')
    plane = n==4 and frame.get('frame_reference_axis') is not None
    s,t=frame['s'],frame['t']
    # R8's grid arithmetic is fixed by the binary64 radius. Use the direction
    # coordinates actually supplied to the exact root, with a conservative
    # half-spacing bound to cover the rounding of the nominal uniform knots.
    knots_s=np.unique(s)
    knots_t=np.unique(t)
    require(len(knots_s)==len(knots_t)==n_grid+1,'malformed graph knots')
    min_step=min(float((I.point(knots_s[1:])-knots_s[:-1]).lo.min()),
                 float((I.point(knots_t[1:])-knots_t[:-1]).lo.min()),float(depth))
    require(min_step>0,'zero lattice spacing')
    ids=np.array([[0,1],[3,2]]) if plane else np.arange(n).reshape(n_grid+1,n_grid+1)
    ll,lr,ul,ur=ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1],ids[1:,1:]
    tris=np.vstack([np.stack([ll,lr,ur],axis=-1).reshape(-1,3),
                    np.stack([ll,ur,ul],axis=-1).reshape(-1,3)])
    as_keys=lambda f:{tuple(sorted(map(int,a))) for a in f}
    base_keys=as_keys(tris)
    expected_faces=2*len(tris)+8*n_grid
    nearf=faces[:len(tris)]
    farf=faces[len(tris):2*len(tris)]
    require(len(faces)==expected_faces and as_keys(nearf)==base_keys
            and as_keys(farf-n)==base_keys and np.all(~caps[:len(tris)])
            and np.all(caps[len(tris):]),'native graph layer face layout differs from proof')
    perimeter = np.r_[ids[0],ids[1:,-1],ids[-1,-2::-1],ids[-2:0:-1,0]]
    side=as_keys(faces[2*len(tris):])
    for p,q in zip(perimeter,np.roll(perimeter,-1),strict=True):
        quad = {p,q,p+n,q+n}
        parts = [set(f) for f in side if set(f)<=quad]
        require(len(parts)==2 and set.union(*parts)==quad and
                set.intersection(*parts) in ({p,q+n},{q,p+n}),'side does not triangulate one prism edge')
    # The center directions define a fixed affine comparison map; every
    # nonzero tolerance below is derived from the interval enclosure.
    def middle(v):
        return v.lo*.5+v.hi*.5
    u,v,nv=(middle(frame[k]) for k in ('first','second','axis'))
    det=dot(cross(I.point(u),I.point(v)),I.point(nv))
    require(det.abs_lower()>.9 and np.max(abs(np.r_[u,v,nv]))<1.02,'comparison frame inverse bound fails')
    # With |det|>.9, each inverse cofactor is at most 2*1.02^2, giving
    # ||U^-1||_infinity < 8. The height values can be arbitrary dyadic numbers.
    y=s[:,None]*u+t[:,None]*v
    h=((center[:n]-y)@nv)/(nv@nv)
    hgrid = h[ids]
    dhs=(I.point(hgrid[:,1:])-hgrid[:,:-1]).abs_upper().max()
    dht=(I.point(hgrid[1:])-hgrid[:-1]).abs_upper().max()
    h_lipschitz=(I.point(dhs)+dht)/min_step
    baseline=I.point(u)*s[:,None]+I.point(v)*t[:,None]+I.point(nv)*h[:,None]
    cmp=_layers(baseline,I.point(nv),depth)
    delta=(vertices-cmp).abs_upper().max()
    error_lipschitz=I.point(108.)*delta/min_step
    inverse=8.*(1.+h_lipschitz)
    ratio=error_lipschitz*inverse
    require(ratio.hi<1.,'enclosure too wide to certify embedded finite boundary')
    return dict(embedding_contraction_upper=float(ratio.hi),max_corner_deviation_upper=float(delta),
                graph_reference_spacing_lower=min_step)


def _mul_matrix(a, v):
    return stack([dot(row,v) for row in a])


def enclose_checked_paraboloid(q, raw_radius, radius, witness):
    """Enclose every *accepted* native eigenframe on the raw box using exact backend residual guards.

    Eigensolver termination and the roundoff-leaf acceptance guard throughout a
    chart are separate obligations. This enclosure alone is never a positive
    coefficient-neighbourhood certificate.
    """
    coeff,frame=canonical(np.asarray(q),raw_radius,False)
    a=[coeff[[4,9,8]],coeff[[9,5,7]],coeff[[8,7,6]]]
    b=coeff[1:4]
    ev=np.asarray(witness['canonical_quadratic_eigenvalues'],float)
    k=int(witness['least_eigenvalue_index'])
    bound=float(witness['eigen_verification_bound'])
    require(bound==2.**-42 and 0<=k<3 and len(ev)==3 and np.all(np.diff(ev)>0),
            'missing pinned exact native eigensystem guards')
    others=[j for j in range(3) if j!=k]
    reference={k:witness['axis'],others[0]:witness['first'],others[1]:witness['second']}
    true_lam=[]
    residuals=[]
    normalized=[]
    for j in range(3):
        d=I.point(reference[j])
        norm0=norm(d)
        require(norm0.lo>.99 and norm0.hi<1.01,'invalid acquisition eigenvector')
        v=d/norm0
        # Relative to a unit vector, a symmetric matrix has at least one
        # eigenvalue in the residual interval about this trial eigenvalue.
        residual=norm(_mul_matrix(a,v)-v*ev[j])
        residuals.append(residual)
        true_lam.append(I.ball(ev[j],float(residual.hi)))
        normalized.append(v)
    gap=min((true_lam[j+1]-true_lam[j]).lo for j in (0,1))
    require(gap>max(1.e-10,1000*bound),'distinct certified eigenspaces unresolved')
    eta=3*bound
    native_lam=[v+I(-eta,eta) for v in true_lam]
    require(native_lam[k].abs_upper()<min(native_lam[i].abs_lower() for i in others),
            'least-eigenvalue branch may switch')
    enclosed=[]
    for j in range(3):
        # The residual projected onto the other exact eigenspaces bounds sin(theta).
        # For equal orientation ||u-v|| <= 2*sin(theta). The measured native
        # vectors have norm within 2*K of one. Two K-orthogonal reported vectors
        # cannot map to a single exact eigenspace when gap > 1000*K.
        rad=2.*(residuals[j]/gap+I.point(eta)/(I.point(gap)-eta))+2*bound
        v=normalized[j]+I(-float(rad.hi),float(rad.hi))
        if j!=k:
            pivot=int(np.argmax(np.abs(reference[j])))
            require(v[pivot].lo>np.delete(v.abs_upper(),pivot).max()+1.e-10,
                    'canonical eigenvector sign not separated')
        enclosed.append(v)
    axis,u,v=enclosed[k],enclosed[others[0]],enclosed[others[1]]
    bn=dot(b,axis)
    require(bn.lo>1.e-12,'axis orientation or graph classification unresolved')
    an=_mul_matrix(a,axis)
    alpha=dot(axis,an)
    res=int(witness['paraboloid_grid_resolution'])
    require(res==24,'unregistered grid resolution')
    # Same binary64 operations and order as the pinned C++ grid, constant in q.
    knots=np.array([-radius+2.*radius*i/res for i in range(res+1)])
    s,t=np.meshgrid(knots,knots,indexing='xy')
    s,t=s.ravel(),t.ravel()
    y=u*s[:,None]+v*t[:,None]
    ay=_mul_matrix(a,y)
    linear=bn+2.*dot(y,an)
    constant=coeff[0]+dot(b,y)+dot(y,ay)
    discr=linear.square()-4.*alpha*constant
    require(linear.lo>0,'positive axial graph coefficient unresolved')
    require(discr.lo>0,'regular real graph root unresolved')
    h=(-2.*constant)/(linear+discr.sqrt())
    pts=y+axis*h[:,None]
    frame.update(axis=axis,first=u,second=v,bn=bn,an=an,s=s,t=t,
                 eigenvalue_gap_lower=float(gap),eigen_check_bound=bound,
                 graph_discriminant_lower=float(discr.lo.min()),
                 coefficient_pivot=frame['pivot'])
    return pts,frame
