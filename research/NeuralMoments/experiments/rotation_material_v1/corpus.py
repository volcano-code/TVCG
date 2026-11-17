"""Small analytic variable-density fixtures, separate from M83 and benchmarking."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def cell_moments(lower, upper, density):
    """Exact zero-to-second moments for disjoint constant-density cuboids.

    Empty clipped cells (upper<=lower in any dimension) contribute zero.
    This independent label routine does not call PINO, its Gauss rule or M69.
    """
    width=np.maximum(upper-lower,0.)
    mass=width.prod(-1)*density
    center=(lower+upper)/2
    x,y,z=center.T
    second=center**2+width**2/12
    basis=np.stack((np.ones_like(x),x,y,z,*second.T,y*z,x*z,x*y),-1)
    return (mass[:,None]*basis).sum(0)


def generate_corpus(target):
    target=Path(target);target.mkdir(parents=True,exist_ok=False)
    records=[]
    index=np.indices((8,8,8)).reshape(3,-1).T
    fixed=((index==0)|(index==7)).any(-1); editable=~fixed
    normalized=(index+.5)/8
    for i in range(12):
        extent=np.array([1.1+.07*i,1.5+.03*i,1.2+.05*i])
        lower=-extent/2+np.array([.02*i,-.01*i,.03*i])
        upper=lower+extent
        cell_low=lower+index*extent/8;cell_high=cell_low+extent/8
        base=1.3+.4*normalized[:,0]+.2*normalized[:,2]+.05*i
        beta=.15+.7*(1+np.tanh(2*(normalized[:,0]-.5)+np.sin((i+1)*.3+normalized[:,1]*3)-normalized[:,2]))/2
        density=base*beta*editable
        fixed_density=np.full(512,2.+.04*i)
        F=cell_moments(cell_low,cell_high,fixed_density*fixed)
        original=F+cell_moments(cell_low,cell_high,base*editable)
        full=F+cell_moments(cell_low,cell_high,density)
        queries=np.zeros((5,10));queries[0,0]=1;queries[1,0]=-1
        moments=[full,F]
        for axis in range(3):
            threshold=lower[axis]+(.35+.1*axis)*extent[axis]
            queries[axis+2,0]=-threshold;queries[axis+2,axis+1]=1
            clipped=cell_low.copy();clipped[:,axis]=np.maximum(clipped[:,axis],threshold)
            moments.append(F+cell_moments(clipped,cell_high,density))
        path=target/f'case_{i:02d}.npz'
        np.savez_compressed(path,density=density.reshape(8,8,8),base_density=base.reshape(8,8,8),
                            fixed_density=fixed_density.reshape(8,8,8),fixed=fixed.reshape(8,8,8),editable=editable.reshape(8,8,8),
                            lower=lower,upper=upper,length=np.array(extent.max()),density_reference=np.array(3.),
                            raw_coefficients=queries,moments=np.stack(moments),fixed_moments=F,original_moments=original)
        records.append({'id':f'synthetic-density-{i:02d}','file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'split':'train' if i<8 else 'validation' if i<10 else 'held_out',
                        'query_count':5,'density_min_editable':float(density[editable].min()),'density_max_editable':float(density[editable].max())})
    manifest={'schema':'spin.synthetic-variable-density.v1','purpose':'diagnostic_schema_and_labels_only',
              'generator':'rotation_material_v1/corpus.py','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'origin':'deterministic synthetic Cartesian boxes, not GSO/Procedural populations or TRELLIS output',
              'grid':[8,8,8],'material_policy':'bounded_variable_density_relaxation.v1',
              'labels':'exact constant-cell cuboid integrals; full/empty and axis-aligned HARD retained halfspaces; fixed support added once',
              'reference':'original_moments use unmodified base density; moments[0] uses current variable density',
              'moment_order':['mass','x','y','z','xx','yy','zz','yz','xz','xy'],
              'query_order':['1','x','y','z','xx','yy','zz','2yz','2xz','2xy'],
              'model_evaluated_splits':[],'held_out_model_evaluation':False,'records':records}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=generate_corpus(args.output)
    print(json.dumps({'output':str(args.output.resolve()),'records':len(result['records']),'optimizer_steps':0}))
