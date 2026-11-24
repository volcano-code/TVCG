"""Index-independent component, edge-incidence and Euler counts of actual accepted center boundaries."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"native"))
from registry import save_json, sha


def closed_counts(n_vertices,n_faces):
    if n_faces%2:
        raise ValueError("a closed triangular surface must have an even total number of faces")
    edges=3*n_faces//2
    return dict(vertices=n_vertices,edges=edges,faces=n_faces,euler_characteristic=n_vertices-edges+n_faces)


def measured(vertices,faces):
    n=len(vertices)
    fr=faces.reshape(-1)
    to=faces[:,[1,2,0]].reshape(-1)
    edge=np.minimum(fr,to)*n+np.maximum(fr,to)
    code,counts=np.unique(edge,return_counts=True)
    if not np.all(counts==2):
        raise ValueError("retained boundary has an edge with valence different from two")
    adjacency=coo_matrix((np.ones(len(code),dtype=np.uint8),(code//n,code%n)),shape=(n,n)).tocsr()
    nc,labels=connected_components(adjacency,directed=False)
    component=labels[faces[:,0]]
    if not np.all(labels[faces]==component[:,None]):
        raise ValueError("triangle straddles disconnected boundary components")
    nv=np.bincount(labels,minlength=nc)
    nf=np.bincount(component,minlength=nc)
    ne=np.bincount(labels[code//n],minlength=nc)
    components=[]
    for i in range(nc):
        c=closed_counts(int(nv[i]),int(nf[i]))
        if c["edges"]!=ne[i]:
            raise ValueError("edge-accounting mismatch on retained component")
        components.append(c)
    return dict(closed_edge_valence_two=True,connected_components=nc,component_counts=components,
                **closed_counts(int(n),len(faces)))


def run(args):
    out={}
    for p in sorted((args.out/'centers').glob('*.json')):
        r=json.loads(p.read_bytes())
        m=r['material']
        if not m.get('moment_valid'):
            out[r['row_id']]=None
            continue
        path=args.out/m['retained_boundary_file']
        if sha(path)!=m['retained_boundary_sha256']:
            raise ValueError('retained boundary digest mismatch')
        with np.load(path,allow_pickle=False) as z:
            facts=measured(z['vertices'],z['faces'])
        out[r['row_id']]=dict(retained_boundary_sha256=sha(path),**facts)
    save_json(args.export,dict(schema="compact-v21.center-surface-topology.v1",rows=out))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--export',type=Path,required=True)
    run(p.parse_args())
