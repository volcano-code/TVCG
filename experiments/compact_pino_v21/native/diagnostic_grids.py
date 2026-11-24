"""Re-encode the six legacy cross/source-control meshes at native 256, in a separate directory."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

from registry import read, save_json, save_npz, sha, verified
from replay import DEFAULT_ALIGNMENT, DEFAULT_CORPUS, DEFAULT_PHYSICS


def run(args):
    align=Path(args.alignment)
    solver=align.parent/'phase3_1b/solver'
    protocol=read(align/'protocol_v2.json')
    historical=Path(args.corpus).parent/'phase2_12_dataset/manifest_v2.json'
    verified(dict(path=str(historical),sha256=protocol['manifest_sha256']))
    registered={r['geometry_id']:r for r in read(historical)['records']}
    index=Path(args.multires)/'multires_geometry_manifest.json'
    work_a=read(index)['records']
    encoder=Path(read(solver/'protocol.json')['rle_operator'])
    expected=read(solver/'protocol.json')['rle_operator_sha256']
    verified(dict(path=str(encoder),sha256=expected))
    sys.path.insert(0,str(encoder.parent))
    import encode as enc
    enc.NATIVE_SIDE=256
    out=Path(args.out).resolve()
    dest=out/'diagnostic256'
    dest.mkdir(parents=True,exist_ok=True)
    manifest=dict(scope='Direct native256 re-encoding for historical cross-domain diagnostic geometries',
                  encoder=str(encoder),encoder_sha256=sha(encoder),operator_reference=str(solver/'protocol.json'),
                  open3d=enc.o3d.__version__,work_a_index=dict(path=str(index),sha256=sha(index)),
                  original_1024_retained=True,rows=[])
    for key in protocol['cross_ids'][:2]+[g for g in protocol['train_ids'] if g!='T2']:
        r=registered[key]
        matches=[e['id'] for e in work_a if e['raw']['sha256']==r['raw_sha256']]
        if matches:
            raise ValueError(f'Use the registered multiresolution asset for {key}: {matches}')
        g=out/'geometries'/(key+'.npz')
        with np.load(g,allow_pickle=False) as data:
            arrays={name:data[name] for name in data}
        if (int(arrays['side'])!=1024 or not np.array_equal(arrays['target_axes'],np.eye(3))
                or np.any(arrays['target_origin']) or float(arrays['density'])!=1 or np.any(arrays['fixed_moments'])):
            raise ValueError('historical geometry does not match direct Work A encoder frame/density')
        import trimesh
        raw=verified(dict(path=r['raw_file'],sha256=r['raw_sha256']))
        source=trimesh.load(raw,process=False,maintain_order=True)
        frame=r['coordinate_convention']
        n=dict(origin=frame['center'],L=frame['length_scale'])
        vertices=(np.asarray(source.vertices)-np.asarray(n['origin']))/n['L']
        if (not np.array_equal(vertices,arrays['vertices']) or
                not np.array_equal(source.faces,arrays['faces']) or sha(g)!=read(g.with_suffix('.json'))['geometry_sha256']):
            raise ValueError('recorded design mesh disagrees with archive')
        scene=enc.o3d.t.geometry.RaycastingScene(nthreads=2)
        tensor=enc.o3d.core.Tensor
        scene.add_triangles(tensor(arrays['vertices'].astype('f4')),tensor(arrays['faces'].astype('uint32')))
        directory=dest/key
        directory.mkdir(exist_ok=True)
        grid=directory/'rle256.npz'
        t=time.monotonic()
        check=enc.encode_rle(scene,grid)
        with np.load(grid,allow_pickle=False) as z:
            if int(z['native_resolution'])!=256:
                raise ValueError('incorrect native query grid')
            arrays.update(columns=z['columns'],side=z['native_resolution'],pitch=z['h'],origin=z['origin'],
                          rle_path=str(grid),rle_sha256=sha(grid))
        target=directory/'geometry.npz'
        save_npz(target,**arrays)
        row=dict(geometry_id=key,split=r['split'],raw_mesh=dict(path=r['raw_file'],sha256=r['raw_sha256']),
                 normalization=n,registered_grid_1024=dict(path=r['geometry_file'],sha256=r['geometry_sha256']),
                 registered_multiresolution_raw_hash_matches=matches,
                 RLE256=dict(path=str(grid),sha256=sha(grid)),target_geometry=dict(path=str(target),sha256=sha(target)),
                 quartering_or_decimation=False,seconds=time.monotonic()-t,ray_check=check)
        manifest['rows'].append(row)
        save_json(dest/'manifest.json',manifest)
        print(key,check,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alignment',default=DEFAULT_ALIGNMENT)
    p.add_argument('--corpus',default=DEFAULT_CORPUS)
    p.add_argument('--physics',default=DEFAULT_PHYSICS)
    p.add_argument('--multires',default='/data/xvxinyao/TVCG/datasets/phase3_1a_multires')
    p.add_argument('--out',required=True)
    run(p.parse_args())
