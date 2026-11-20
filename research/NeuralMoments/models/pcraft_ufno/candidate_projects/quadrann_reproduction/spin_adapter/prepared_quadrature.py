"""Generate and audit Gaussian nodes in interval blocks, preserving native order."""
import numpy as np

from .rle_quadrature import native_moment_basis


def prepared_quadrature(geometry, order, storage_dtype):
    if not isinstance(order,int) or order<2:
        raise ValueError("order must be an integer >= 2")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    ijk = np.stack(np.meshgrid(*([np.arange(order)]*3),indexing="ij"),axis=-1).reshape(-1,3)
    ref_xyz, ref_w = nodes[ijk], weights[ijk].prod(axis=1)
    cols, h, origin = geometry["columns"], geometry["h"], geometry["origin"]
    n, k = len(cols), len(ijk)
    xyz = np.empty((n,k,3),dtype=storage_dtype)
    fraction_weights = np.empty((n,k),dtype=storage_dtype)
    base_mass = np.empty(n,dtype=storage_dtype)
    for start in range(0,n,8192):
        sl = slice(start,min(n,start+8192))
        rows = cols[sl]
        lo = origin+rows[:,:3]*h
        hi = origin+np.column_stack((rows[:,0]+1,rows[:,1]+1,rows[:,3]))*h
        half=(hi-lo)*.5
        points=(lo+hi)[:,None,:]*.5+ref_xyz[None,:,:]*half[:,None,:]
        physical=geometry["density"]*half.prod(axis=1)[:,None]*ref_w
        # Use the same interval-by-interval FP64 mass and moment identities as
        # the full-table oracle. The ten-column node basis dies at each block.
        mass=np.add.accumulate(physical,axis=1)[:,-1]
        if not np.isfinite(points).all() or not (physical>0).all():
            raise ValueError("invalid Gaussian nodes or weights")
        if not np.allclose(mass,geometry["mass"][sl],rtol=1e-12,atol=1e-14):
            raise RuntimeError("quadrature does not preserve interval mass")
        basis=native_moment_basis(points.reshape(-1,3)).reshape(-1,k,10)
        moments=np.einsum("nk,nki->ni",physical,basis)
        if not np.allclose(moments,geometry["mass"][sl,None]*geometry["moment_basis"][sl],
                           rtol=1e-11,atol=1e-12):
            raise RuntimeError("quadrature failed exact ten-moment audit")
        xyz[sl],fraction_weights[sl],base_mass[sl]=points,physical,mass
    return xyz.reshape(-1,3),fraction_weights.ravel(),base_mass
