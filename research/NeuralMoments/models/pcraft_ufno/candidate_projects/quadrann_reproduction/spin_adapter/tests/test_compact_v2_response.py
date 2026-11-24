import numpy as np
import pytest
import torch
from spin_adapter.compact_v2_response import physical_tokens,pool_features,CutAwareMomentResponse


def make():
    xyz=np.array([[-.25,.1,.2],[.15,.1,.35],[.25,-.1,.15],[-.1,-.2,.1],[.22,.25,.2]])
    extent=np.array([[.08,.08,.25],[.08,.08,.12],[.08,.08,.09],[.08,.08,.1],[.08,.08,.12]])
    mass=extent.prod(1)*2
    x,y,z=xyz.T;ex,ey,ez=extent.T
    basis=np.stack([np.ones(5),x,y,z,x*x+ex*ex/12,y*y+ey*ey/12,z*z+ez*ez/12,y*z,x*z,x*y],1)
    return dict(xyz=xyz,cell_extent=extent,mass=mass,moment_basis=basis,origin=np.array([-.5,-.5,-.5]),native_resolution=np.array(256),h=np.array(1/256)),mass@basis


@pytest.mark.parametrize('grid',[1,4,16])
def test_physics_and_derivatives(grid):
    torch.manual_seed(18)
    geom,total=make();tokens=physical_tokens(geom,grid,device='cpu')
    np.testing.assert_allclose(tokens.moments.sum(0).numpy(),total,atol=1e-15)
    f=torch.randn(5,12,dtype=torch.float64,requires_grad=True)
    mass=torch.from_numpy(geom['mass'])
    op=CutAwareMomentResponse(12,16,4).double()
    mem=op.prepare(pool_features(f,mass,tokens),tokens)
    fixed=torch.tensor([.01,.002,0,0,.001,.001,.001,0,0,0],dtype=torch.float64)
    q=torch.tensor([[.01,.7,-.2,.15,.12,.14,.1,.08,-.05,.09],[-.05,.4,.2,-.7,.05,-.2,.15,.1,.03,-.07]],dtype=torch.float64,requires_grad=True)
    m=op(mem,fixed,q)
    torch.testing.assert_close(m+op(mem,fixed,-q),2*fixed[None]+tokens.moments.sum(0)[None].expand_as(m),rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(m,op(mem,fixed,3*q),rtol=1e-12,atol=1e-14)
    assert (m[:,0]>=fixed[0]).all() and (m[:,0]<=fixed[0]+tokens.moments[:,0].sum()).all()
    ends=torch.zeros(2,10,dtype=torch.float64);ends[:,0]=torch.tensor([1.,-1.]);ends.requires_grad_()
    e=op(mem,fixed,ends)
    torch.testing.assert_close(e[0],fixed+tokens.moments.sum(0));torch.testing.assert_close(e[1],fixed)
    torch.testing.assert_close(torch.autograd.grad(e.sum(),ends)[0],torch.zeros_like(ends),rtol=0,atol=1e-14)
    j=torch.stack([torch.autograd.grad(m[:,i].sum(),q,create_graph=True,retain_graph=True)[0] for i in range(10)],1)
    torch.testing.assert_close(j@q.unsqueeze(-1),torch.zeros(2,10,1,dtype=q.dtype),rtol=0,atol=1e-12)
    cols=[]
    for i in range(10):
        dx=torch.zeros_like(q);dx[:,i]=1e-5
        cols.append((op(mem,fixed,q+dx)-op(mem,fixed,q-dx))/(2e-5))
    torch.testing.assert_close(j,torch.stack(cols,-1),rtol=2e-4,atol=2e-9)
    (m.square().sum()+j.square().sum()).backward()
    assert f.grad is not None and torch.isfinite(f.grad).all() and f.grad.abs().sum()>0
    for name,p in op.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert op.cut[-1].weight.grad.abs().sum()>0
    if len(tokens.moments)>1:assert op.key.weight.grad.abs().sum()>0


@pytest.mark.parametrize('grid',[1,4,16])
def test_planar_zero_coefficients_are_locally_differentiable(grid):
    torch.manual_seed(22)
    geom,_=make(); tokens=physical_tokens(geom,grid,device='cpu')
    f=torch.randn(5,12,dtype=torch.float64)
    op=CutAwareMomentResponse(12,16,4).double();mem=op.prepare(pool_features(f,torch.from_numpy(geom['mass']),tokens),tokens)
    fixed=torch.zeros(10,dtype=torch.float64)
    q=torch.tensor([[.04,1.,0.,0.,0.,0.,0.,0.,0.,0.]],dtype=torch.float64,requires_grad=True)
    m=op(mem,fixed,q)
    j=torch.stack([torch.autograd.grad(m[:,i].sum(),q,retain_graph=True)[0] for i in range(10)],1)
    # The smoothing width is 1e-6 in normalized coefficient coordinates.
    # Test one-sided slopes within that local neighborhood, away from cut
    # saturation boundaries (this is not a native mesh FD check).
    for i in (2,3,7,8,9):
        dx=torch.zeros_like(q);dx[:,i]=1e-8
        plus=(op(mem,fixed,q+dx)-m)/1e-8
        minus=(m-op(mem,fixed,q-dx))/1e-8
        torch.testing.assert_close(plus,j[:,:,i],rtol=0.04,atol=2e-6)
        torch.testing.assert_close(minus,j[:,:,i],rtol=0.04,atol=2e-6)
