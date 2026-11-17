"""Physical expectations are analytic or finite differences, not a second model."""
from dataclasses import replace
import pytest
import torch

from .model import MaterialGrid, PINOMaterialV1

torch.set_num_threads(2)
D = torch.float64

def fixture(shell=True, variable=True):
    fixed = torch.zeros((8, 8, 8), dtype=D)
    if shell:
        fixed[0] = 1
        fixed[-1] = 1
    editable = 1 - fixed
    rho = torch.ones_like(fixed) * 2
    if variable:
        rho = rho + torch.linspace(0, 1, 8, dtype=D)[:, None, None]
    return MaterialGrid(rho, editable, torch.ones_like(rho)*3, fixed,
                        (-1., -1., -1.), (1., 1., 1.), 2., 3.)

@pytest.fixture
def model():
    torch.manual_seed(41)
    return PINOMaterialV1(width=2, modes=2, layers=2).eval()

def queries():
    return torch.tensor([[1,0,0,0,0,0,0,0,0,0],
                         [-1,0,0,0,0,0,0,0,0,0],
                         [.031,1,.2,-.3,.2,-.1,.1,.05,-.03,.07]], dtype=D)

def test_exact_full_box_moments_and_empty(model):
    g = fixture(False, False)
    m = model(g, queries())
    torch.testing.assert_close(m[0], torch.tensor([16,0,0,0,16/3,16/3,16/3,0,0,0],dtype=D),atol=1e-12,rtol=1e-12)
    assert torch.count_nonzero(m[1]) == 0

def test_fixed_slabs_preserved_and_spatial_density_mass(model):
    g = fixture()
    m = model(g, queries())
    # Two x slabs of width .25, y/z length2, density3 => mass6.
    assert m[1,0].item() == pytest.approx(6)
    assert m[0,0].item() == pytest.approx(21)  # editable6 volume * mean rho2.5 +6
    torch.testing.assert_close(m[1], model.encode(g).fixed,atol=1e-12,rtol=1e-12)

def test_bounds_complement_and_inertia_psd(model):
    cache = model.encode(fixture())
    torch.manual_seed(12)
    c = torch.randn(25,10,dtype=D)
    m, n = model.evaluate(cache,c), model.evaluate(cache,-c)
    assert bool((m[:,0] >= cache.fixed[0]-1e-12).all())
    assert bool((m[:,0] <= cache.full[0]+1e-12).all())
    torch.testing.assert_close(m+n,(cache.full+cache.fixed).expand_as(m),atol=1e-12,rtol=1e-12)
    for a in m:
        center=a[1:4]/a[0]
        second=torch.stack((torch.stack((a[4],a[9],a[8])),torch.stack((a[9],a[5],a[7])),torch.stack((a[8],a[7],a[6]))))
        cov=second-a[0]*torch.outer(center,center)
        inertia=torch.eye(3,dtype=D)*cov.trace()-cov
        assert torch.linalg.eigvalsh(inertia).min()>-1e-11

@pytest.mark.parametrize('factor',[.001,7.,1000.])
def test_query_positive_gauge_and_physical_length_covariance(model,factor):
    g=fixture(); c=queries(); deg=torch.tensor([0,1,1,1,2,2,2,2,2,2],dtype=D)
    original=model(g,c)
    torch.testing.assert_close(model(g,c*factor),original,atol=1e-11,rtol=1e-11)
    scaled=replace(g,lower=tuple(x*factor for x in g.lower),upper=tuple(x*factor for x in g.upper),length=g.length*factor)
    torch.testing.assert_close(model(scaled,c/factor**deg)/factor**(3+deg),original,atol=1e-10,rtol=1e-10)

def test_raw_jacobian_and_nonunit_finite_difference(model):
    cache=model.encode(fixture()); c=queries()[2:]*3.7
    jac=model.jacobian_input(cache,c)
    eps=1e-5
    for j in range(10):
        perturb=torch.zeros_like(c); perturb[0,j]=eps
        fd=(model.evaluate(cache,c+perturb)-model.evaluate(cache,c-perturb))/(2*eps)
        torch.testing.assert_close(jac[:,:,j],fd,atol=2e-8,rtol=2e-6)
    assert jac.abs().max()>1e-4

def test_density_total_gradient_and_neural_gradient(model):
    g=fixture(); rho=g.density.clone().requires_grad_(); g=replace(g,density=rho)
    c=queries()[2:]; loss=model(g,c)[0,4]
    grad=torch.autograd.grad(loss,rho,retain_graph=True)[0]
    direction=torch.linspace(-.7,1.,rho.numel(),dtype=D).reshape_as(rho)*g.editable
    eps=.002
    fd=(model(replace(g,density=rho.detach()+eps*direction),c)[0,4]-model(replace(g,density=rho.detach()-eps*direction),c)[0,4])/(2*eps)
    torch.testing.assert_close((grad*direction).sum(),fd,atol=2e-5,rtol=3e-3)
    loss.backward()
    spectral=[p.grad for n,p in model.named_parameters() if 'weights_' in n]
    assert all(x is not None and torch.isfinite(x).all() for x in spectral)
    assert sum(x.abs().square().sum().item() for x in spectral)>1e-12
    assert model.bandwidth.weight.grad.abs().max()>1e-6
    assert torch.count_nonzero(grad[g.fixed.bool()])==0

def test_zero_density_and_fixed_only(model):
    g=fixture(); zero=torch.zeros_like(g.density)
    empty=replace(g,density=zero,fixed_density=zero)
    assert torch.count_nonzero(model(empty,queries()))==0
    fixed_only=replace(g,editable=zero)
    m=model(fixed_only,queries())
    torch.testing.assert_close(m,m[0].expand_as(m))

@pytest.mark.parametrize('bad', ['negative_density','overlap','fraction_mask','fixed_grad','density_nan','length_zero'])
def test_reject_invalid_material(model,bad):
    g=fixture()
    changes={'negative_density':dict(density=-g.density),'overlap':dict(editable=torch.ones_like(g.editable)),
             'fraction_mask':dict(editable=g.editable*.5),'fixed_grad':dict(fixed_density=g.fixed_density.requires_grad_()),
             'density_nan':dict(density=g.density*float('nan')),'length_zero':dict(length=0.)}
    # requires_grad_ above changes the original; replace it for other cases.
    if bad!='fixed_grad': g=replace(g,fixed_density=g.fixed_density.detach())
    with pytest.raises(ValueError): model(replace(g,**changes[bad]),queries())

@pytest.mark.parametrize('c',[torch.zeros(1,10,dtype=D),torch.full((1,10),float('nan'),dtype=D)])
def test_reject_degenerate_queries(model,c):
    with pytest.raises(ValueError): model(fixture(),c)

def test_plane_cut_against_analytic_half_box(model):
    # Constant tau isolates quadrature/smoothing; physical x>0 unit-density cube.
    with torch.no_grad():
        model.bandwidth.weight.zero_(); model.bandwidth.bias.zero_()
    g=replace(fixture(False,False),density=torch.ones((8,8,8),dtype=D))
    c=torch.tensor([[0,1,0,0,0,0,0,0,0,0]],dtype=D)
    expected=torch.tensor([4,2,0,0,4/3,4/3,4/3,0,0,0],dtype=D)
    # Smooth boundary biases first moment. Exact mass and second moments by symmetry.
    m=model(g,c)[0]
    torch.testing.assert_close(m[[0,2,3,4,5,6,7,8,9]],expected[[0,2,3,4,5,6,7,8,9]],atol=1e-12,rtol=1e-12)
    assert abs(m[1]-2)<.03


def test_closed_fixed_shell_matches_analytic_outer_minus_inner_cube(model):
    g=fixture(False,False)
    shell=torch.ones_like(g.fixed); shell[1:-1,1:-1,1:-1]=0
    g=replace(g,fixed=shell,editable=1-shell)
    # Outer side2 minus inner side1.5, both centered, fixed density3.
    mass=3*(8-1.5**3); diagonal=3*(8/3-1.5**5/12)
    expected=torch.tensor([mass,0,0,0,diagonal,diagonal,diagonal,0,0,0],dtype=D)
    torch.testing.assert_close(model(g,queries()[1:2])[0],expected,atol=1e-12,rtol=1e-12)


def test_grid_shape_and_bounds_rejected(model):
    g=fixture()
    with pytest.raises(ValueError): model(replace(g,density=g.density[:7]),queries())
    with pytest.raises(ValueError): model(replace(g,upper=g.lower),queries())


def test_jacobian_graph_reaches_pino_and_endpoint_derivative_zero(model):
    cache=model.encode(fixture())
    jac=model.jacobian_input(cache,queries(),create_graph=True)
    assert torch.count_nonzero(jac[:2])==0
    jac[2].square().sum().backward()
    grads=[p.grad for n,p in model.named_parameters() if 'weights_' in n]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().square().sum().item() for g in grads)>1e-12
