import numpy as np
import pytest

from devito import Grid, TimeFunction, Eq, Operator, solve

from xdsl.dialects.scf import For, Yield
from xdsl.dialects.arith import Addi
from xdsl.dialects.func import Call, Return
from xdsl.dialects.stencil import FieldType, ApplyOp, LoadOp, StoreOp
from xdsl.dialects.llvm import LLVMPointerType
from xdsl.printer import Printer


def test_xdsl_noop_I():
    # Define a simple Devito Operator
    grid = Grid(shape=(3,))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = Operator([eq], opt='xdsl-noop')
    op.apply(time_M=1)
    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()


def test_xdsl_noop_structure():
    # Define a simple Devito Operator
    grid = Grid(shape=(5, 5, 5))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)

    op1 = Operator([eq], opt='xdsl-noop')
    op1.apply(time_M=1)

    op2 = Operator([eq], opt='xdsl')
    op2.apply(time_M=1)
    # No correctness check, just running

    assert Printer().print(op1._module) == Printer().print(op2._module)


@pytest.mark.parametrize('shape', [(101, 101, 101)])
@pytest.mark.parametrize('so', [2, 4, 8])
@pytest.mark.parametrize('to', [2])
@pytest.mark.parametrize('nt', [10, 20, 100])
def test_acoustic_3D(shape, so, to, nt):

    grid = Grid(shape=shape)

    # Define the wavefield with the size of the model and the time dimension
    u = TimeFunction(name="u", grid=grid, time_order=to, space_order=so)

    pde = u.dt2 - u.laplace
    eq0 = solve(pde, u.forward)

    stencil = Eq(u.forward, eq0)
    u.data[:, :, :] = 0
    u.data[:, 40:50, 40:50] = 1

    # Devito Operator
    op1 = Operator([stencil], opt='xdsl-noop')
    op2 = Operator([stencil], opt='xdsl')

    # We here test only the initial code, so not really useful
    assert Printer().print(op1._module) == Printer().print(op2._module)


def test_xdsl_III():
    # Define a simple Devito Operator
    grid = Grid(shape=(5, 5, 5))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = Operator([eq], opt='xdsl-noop')
    op.apply(time_M=1)

    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()

    # Check number of args
    assert len(op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args) == 3
    assert isinstance(op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args[0].type, FieldType)  # noqa
    assert isinstance(op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args[1].type, FieldType)  # noqa
    assert isinstance(op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args[2].type, LLVMPointerType)  # noqa

    ops = list(op._module.regions[0].blocks[0].ops.first.body.blocks[0].ops)
    assert type(ops[5] == Addi)
    assert type(ops[6] == For)

    scffor_ops = list(ops[6].regions[0].blocks[0].ops)

    assert isinstance(scffor_ops[0], LoadOp)
    assert isinstance(scffor_ops[1], ApplyOp)
    assert isinstance(scffor_ops[2], StoreOp)
    assert isinstance(scffor_ops[3], Yield)

    assert type(ops[7] == Call)
    assert type(ops[8] == StoreOp)
    assert type(ops[9] == Return)


def test_diffusion_2D():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))

    # Devito setup
    f = TimeFunction(name='f', grid=grid, space_order=2)
    f.data[:] = 1
    eqn = Eq(f.dt, 0.5 * f.laplace)
    op = Operator(Eq(f.forward, solve(eqn, f.forward)))
    op.apply(time_M=1, dt=0.1)

    # xDSL-Devito setup
    f2 = TimeFunction(name='f2', grid=grid, space_order=2)
    f2.data[:] = 1
    eqn = Eq(f2.dt, 0.5 * f2.laplace)

    op1 = Operator([Eq(f2.forward, solve(eqn, f2.forward))], opt='xdsl-noop')
    op2 = Operator([Eq(f2.forward, solve(eqn, f2.forward))], opt='xdsl')

    assert Printer().print(op1._module) == Printer().print(op2._module)


@pytest.mark.parametrize('shape', [(11, 11), (31, 31), (51, 51), (101, 101)])
def test_diffusion_2D_II(shape):
    # Define a simple Devito Operator
    grid = Grid(shape=shape)
    rng = np.random.default_rng(123)
    dx = 2. / (shape[0] - 1)
    dy = 2. / (shape[1] - 1)
    sigma = .1
    dt = sigma * dx * dy
    nt = 10

    arr1 = rng.random(shape)

    # Devito setup
    f = TimeFunction(name='f', grid=grid, space_order=2)
    f.data[:] = arr1
    eqn = Eq(f.dt, 0.5 * f.laplace)
    op = Operator(Eq(f.forward, solve(eqn, f.forward)))
    op.apply(time_M=nt, dt=dt)

    # xDSL-Devito setup
    f2 = TimeFunction(name='f2', grid=grid, space_order=2)
    f2.data[:] = arr1
    eqn = Eq(f2.dt, 0.5 * f2.laplace)
    op = Operator(Eq(f2.forward, solve(eqn, f2.forward)), opt='xdsl-noop')
    op.apply(time_M=nt, dt=dt)

    max_error = np.max(np.abs(f.data - f2.data))
    assert np.isclose(max_error, 0.0, atol=1e-04)
    assert np.isclose(f.data, f2.data, rtol=1e-05).all()


@pytest.mark.parametrize('shape', [(11, 11, 11), (31, 31, 31),
                         (51, 51, 51), (101, 101, 101)])
def test_diffusion_3D_II(shape):
    shape = (10, 10, 10)
    grid = Grid(shape=shape)
    rng = np.random.default_rng(123)
    dx = 2. / (shape[0] - 1)
    dy = 2. / (shape[1] - 1)
    dz = 2. / (shape[2] - 1)

    sigma = .1
    dt = sigma * dx * dy * dz
    nt = 50

    rng = np.random.default_rng(123)

    arr1 = rng.random(shape)

    # Devito setup
    f = TimeFunction(name='f', grid=grid, space_order=2)
    f.data[:] = arr1
    eqn = Eq(f.dt, 0.5 * f.laplace)
    op = Operator(Eq(f.forward, solve(eqn, f.forward)))
    op.apply(time_M=nt, dt=dt)

    # xDSL-Devito setup
    f2 = TimeFunction(name='f2', grid=grid, space_order=2)
    f2.data[:] = arr1
    eqn = Eq(f2.dt, 0.5 * f2.laplace)
    op = Operator(Eq(f2.forward, solve(eqn, f2.forward)), opt='xdsl-noop')
    op.apply(time_M=50, dt=dt)

    max_error = np.max(np.abs(f.data - f2.data))
    assert np.isclose(max_error, 0.0, atol=1e-04)
    assert np.isclose(f.data, f2.data, rtol=1e-05).all()


@pytest.mark.parametrize('shape', [(11, 11, 11), (31, 31, 31),
                         (51, 51, 51), (101, 101, 101)])
@pytest.mark.parametrize('steps', [1, 3, 8, 40])
def test_unary(shape, steps):

    grid = Grid(shape=shape)

    u = TimeFunction(name='u', grid=grid)
    u.data[:, :] = 5
    eq = Eq(u.forward, u + 0.1)
    xop = Operator([eq], opt='xdsl-noop')
    xop.apply(time_M=steps)
    xdsl_data = u.data_with_halo.copy()

    u.data[:, :] = 5
    op = Operator([eq])
    op.apply(time_M=steps)
    orig_data = u.data_with_halo.copy()

    assert np.isclose(xdsl_data, orig_data, rtol=1e-06).all()


def test_xdsl_sine():
    import numpy as np
    from devito import (Function, TimeFunction, cos, sin, solve,
                        Eq, Operator)
    from examples.seismic import TimeAxis, RickerSource, Receiver, demo_model
    from matplotlib import pyplot as plt

    # We will start with the definitions of the grid and the physical parameters $v_{p}, \theta, \epsilon, \delta$. For simplicity, we won't use any absorbing boundary conditions to avoid reflections of outgoing waves at the boundaries of the computational domain, but we will have boundary conditions (zero Dirichlet) at $x=0,nx$ and $z=0,nz$ for the solution of the Poisson equation. We use a homogeneous model. The model is discretized with a grid of $101 \times 101$ and spacing of 10 m. The $v_{p}, \epsilon, \delta$ and $\theta$ parameters of this model are 3600 m∕s, 0.23, 0.17, and 45°, respectively. 

    # NBVAL_IGNORE_OUTPUT   

    shape = (101, 101)  # 101x101 grid
    spacing = (10., 10.)  # spacing of 10 meters
    nbl = 0  # number of pad points

    model = demo_model('layers-tti', spacing=spacing, space_order=8,
                       shape=shape, nbl=nbl, nlayers=1)

    # initialize Thomsem parameters to those used in Mu et al., (2020)
    model.update('vp', np.ones(shape)*3.6) # km/s
    model.update('epsilon', np.ones(shape)*0.23)
    model.update('delta', np.ones(shape)*0.17)
    model.update('theta', np.ones(shape)*(45.*(np.pi/180.)))  # radians

    # In cell below, symbols used in the PDE definition are obtained from the `model` object. Note that trigonometric functions proper of Devito are exploited.

    # %%
    # Get symbols from model
    theta = model.theta
    delta = model.delta
    epsilon = model.epsilon
    m = model.m

    # Use trigonometric functions from Devito
    costheta = cos(theta)

    # Values used to compute the time sampling
    epsilonmax = np.max(np.abs(epsilon.data[:]))
    deltamax = np.max(np.abs(delta.data[:]))
    etamax = max(epsilonmax, deltamax)
    vmax = model._max_vp
    max_cos_sin = np.amax(np.abs(np.cos(theta.data[:]) - np.sin(theta.data[:])))
    dvalue = min(spacing)

    # Compute the dt and set time range
    t0 = 0.  # Simulation time start
    tn = 150.  # Simulation time end (0.15 second = 150 msec)
    dt = (dvalue/(np.pi*vmax))*np.sqrt(1/(1+etamax*(max_cos_sin)**2))  # eq. above (cell 3)
    time_range = TimeAxis(start=t0, stop=tn, step=dt)
    print("time_range; ", time_range)

    # time stepping 
    p = TimeFunction(name="p", grid=model.grid, time_order=2, space_order=2)
    q = Function(name="q", grid=model.grid, space_order=8)

    # Main equations
    term1_p = (1 + 2*delta*(costheta**2) + 2*epsilon*costheta**4)*q.dx4

    # Create stencil and boundary condition expressions
    x, z = model.grid.dimensions

    optime = Operator([term1_p], opt='xdsl')
    optime.apply(time_M=100, dt=dt)
    import pdb;pdb.set_trace()