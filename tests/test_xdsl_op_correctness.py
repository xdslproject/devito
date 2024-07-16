import numpy as np
from devito import (Grid, TimeFunction, Eq, Operator, norm, switchconfig,
                    Function, NODE, div, grad, solve)
import pytest

from xdsl.dialects.scf import For, Yield
from xdsl.dialects.arith import Addi, Constant
from xdsl.dialects.func import Call, Return
from xdsl.dialects.stencil import FieldType, ApplyOp, LoadOp, StoreOp

from devito.types.basic import Symbol

from examples.seismic import demo_model, TimeAxis, RickerSource


def test_udx():

    # Define a simple Devito Operator
    grid = Grid(shape=(5, 5))
    u = TimeFunction(name='u', grid=grid)
    u.data[:] = 0.1
    eq = Eq(u.forward, u.dx)
    op = Operator([eq])
    op.apply(time_M=5)
    norm1 = norm(u)

    u.data[:] = 0.1

    xdsl_op = Operator([eq], opt='xdsl')
    xdsl_op.apply(time_M=5)
    norm2 = norm(u)

    assert np.isclose(norm1, norm2, atol=1e-5, rtol=0)
    assert np.isclose(norm1, 14636.3955, atol=1e-5, rtol=0)


def test_u_plus1_conversion():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    u.data[:] = 0
    eq = Eq(u.forward, u + 1)
    op = Operator([eq])
    op.apply(time_M=5)
    norm1 = norm(u)

    u.data[:] = 0
    xdsl_op = Operator([eq], opt='xdsl')
    xdsl_op.apply(time_M=5)
    norm2 = norm(u)

    assert np.isclose(norm1, norm2, atol=1e-5, rtol=0)
    assert np.isclose(norm1, 23.43075, atol=1e-5, rtol=0)


def test_u_and_v_conversion():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid, time_order=2)
    v = TimeFunction(name='v', grid=grid, time_order=2)
    u.data[:] = 0.0001
    v.data[:] = 0.0001
    eq0 = Eq(u.forward, u.dx + v.dy)
    eq1 = Eq(v.forward, u.dy + v.dx)
    op = Operator([eq0, eq1])
    op.apply(time_M=5, dt=0.1)
    norm_u = norm(u)
    norm_v = norm(v)

    u.data[:] = 0.0001
    v.data[:] = 0.0001
    xdsl_op = Operator([eq0, eq1], opt='xdsl')
    xdsl_op.apply(time_M=5, dt=0.1)
    norm_u2 = norm(u)
    norm_v2 = norm(v)

    assert np.isclose(norm_u, norm_u2, atol=1e-5, rtol=0)
    assert np.isclose(norm_u, 2.0664592, atol=1e-5, rtol=0)
    assert np.isclose(norm_v, norm_v2, atol=1e-5, rtol=0)
    assert np.isclose(norm_v, 2.0664592, atol=1e-5, rtol=0)

    assert len(xdsl_op._module.regions[0].blocks[0].ops.first.body.blocks[0].ops) == 10

    assert isinstance(xdsl_op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args[0].type, FieldType)  # noqa
    assert isinstance(xdsl_op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args[1].type, FieldType)  # noqa
    assert isinstance(xdsl_op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args[2].type, FieldType)  # noqa

    ops = list(xdsl_op._module.regions[0].blocks[0].ops.first.body.blocks[0].ops)
    assert type(ops[5] == Addi)
    assert type(ops[6] == For)

    scffor_ops = list(ops[6].regions[0].blocks[0].ops)

    assert len(scffor_ops) == 7

    # First
    assert isinstance(scffor_ops[0], LoadOp)
    assert isinstance(scffor_ops[1], LoadOp)
    assert isinstance(scffor_ops[2], ApplyOp)
    assert isinstance(scffor_ops[3], StoreOp)

    # Second
    assert isinstance(scffor_ops[4], ApplyOp)
    assert isinstance(scffor_ops[5], StoreOp)

    # Yield
    assert isinstance(scffor_ops[6], Yield)

    assert type(ops[7] == Call)
    assert type(ops[8] == StoreOp)
    assert type(ops[9] == Return)


def test_symbol_I():
    # Define a simple Devito a = 1 operator

    a = Symbol('a')
    eq0 = Eq(a, 1)

    op = Operator([eq0], opt='xdsl')

    op.apply()

    assert len(op._module.regions[0].blocks[0].ops.first.body.blocks[0].ops) == 2

    ops = list(op._module.regions[0].blocks[0].ops.first.body.blocks[0].ops)
    assert isinstance(ops[0], Constant)
    assert ops[0].result.name_hint == a.name
    assert type(ops[0] == Return)


# This test should fail, as we are trying to use an inplace operation
@pytest.mark.xfail(reason="Cannot store to a field that is loaded from")
def test_inplace():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid, time_order=2)

    u.data[:] = 0.0001

    eq0 = Eq(u, u.dx)

    xdsl_op = Operator([eq0], opt='xdsl')
    xdsl_op.apply(time_M=5, dt=0.1)


@pytest.mark.parametrize('rotate', [False, True])
@switchconfig(profiling='advanced')
def test_extraction_from_lifted_ispace(rotate):
    """
    Test that the aliases are scheduled correctly when extracted from
    Clusters whose iteration space is lifted (ie, stamp != 0).
    """
    so = 8
    grid = Grid(shape=(6, 6, 6))

    f = Function(name='f', grid=grid, space_order=so, is_param=True)
    v = TimeFunction(name="v", grid=grid, space_order=so)
    v1 = TimeFunction(name="v1", grid=grid, space_order=so)
    p = TimeFunction(name="p", grid=grid, space_order=so, staggered=NODE)
    p1 = TimeFunction(name="p1", grid=grid, space_order=so, staggered=NODE)

    v.data_with_halo[:] = 1.
    v1.data_with_halo[:] = 1.
    p.data_with_halo[:] = 0.5
    p1.data_with_halo[:] = 0.5
    f.data_with_halo[:] = 0.2

    eqns = [Eq(v.forward, v - f*p),
            Eq(p.forward, p - v.forward.dx + div(f*grad(p)))]

    # Operator
    op0 = Operator(eqns, opt='xdsl')
    op1 = Operator(eqns, opt=('advanced', {'openmp': True, 'cire-mingain': 1,
                              'cire-rotate': rotate}))

    # Check numerical output
    op0(time_M=1)
    summary = op1(time_M=1, v=v1, p=p1)
    assert np.isclose(norm(v), norm(v1), rtol=1e-5)
    assert np.isclose(norm(p), norm(p1), atol=1e-5)

    # Also check against expected operation count to make sure
    # all redundancies have been detected correctly
    assert summary[('section0', None)].ops == 93


@pytest.mark.parametrize('shape', [(101, 101), (201, 201)])
@pytest.mark.parametrize('nt', [10, 20, ])
def test_dt2_sources_script(shape, nt):

    spacing = (10., 10.)  # spacing of 10 meters
    nbl = 1  # number of pad layers

    model = demo_model('layers-tti', spacing=spacing, space_order=4,
                       shape=shape, nbl=nbl, nlayers=nbl)

    # Compute the dt and set time range
    t0 = 0.  # Simulation time start
    dt = model.critical_dt

    time_range = TimeAxis(start=t0, stop=nt, step=dt)

    p = TimeFunction(name="p", grid=model.grid, time_order=2, space_order=8)

    # Main equations
    stencil_p = solve(p.dt2, p.forward)
    update_p = Eq(p.forward, stencil_p)

    # Create stencil and boundary condition expressions
    x, z = model.grid.dimensions

    # set source and receivers
    src = RickerSource(name='src', grid=model.grid, f0=0.02, npoint=1,
                       time_range=time_range)

    src.coordinates.data[:, 0] = model.domain_size[0] * .5
    src.coordinates.data[:, 1] = model.domain_size[0] * .5
    # Define the source injection

    src_term = src.inject(field=p.forward, expr=src)

    optime = Operator([update_p] + src_term)
    optime.apply(time=time_range.num-1, dt=dt)
    norm0 = norm(p)

    p.data[:] = 0.

    optime = Operator([update_p] + src_term, opt='xdsl')
    optime.apply(time=time_range.num-1, dt=dt)
    norm1 = norm(p)

    assert np.isclose(norm0, norm1, atol=1e-5, rtol=0)
