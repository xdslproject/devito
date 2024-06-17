import numpy as np
import pytest

from devito import (Grid, TensorTimeFunction, VectorTimeFunction, div, grad, diag, solve,
                    Operator, Eq, Constant, norm, SpaceDimension)
from devito.types import Array, Function, TimeFunction
from devito.tools import as_tuple

from examples.seismic.source import RickerSource, TimeAxis
from xdsl.dialects.scf import For, Yield
from xdsl.dialects.arith import Addi
from xdsl.dialects.func import Call, Return
from xdsl.dialects.stencil import FieldType, ApplyOp, LoadOp, StoreOp
from xdsl.dialects.llvm import LLVMPointerType

from examples.seismic.source import RickerSource, TimeAxis


def test_xdsl_I():
    # Define a simple Devito Operator
    grid = Grid(shape=(3,))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = Operator([eq], opt='xdsl')
    op.apply(time_M=1)
    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()


def test_xdsl_II():
    # Define a simple Devito Operator
    grid = Grid(shape=(4, 4))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = Operator([eq], opt='xdsl')
    op.apply(time_M=1)
    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()


def test_xdsl_III():
    # Define a simple Devito Operator
    grid = Grid(shape=(5, 5, 5))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = Operator([eq], opt='xdsl')
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
    op = Operator(Eq(f2.forward, solve(eqn, f2.forward)), opt='xdsl')
    op.apply(time_M=1, dt=0.1)

    assert np.isclose(f.data, f2.data, rtol=1e-06).all()


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
    op = Operator(Eq(f2.forward, solve(eqn, f2.forward)), opt='xdsl')
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
    op = Operator(Eq(f2.forward, solve(eqn, f2.forward)), opt='xdsl')
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
    xop = Operator([eq], opt='xdsl')
    xop.apply(time_M=steps)
    xdsl_data = u.data_with_halo.copy()

    u.data[:, :] = 5
    op = Operator([eq])
    op.apply(time_M=steps)
    orig_data = u.data_with_halo.copy()

    assert np.isclose(xdsl_data, orig_data, rtol=1e-06).all()


@pytest.mark.parametrize('shape', [(101, 101, 101)])
@pytest.mark.parametrize('so', [2, 4, 8])
@pytest.mark.parametrize('to', [2])
@pytest.mark.parametrize('nt', [10, 20, 100])
def test_acoustic_3D(shape, so, to, nt):

    grid = Grid(shape=shape)
    dt = 0.0001

    # Define the wavefield with the size of the model and the time dimension
    u = TimeFunction(name="u", grid=grid, time_order=to, space_order=so)

    pde = u.dt2 - u.laplace
    eq0 = solve(pde, u.forward)

    stencil = Eq(u.forward, eq0)
    u.data[:, :, :] = 0
    u.data[:, 40:50, 40:50] = 1

    # Devito Operator
    op = Operator([stencil])
    op.apply(time=nt, dt=dt)
    devito_norm = norm(u)

    u.data[:, :, :] = 0
    u.data[:, 40:50, 40:50] = 1

    # XDSL Operator
    xdslop = Operator([stencil], opt='xdsl')
    xdslop.apply(time=nt, dt=dt)
    xdsl_norm = norm(u)

    assert np.isclose(devito_norm, xdsl_norm, rtol=1e-04).all()


@pytest.mark.parametrize('shape', [(21, 21, 21)])
@pytest.mark.parametrize('so', [2, 4])
@pytest.mark.parametrize('to', [2])
@pytest.mark.parametrize('nt', [20])
def test_standard_mlir_rewrites(shape, so, to, nt):

    grid = Grid(shape=shape)
    dt = 0.0001

    # Define the wavefield with the size of the model and the time dimension
    u = TimeFunction(name="u", grid=grid, time_order=to, space_order=so)

    pde = u.dt2 - u.laplace
    eq0 = solve(pde, u.forward)

    stencil = Eq(u.forward, eq0)
    u.data[:, :, :] = 0
    u.data[:, 40:50, 40:50] = 1

    # Devito Operator
    op = Operator([stencil])
    op.apply(time=nt, dt=dt)

    u.data[:, :, :] = 0
    u.data[:, 40:50, 40:50] = 1

    # XDSL Operator
    xdslop = Operator([stencil], opt='xdsl')
    xdslop.apply(time=nt, dt=dt)


class TestSources:

    @switchconfig(openmp=False)
    @pytest.mark.parametrize('shape', [(8, 8), (38, 38), ])
    @pytest.mark.parametrize('tn', [20, 80])
    @pytest.mark.parametrize('factor', [-0.1, 0.0, 0.1, 0.5, 1.1])
    @pytest.mark.parametrize('factor2', [-0.1, 0.1, 0.5, 1.1])
    def test_source_only(self, shape, tn, factor, factor2):
        spacing = (10.0, 10.0)
        extent = tuple(np.array(spacing) * (shape[0] - 1))
        origin = (0.0, 0.0)

        v = np.empty(shape, dtype=np.float32)
        v[:, :51] = 1.5
        v[:, 51:] = 2.5

        grid = Grid(shape=shape, extent=extent, origin=origin)

        t0 = 0.0
        # Comes from args
        tn = tn
        dt = 1.6
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        f0 = 0.010
        src = RickerSource(name="src", grid=grid, f0=f0, npoint=5, time_range=time_range)

        domain_size = np.array(extent)

        src.coordinates.data[0, :] = domain_size * factor
        src.coordinates.data[0, -1] = 19.0 * factor2

        u = TimeFunction(name="u", grid=grid, space_order=2)
        m = Function(name='m', grid=grid)
        m.data[:] = 1./(v*v)

        src_term = src.inject(field=u.forward, expr=src * dt**2 / m)

        op = Operator([src_term], opt="advanced")
        op(time=time_range.num-1, dt=dt)
        normdv = np.linalg.norm(u.data[0])
        u.data[:, :] = 0

        opx = Operator([src_term], opt="xdsl")
        opx(time=time_range.num-1, dt=dt)
        normxdsl = np.linalg.norm(u.data[0])

        assert np.isclose(normdv, normxdsl, rtol=1e-04)

    @switchconfig(openmp=False)
    @pytest.mark.parametrize('shape', [(8, 8)])
    @pytest.mark.parametrize('tn', [20, 80])
    @pytest.mark.parametrize('factor', [-0.1, 0.0, 0.1, 0.5, 1.1])
    @pytest.mark.parametrize('factor2', [-0.1, 0.1, 0.5, 1.1])
    def test_source_structure(self, shape, tn, factor, factor2):
        spacing = (10.0, 10.0)
        extent = tuple(np.array(spacing) * (shape[0] - 1))
        origin = (0.0, 0.0)

        v = np.empty(shape, dtype=np.float32)
        v[:, :51] = 1.5
        v[:, 51:] = 2.5

        grid = Grid(shape=shape, extent=extent, origin=origin)

        t0 = 0.0
        # Comes from args
        tn = tn
        dt = 1.6
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        f0 = 0.010
        src = RickerSource(name="src", grid=grid, f0=f0, npoint=5, time_range=time_range)

        domain_size = np.array(extent)

        src.coordinates.data[0, :] = domain_size * factor
        src.coordinates.data[0, -1] = 19.0 * factor2

        u = TimeFunction(name="u", grid=grid, space_order=2)
        m = Function(name='m', grid=grid)
        m.data[:] = 1./(v*v)

        src_term = src.inject(field=u.forward, expr=src * dt**2 / m)

        op = Operator([src_term], opt="advanced")
        op(time=time_range.num-1, dt=dt)
        normdv = np.linalg.norm(u.data[0])
        u.data[:, :] = 0

        opx = Operator([src_term], opt="xdsl")
        opx(time=time_range.num-1, dt=dt)
        normxdsl = np.linalg.norm(u.data[0])

        assert np.isclose(normdv, normxdsl, rtol=1e-04)


    @switchconfig(openmp=False)
    @pytest.mark.parametrize('shape', [(38, 38), ])
    @pytest.mark.parametrize('tn', [20, 80])
    @pytest.mark.parametrize('factor', [0.5, 0.8])
    @pytest.mark.parametrize('factor2', [0.5, 0.8])
    def test_forward_src_stencil(self, shape, tn, factor, factor2):
        spacing = (10.0, 10.0)
        extent = tuple(np.array(spacing) * (shape[0] - 1))
        origin = (0.0, 0.0)

        v = np.empty(shape, dtype=np.float32)
        v[:, :51] = 1.5
        v[:, 51:] = 2.5

        grid = Grid(shape=shape, extent=extent, origin=origin)

        t0 = 0.0
        # Comes from args
        tn = tn
        dt = 1.6
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        f0 = 0.010
        src = RickerSource(name="src", grid=grid, f0=f0, npoint=5, time_range=time_range)

        domain_size = np.array(extent)

        src.coordinates.data[0, :] = domain_size * factor
        src.coordinates.data[0, -1] = 100.0 * factor2

        u = TimeFunction(name="u", grid=grid, space_order=2, time_order=2)
        m = Function(name='m', grid=grid)
        m.data[:] = 1./(v*v)

        src_term = src.inject(field=u.forward, expr=src * dt**2 / m)

        pde = u.dt2 - u.laplace
        eq0 = solve(pde, u.forward)
        stencil = Eq(u.forward, eq0)

        op = Operator([stencil, src_term], opt="advanced")
        op(time=time_range.num-1, dt=dt)
        # normdv = norm(u)
        normdv = np.linalg.norm(u.data)

        u.data[:, :] = 0

        opx = Operator([stencil, src_term], opt="xdsl")
        opx(time=time_range.num-1, dt=dt)
        normxdsl = np.linalg.norm(u.data)
        # normxdsl = norm(u)

        assert not np.isclose(normdv, 0.0, rtol=1e-04)
        assert np.isclose(normdv, normxdsl, rtol=1e-04)


def test_xdsl_mul_eqs_I():
    # Define a Devito Operator with multiple eqs
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name="u", grid=grid, space_order=2)
    u.data[:, :, :] = 0

    eq0 = Eq(u.forward, u + 1)
    eq1 = Eq(u.forward, u + 2)

    op = Operator([eq0, eq1], opt="advanced")

    op.apply(time_M=4)

    assert (u.data[1, :] == 10.0).all()
    assert (u.data[0, :] == 8.0).all()

    u.data[:, :, :] = 0

    op = Operator([eq0, eq1], opt="xdsl")

    op.apply(time_M=4)

    assert (u.data[1, :] == 10.0).all()
    assert (u.data[0, :] == 8.0).all()


def test_xdsl_mul_eqs_II():
    # Define a Devito Operator with multiple eqs
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name="u", grid=grid, space_order=2)
    u.data[:, :, :] = 0

    eq0 = Eq(u.forward, 0.01*u.dx)
    eq1 = Eq(u.forward, u.dx + 2)

    op = Operator([eq0, eq1], opt="advanced")

    op.apply(time_M=4)

    norm_devito = norm(u)

    u.data[:, :, :] = 0

    op = Operator([eq0, eq1], opt="xdsl")

    op.apply(time_M=4)

    norm_xdsl = norm(u)

    assert np.isclose(norm_devito, norm_xdsl, rtol=0.0001)


def test_xdsl_mul_eqs_III():
    # Define a Devito Operator with multiple eqs
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name="u", grid=grid, space_order=2)
    u.data[:, :, :] = 0

    eq0 = Eq(u.forward, u + 2)
    eq1 = Eq(u.forward, u + 1)

    op = Operator([eq0, eq1], opt="advanced")

    op.apply(time_M=4)

    norm_devito = norm(u)

    u.data[:, :, :] = 0

    op = Operator([eq0, eq1], opt="xdsl")

    op.apply(time_M=4)

    norm_xdsl = norm(u)

    assert np.isclose(norm_devito, norm_xdsl, rtol=0.0001)


def test_xdsl_mul_eqs_IV():
    # Define a Devito Operator with multiple eqs
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name="u", grid=grid, space_order=2)
    u.data[:, :, :] = 0

    eq0 = Eq(u.forward, u + 2)
    eq1 = Eq(u.forward, u + 1)

    op = Operator([eq0, eq1], opt="advanced")

    op.apply(time_M=4)

    assert (u.data[1, :] == 5.0).all()
    assert (u.data[0, :] == 4.0).all()

    u.data[:, :, :] = 0

    op = Operator([eq0, eq1], opt="xdsl")

    op.apply(time_M=4)

    assert (u.data[1, :] == 5.0).all()
    assert (u.data[0, :] == 4.0).all()


def test_xdsl_mul_eqs_V():
    # Define a Devito Operator with multiple eqs
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name="u", grid=grid, space_order=2)
    v = TimeFunction(name="v", grid=grid, space_order=2)

    u.data[:, :, :] = 0
    v.data[:, :, :] = 0

    eq0 = Eq(u.forward, u + 1)
    eq1 = Eq(v.forward, v + 1)

    op = Operator([eq0, eq1], opt="advanced")

    op.apply(time_M=4)

    norm_u_devito = norm(u)
    norm_v_devito = norm(v)

    u.data[:, :, :] = 0
    v.data[:, :, :] = 0

    op = Operator([eq0, eq1], opt="xdsl")

    op.apply(time_M=4)

    norm_u_xdsl = norm(u)
    norm_v_xdsl = norm(v)

    assert np.isclose(norm_u_devito, 25.612497, rtol=0.0001)
    assert np.isclose(norm_v_devito, 25.612497, rtol=0.0001)

    assert np.isclose(norm_u_devito, norm_u_xdsl, rtol=0.0001)
    assert np.isclose(norm_v_devito, norm_v_xdsl, rtol=0.0001)


def test_forward_assignment_f32():
    # simple Devito a = 1 operator

    grid = Grid(shape=(4, 4))
    u = TimeFunction(name="u", grid=grid, space_order=2)
    u.data[:, :, :] = 0

    eq0 = Eq(u.forward, 0.1)

    op = Operator([eq0], opt='xdsl')

    op.apply(time_M=1)

    assert np.isclose(norm(u), 0.56584, rtol=0.001)


class TestAntiDepSupported(object):

    @pytest.mark.xfail(reason="Cannot store to a field that is loaded from")
    @pytest.mark.parametrize('exprs,directions,expected,visit', [
        # 6) No dependencies
        (('Eq(u[t+1,x,y,z], u[t,x,y,z] + v[t,x,y,z])',
            'Eq(v[t+1,x,y,z], u[t,x,y,z] + v[t,x,y,z])'),
            '+++++', ['txyz', 'txyz', 'txy'], 'txyzz'),
        (('Eq(u[t+1,x,y,z], u[t,x,y,z] + v[t,x,y,z])',
            'Eq(v[t+1,x,y,z], u[t,x,y,z] + v[t,x,y,z])'),
            '+++++', ['txyz', 'txyz', 'txy'], 'txyzz'),
        # 7)
        (('Eq(u[t+1,x,y,z], u[t,x,y,z] + v[t,x,y,z])',
            'Eq(v[t+1,x,y,z], u[t+1,x,y,z] + v[t,x,y,z])'),
            '++++++', ['txyz', 'txyz', 'txyz'], 'txyzzz'),
        (('Eq(u[t+1,x,y,z], u[t,x,y,z] + v[t,x,y,z])',
            'Eq(v[t+1,x,y,z], u[t+1,x,y+1,z] + u[t+1,x,y,z] + v[t,x,y,z])'),
            '++++++', ['txyz', 'txyz', 'txyz'], 'txyzzz'),
    ])
    def test_consistency_anti_dependences(self, exprs, directions, expected, visit):
        """
        Test that anti dependences end up generating multi loop nests, rather
        than a single loop nest enclosing all of the equations.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions  # noqa
        xi, yi, zi = grid.interior.dimensions  # noqa
        t = grid.stepping_dim  # noqa

        ti0 = Function(name='ti0', shape=grid.shape, dimensions=grid.dimensions)  # noqa
        ti1 = Function(name='ti1', shape=grid.shape, dimensions=grid.dimensions)  # noqa
        ti3 = Function(name='ti3', shape=grid.shape, dimensions=grid.dimensions)  # noqa
        f = Function(name='f', grid=grid)  # noqa

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        w = TimeFunction(name='w', grid=grid)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        eqns = []
        for e in exprs:
            eqns.append(eval(e))

        u.data[:, :, :] = 1
        v.data[:, :, :] = 1

        op = Operator(eqns)
        op.apply(time_M=2)
        devito_a = u.data_with_halo[:, :, :]

        # Re-initialize
        u.data[:, :, :] = 1
        v.data[:, :, :] = 1

        xdsl_op = Operator(eqns, opt='xdsl')
        xdsl_op.apply(time_M=2)

        xdsl_a = u.data_with_halo[:, :, :]

        assert np.all(devito_a == xdsl_a)


@pytest.mark.xfail(reason="not supported in xDSL yet")
class TestAntiDepNotSupported(object):

    @pytest.mark.parametrize('exprs,directions,expected,visit', [
        # 0) WAR 2->3, 3 fissioned to maximize parallelism
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
            'Eq(ti1[x,y,z], ti3[x,y,z])',
            'Eq(ti3[x,y,z], ti1[x,y,z+1] + 1.)'),
            '+++++', ['xyz', 'xyz', 'xyz'], 'xyzzz'),
        # 1) WAR 1->2, 2->3
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
            'Eq(ti1[x,y,z], ti0[x,y,z+1])',
            'Eq(ti3[x,y,z], ti1[x,y,z-2] + 1.)'),
            '+++++', ['xyz', 'xyz', 'xyz'], 'xyzzz'),
        # 2) WAR 1->2, 2->3, RAW 2->3
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
            'Eq(ti1[x,y,z], ti0[x,y,z+1])',
            'Eq(ti3[x,y,z], ti1[x,y,z-2] + ti1[x,y,z+2])'),
            '+++++', ['xyz', 'xyz', 'xyz'], 'xyzzz'),
        # 3) WAR 1->3
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
            'Eq(ti1[x,y,z], ti3[x,y,z])',
            'Eq(ti3[x,y,z], ti0[x,y,z+1] + 1.)'),
            '++++', ['xyz', 'xyz'], 'xyzz'),
        # 4) WAR 1->3
        # Like before, but the WAR is along `y`, an inner Dimension
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
            'Eq(ti1[x,y,z], ti3[x,y,z])',
            'Eq(ti3[x,y,z], ti0[x,y+1,z] + 1.)'),
            '+++++', ['xyz', 'xyz'], 'xyzyz'),
        # 5) WAR 1->2, 2->3; WAW 1->3
        # Similar to the cases above, but the last equation does not iterate over `z`
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
            'Eq(ti1[x,y,z], ti0[x,y,z+2])',
            'Eq(ti0[x,y,0], ti0[x,y,0] + 1.)'),
            '++++', ['xyz', 'xyz', 'xy'], 'xyzz'),
        # 6) WAR 1->2; WAW 1->3
        # Basically like above, but with the time dimension. This should have no impact
        (('Eq(tu[t,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
            'Eq(tv[t,x,y,z], tu[t,x,y,z+2])',
            'Eq(tu[t,x,y,0], tu[t,x,y,0] + 1.)'),
            '+++++', ['txyz', 'txyz', 'txy'], 'txyzz'),
        # 7) WAR 1->2, 2->3
        (('Eq(tu[t,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
            'Eq(tv[t,x,y,z], tu[t,x,y,z+2])',
            'Eq(tw[t,x,y,z], tv[t,x,y,z-1] + 1.)'),
            '++++++', ['txyz', 'txyz', 'txyz'], 'txyzzz'),
        # 8) WAR 1->2; WAW 1->3
        (('Eq(tu[t,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
            'Eq(tv[t,x,y,z], tu[t,x+2,y,z])',
            'Eq(tu[t,3,y,0], tu[t,3,y,0] + 1.)'),
            '++++++++', ['txyz', 'txyz', 'ty'], 'txyzxyzy'),
        # 9) RAW 1->2, WAR 2->3
        (('Eq(tu[t,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
            'Eq(tv[t,x,y,z], tu[t,x,y,z-2])',
            'Eq(tw[t,x,y,z], tv[t,x,y+1,z] + 1.)'),
            '++++++++', ['txyz', 'txyz', 'txyz'], 'txyzyzyz'),
        # 10) WAR 1->2; WAW 1->3
        (('Eq(tu[t-1,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
            'Eq(tv[t,x,y,z], tu[t,x,y,z+2])',
            'Eq(tu[t-1,x,y,0], tu[t,x,y,0] + 1.)'),
            '-+++', ['txyz', 'txy'], 'txyz'),
        # 11) WAR 1->2
        (('Eq(tu[t-1,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
            'Eq(tv[t,x,y,z], tu[t,x,y,z+2] + tu[t,x,y,z-2])',
            'Eq(tw[t,x,y,z], tv[t,x,y,z] + 2)'),
            '-+++', ['txyz'], 'txyz'),
        # 12) Time goes backward so that information flows in time
        (('Eq(tu[t-1,x,y,z], tu[t,x+3,y,z] + tv[t,x,y,z])',
            'Eq(tv[t-1,x,y,z], tu[t,x,y,z+2])',
            'Eq(tw[t-1,x,y,z], tu[t,x,y+1,z] + tv[t,x,y-1,z])'),
            '-+++', ['txyz'], 'txyz'),
        # 13) Time goes backward so that information flows in time, but the
        # first and last Eqs are interleaved by a completely independent
        # Eq. This results in three disjoint sets of loops
        (('Eq(tu[t-1,x,y,z], tu[t,x+3,y,z] + tv[t,x,y,z])',
            'Eq(ti0[x,y,z], ti1[x,y,z+2])',
            'Eq(tw[t-1,x,y,z], tu[t,x,y+1,z] + tv[t,x,y-1,z])'),
            '-++++++++++', ['txyz', 'xyz', 'txyz'], 'txyzxyztxyz'),
        # 14) Time goes backward so that information flows in time
        (('Eq(ti0[x,y,z], ti1[x,y,z+2])',
            'Eq(tu[t-1,x,y,z], tu[t,x+3,y,z] + tv[t,x,y,z])',
            'Eq(tw[t-1,x,y,z], tu[t,x,y+1,z] + ti0[x,y-1,z])'),
            '+++-+++', ['xyz', 'txyz'], 'xyztxyz'),
        # 15) WAR 2->1
        # Here the difference is that we're using SubDimensions
        (('Eq(tv[t,xi,yi,zi], tu[t,xi-1,yi,zi] + tu[t,xi+1,yi,zi])',
            'Eq(tu[t+1,xi,yi,zi], tu[t,xi,yi,zi] + tv[t,xi-1,yi,zi] + tv[t,xi+1,yi,zi])'),
            '+++++++', ['ti0xi0yi0z', 'ti0xi0yi0z'], 'ti0xi0yi0zi0xi0yi0z'),
        # 16) RAW 3->1; expected=2
        # Time goes backward, but the third equation should get fused with
        # the first one, as the time dependence is loop-carried
        (('Eq(tv[t-1,x,y,z], tv[t,x-1,y,z] + tv[t,x+1,y,z])',
            'Eq(tv[t-1,z,z,z], tv[t-1,z,z,z] + 1)',
            'Eq(f[x,y,z], tu[t-1,x,y,z] + tu[t,x,y,z] + tu[t+1,x,y,z] + tv[t,x,y,z])'),
            '-++++', ['txyz', 'tz'], 'txyzz'),
        # 17) WAR 2->3, 2->4; expected=4
        (('Eq(tu[t+1,x,y,z], tu[t,x,y,z] + 1.)',
            'Eq(tu[t+1,y,y,y], tu[t+1,y,y,y] + tw[t+1,y,y,y])',
            'Eq(tw[t+1,z,z,z], tw[t+1,z,z,z] + 1.)',
            'Eq(tv[t+1,x,y,z], tu[t+1,x,y,z] + 1.)'),
            '+++++++++', ['txyz', 'ty', 'tz', 'txyz'], 'txyzyzxyz'),
        # 18) WAR 1->3; expected=3
        # 5 is expected to be moved before 4 but after 3, to be merged with 3
        (('Eq(tu[t+1,x,y,z], tv[t,x,y,z] + 1.)',
            'Eq(tv[t+1,x,y,z], tu[t,x,y,z] + 1.)',
            'Eq(tw[t+1,x,y,z], tu[t+1,x+1,y,z] + tu[t+1,x-1,y,z])',
            'Eq(f[x,x,z], tu[t,x,x,z] + tw[t,x,x,z])',
            'Eq(ti0[x,y,z], tw[t+1,x,y,z] + 1.)'),
            '++++++++', ['txyz', 'txyz', 'txz'], 'txyzxyzz'),
        # 19) WAR 1->3; expected=3
        # Cannot merge 1 with 3 otherwise we would break an anti-dependence
        (('Eq(tv[t+1,x,y,z], tu[t,x,y,z] + tu[t,x+1,y,z])',
            'Eq(tu[t+1,xi,yi,zi], tv[t+1,xi,yi,zi] + tv[t+1,xi+1,yi,zi])',
            'Eq(tw[t+1,x,y,z], tv[t+1,x,y,z] + tv[t+1,x+1,y,z])'),
            '++++++++++', ['txyz', 'ti0xi0yi0z', 'txyz'], 'txyzi0xi0yi0zxyz'),
    ])
    def test_consistency_anti_dependences(self, exprs, directions, expected, visit):
        """
        Test that anti dependences end up generating multi loop nests, rather
        than a single loop nest enclosing all of the equations.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions  # noqa
        xi, yi, zi = grid.interior.dimensions  # noqa
        t = grid.stepping_dim  # noqa

        ti0 = Array(name='ti0', shape=grid.shape, dimensions=grid.dimensions)  # noqa
        ti1 = Array(name='ti1', shape=grid.shape, dimensions=grid.dimensions)  # noqa
        ti3 = Array(name='ti3', shape=grid.shape, dimensions=grid.dimensions)  # noqa
        f = Function(name='f', grid=grid)  # noqa
        tu = TimeFunction(name='tu', grid=grid)  # noqa
        tv = TimeFunction(name='tv', grid=grid)  # noqa
        tw = TimeFunction(name='tw', grid=grid)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        eqns = []
        for e in exprs:
            eqns.append(eval(e))

        # Note: `topofuse` is a subset of `advanced` mode. We use it merely to
        # bypass 'blocking', which would complicate the asserts below
        op = Operator(eqns, opt='xdsl')
        op.apply(time_M=1)


def test_xdsl_mul_eqs_VI():
    # Define a Devito Operator with multiple eqs
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name="u", grid=grid, time_order=2)
    v = TimeFunction(name="v", grid=grid, time_order=2)

    u.data[:, :, :] = np.random.rand(*u.shape)
    v.data[:, :, :] = np.random.rand(*v.shape)

    u_init = u.data[:, :, :]
    v_init = v.data[:, :, :]

    eq0 = Eq(u.forward, u + 2)
    eq1 = Eq(v, u.forward * 2)

    op = Operator([eq0, eq1], opt="advanced")

    op.apply(time_M=4, dt=0.1)

    devito_res_u = u.data_with_halo[:, :, :]
    devito_res_v = v.data_with_halo[:, :, :]

    u.data[:, :, :] = u_init
    v.data[:, :, :] = v_init

    op = Operator([eq0, eq1], opt="xdsl")

    op.apply(time_M=4, dt=0.1)

    assert np.isclose(u.data_with_halo, devito_res_u).all()
    assert np.isclose(v.data_with_halo, devito_res_v).all()


def test_xdsl_mul_eqs_VII():
    # Define a Devito Operator with multiple eqs
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name="u", grid=grid, time_order=2)
    v = TimeFunction(name="v", grid=grid, time_order=2)

    u.data[:, :, :] = 0.1
    v.data[:, :, :] = 0.1

    eq0 = Eq(u.forward, u + 2)
    eq1 = Eq(v, u.forward * 2)

    op = Operator([eq0, eq1], opt="noop")
    op.apply(time_M=4, dt=0.1)

    devito_res_u = u.data_with_halo[:, :, :]
    devito_res_v = v.data_with_halo[:, :, :]

    u.data[:, :, :] = 0.1
    v.data[:, :, :] = 0.1

    xdsl_op = Operator([eq0, eq1], opt="xdsl")
    xdsl_op.apply(time_M=4, dt=0.1)
    assert np.isclose(norm(u), np.linalg.norm(devito_res_u))
    assert np.isclose(norm(v), np.linalg.norm(devito_res_v))


def test_xdsl_mul_eqs_VIII():
    # Define a Devito Operator with multiple eqs
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name="u", grid=grid, time_order=2)
    v = TimeFunction(name="v", grid=grid, time_order=2)

    u.data[:, :, :] = 0.1
    v.data[:, :, :] = 0.1

    eq0 = Eq(v, u.forward.dx * 2)

    op = Operator([eq0], opt="advanced")
    op.apply(time_M=4, dt=0.1)

    devito_res_u = u.data_with_halo[:, :, :]
    devito_res_v = v.data_with_halo[:, :, :]

    u.data[:, :, :] = 0.1
    v.data[:, :, :] = 0.1

    op = Operator([eq0], opt="xdsl")
    op.apply(time_M=4, dt=0.1)

    assert np.isclose(norm(u), np.linalg.norm(devito_res_u))
    assert np.isclose(norm(v), np.linalg.norm(devito_res_v))


def test_function():
    grid = Grid(shape=(2, 2))
    x, y = grid.dimensions

    f = Function(name="f", grid=grid)

    eqns = [Eq(f, 2.0)]

    op = Operator(eqns, opt="xdsl")
    op.apply()

    assert np.all(f.data == 2.0)


def test_function_II():
    # Define a Devito Operator with multiple eqs
    grid = Grid(shape=(4, 4))

    u = Function(name="u", grid=grid)
    v = TimeFunction(name="v", grid=grid)
    w = Function(name="w", grid=grid)

    u.data[:, :] = 1.0
    v.data[:, :] = 2.0

    eq0 = Eq(u, v * v)
    eq1 = Eq(w, v.dt)

    op = Operator([eq0, eq1], opt="advanced")
    op.apply(time_M=4, dt=0.1)

    devito_norm_u = np.linalg.norm(u.data_with_halo)
    devito_norm_v = np.linalg.norm(v.data_with_halo)

    u.data[:, :] = 1.0
    v.data[:, :] = 2.0

    op = Operator([eq0, eq1], opt="xdsl")
    op.apply(time_M=4, dt=0.1)

    assert np.isclose(norm(u), devito_norm_u)
    assert np.isclose(norm(v), devito_norm_v)


def test_function_III():
    # Define a Devito Operator with multiple eqs
    grid = Grid(shape=(4, 4))

    u = Function(name="u", grid=grid)
    v = TimeFunction(name="v", grid=grid)
    w = TimeFunction(name="w", grid=grid)

    u.data[:, :] = 1.0
    v.data[:, :] = 2.0

    eq0 = Eq(v.forward, u * u + w)

    op = Operator([eq0], opt="advanced")
    op.apply(time_M=4, dt=0.1)

    devito_norm_u = np.linalg.norm(u.data_ro_with_halo)
    devito_norm_v = np.linalg.norm(v.data_ro_with_halo)

    u.data[:, :] = 1.0
    v.data[:, :] = 2.0

    op = Operator([eq0], opt="xdsl")
    op.apply(time_M=4, dt=0.1)

    assert np.isclose(norm(u), devito_norm_u)
    assert np.isclose(norm(v), devito_norm_v)


@pytest.mark.xfail(reason="Operation does not verify: Cannot Load and Store the same field!")  # noqa
def test_function_IV():
    # Define a Devito Operator with multiple eqs
    grid = Grid(shape=(4, 4))

    u = Function(name="u", grid=grid)
    w = Function(name="w", grid=grid)

    u.data[:, :] = 1.0

    eq0 = Eq(w, u * u + w)

    op = Operator([eq0], opt="advanced")
    op.apply()

    devito_norm_u = np.linalg.norm(u.data_ro_with_halo)

    u.data[:, :] = 1.0

    op = Operator([eq0], opt="xdsl")
    op.apply()

    assert np.isclose(norm(u), devito_norm_u)


class TestOperatorUnsupported(object):

    @pytest.mark.xfail(reason="stencil.return operation does not verify for i64")
    def test_forward_assignment(self):
        # simple forward assignment

        grid = Grid(shape=(4, 4))
        u = TimeFunction(name="u", grid=grid, space_order=2)
        u.data[:, :, :] = 0

        eq0 = Eq(u.forward, 1)

        op = Operator([eq0], opt='xdsl')

        op.apply(time_M=1)

        assert np.isclose(norm(u), 5.6584, rtol=0.001)

    @pytest.mark.xfail(reason="stencil.return operation does not verify for i64")
    def test_function(self):
        grid = Grid(shape=(5, 5))
        x, y = grid.dimensions

        f = Function(name="f", grid=grid)

        eqns = [Eq(f, 2)]

        op = Operator(eqns, opt='xdsl')
        op.apply()

        assert np.all(f.data == 4)


class TestElastic():

    @pytest.mark.parametrize('shape', [(101, 101), (201, 201), (301, 301)])
    @pytest.mark.parametrize('so', [2, 4, 8])
    @pytest.mark.parametrize('nt', [10, 20, 50, 100])
    def test_elastic_2D(self, shape, so, nt):

        # Initial grid: km x km, with spacing
        shape = shape  # Number of grid point (nx, nz)
        spacing = as_tuple(10.0 for _ in range(len(shape)))
        extent = tuple([s*(n-1) for s, n in zip(spacing, shape)])

        x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))  # noqa
        z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[1]/(shape[1]-1)))  # noqa
        grid = Grid(extent=extent, shape=shape, dimensions=(x, z))

        # To be checked again in the future
        # class DGaussSource(WaveletSource):

        #    def wavelet(self, f0, t):
        #        a = 0.004
        #        return -2.*a*(t - 1/f0) * np.exp(-a * (t - 1/f0)**2)

        # Timestep size from Eq. 7 with V_p=6000. and dx=100
        t0, tn = 0., nt
        dt = (10. / np.sqrt(2.)) / 6.
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        src = RickerSource(name='src', grid=grid, f0=0.01, time_range=time_range)
        src.coordinates.data[:] = [250., 250.]

        # Now we create the velocity and pressure fields
        v = VectorTimeFunction(name='v', grid=grid, space_order=so, time_order=1)
        tau = TensorTimeFunction(name='t', grid=grid, space_order=so, time_order=1)

        # We need some initial conditions
        V_p = 2.0
        V_s = 1.0
        density = 1.8

        # The source injection term
        src_xx = src.inject(field=tau.forward[0, 0], expr=src)
        src_zz = src.inject(field=tau.forward[1, 1], expr=src)

        # Thorbecke's parameter notation
        cp2 = V_p*V_p
        cs2 = V_s*V_s
        ro = 1/density

        mu = cs2*density
        l = (cp2*density - 2*mu)

        # First order elastic wave equation
        pde_v = v.dt - ro * div(tau)
        pde_tau = (tau.dt - l * diag(div(v.forward)) - mu * (grad(v.forward) +
                   grad(v.forward).transpose(inner=False)))

        # Time update
        u_v = Eq(v.forward, solve(pde_v, v.forward))
        u_t = Eq(tau.forward, solve(pde_tau, tau.forward))

        # Inject sources. We use it to preinject data
        # Up to here, let's only use Devito
        op = Operator([u_v] + [u_t] + src_xx + src_zz)
        op(dt=dt)

        op = Operator([u_v] + [u_t], opt='xdsl')
        op(dt=dt, time_M=nt)

        xdsl_norm_v0 = norm(v[0])
        xdsl_norm_v1 = norm(v[1])
        xdsl_norm_tau0 = norm(tau[0])
        xdsl_norm_tau1 = norm(tau[1])
        xdsl_norm_tau2 = norm(tau[2])
        xdsl_norm_tau3 = norm(tau[3])

        # Reinitialize the fields to zero

        v[0].data[:] = 0
        v[1].data[:] = 0

        tau[0].data[:] = 0
        tau[1].data[:] = 0
        tau[2].data[:] = 0
        tau[3].data[:] = 0

        # Inject sources. We use it to preinject data
        op = Operator([u_v] + [u_t] + src_xx + src_zz)
        op(dt=dt)

        op = Operator([u_v] + [u_t], opt='advanced')
        op(dt=dt, time_M=nt)

        dv_norm_v0 = norm(v[0])
        dv_norm_v1 = norm(v[1])
        dv_norm_tau0 = norm(tau[0])
        dv_norm_tau1 = norm(tau[1])
        dv_norm_tau2 = norm(tau[2])
        dv_norm_tau3 = norm(tau[3])

        assert np.isclose(xdsl_norm_v0, dv_norm_v0, rtol=1e-04)
        assert np.isclose(xdsl_norm_v1, dv_norm_v1, rtol=1e-04)
        assert np.isclose(xdsl_norm_tau0, dv_norm_tau0, rtol=1e-04)
        assert np.isclose(xdsl_norm_tau1, dv_norm_tau1, rtol=1e-04)
        assert np.isclose(xdsl_norm_tau2, dv_norm_tau2, rtol=1e-04)
        assert np.isclose(xdsl_norm_tau3, dv_norm_tau3, rtol=1e-04)

        assert not np.isclose(xdsl_norm_v0, 0.0, rtol=1e-04)
        assert not np.isclose(xdsl_norm_v1, 0.0, rtol=1e-04)
        assert not np.isclose(xdsl_norm_tau0, 0.0, rtol=1e-04)
        assert not np.isclose(xdsl_norm_tau1, 0.0, rtol=1e-04)
        assert not np.isclose(xdsl_norm_tau2, 0.0, rtol=1e-04)
        assert not np.isclose(xdsl_norm_tau3, 0.0, rtol=1e-04)
