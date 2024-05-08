import numpy as np
import pytest

from devito import Grid, TimeFunction, Eq, Operator, solve, norm
from devito.types import Symbol

from xdsl.dialects.scf import For, Yield
from xdsl.dialects.arith import Addi
from xdsl.dialects.func import Call, Return
from xdsl.dialects.stencil import FieldType, ApplyOp, LoadOp, StoreOp
from xdsl.dialects.llvm import LLVMPointerType


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

    def test_xdsl_mul_eqs_III(self):
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

    def test_forward_assignment_f32(self):
        # simple Devito a = 1 operator

        grid = Grid(shape=(4, 4))
        u = TimeFunction(name="u", grid=grid, space_order=2)
        u.data[:, :, :] = 0

        eq0 = Eq(u.forward, 0.1)

        op = Operator([eq0], opt='xdsl')

        op.apply(time_M=1)

        assert np.isclose(norm(u), 0.56584, rtol=0.001)


class TestOperatorUnsupported(object):

    @pytest.mark.xfail(reason="Symbols are not supported in xDSL yet")
    def test_symbol_I(self):
        # Define a simple Devito a = 1 operator

        a = Symbol('a')
        eq0 = Eq(a, 1)

        op = Operator([eq0], opt='xdsl')

        op.apply()

        assert a == 1

    @pytest.mark.xfail(reason="stencil.return operation does not verify f32 works but not i64")
    def test_forward_assignment(self):
        # simple forward assignment

        grid = Grid(shape=(4, 4))
        u = TimeFunction(name="u", grid=grid, space_order=2)
        u.data[:, :, :] = 0

        eq0 = Eq(u.forward, 1)

        op = Operator([eq0], opt='xdsl')

        op.apply(time_M=1)

        assert np.isclose(norm(u), 5.6584, rtol=0.001)
