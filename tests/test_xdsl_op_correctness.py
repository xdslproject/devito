import numpy as np
from devito import Grid, TimeFunction, Eq, Operator, norm
import pytest
# flake8: noqa

from xdsl.dialects.scf import For, Yield
from xdsl.dialects.arith import Addi, Constant
from xdsl.dialects.func import Call, Return
from xdsl.dialects.stencil import FieldType, ApplyOp, LoadOp, StoreOp
from xdsl.dialects.llvm import LLVMPointerType
from xdsl.printer import Printer

from devito.types.basic import Symbol

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
    
    assert np.isclose(norm1, norm2,   atol=1e-5, rtol=0)
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
