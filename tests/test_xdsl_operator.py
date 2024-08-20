from devito import Grid, TimeFunction, Eq, Operator

from xdsl.dialects.scf import For, Yield
from xdsl.dialects.arith import Addi
from xdsl.dialects.func import Call, Return
from xdsl.dialects.stencil import FieldType, ApplyOp, LoadOp, StoreOp
from xdsl.dialects.llvm import LLVMPointerType


def test_create_xdsl_operator():
    # Create a simple Devito Operator
    # and check structure

    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    xdsl_op = Operator([eq], opt='xdsl')

    assert xdsl_op.name == 'Kernel'
    assert len(xdsl_op._module.regions[0].blocks[0].ops) == 3

    xdsl_op.apply(time_M=5)

    assert len(xdsl_op._module.regions[0].blocks[0].ops.first.body.blocks[0].ops) == 10

    assert isinstance(xdsl_op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args[0].type, FieldType)  # noqa
    assert isinstance(xdsl_op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args[1].type, FieldType)  # noqa
    assert isinstance(xdsl_op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args[2].type, LLVMPointerType)  # noqa

    ops = list(xdsl_op._module.regions[0].blocks[0].ops.first.body.blocks[0].ops)
    assert type(ops[5] == Addi)
    assert type(ops[6] == For)

    scffor_ops = list(ops[6].regions[0].blocks[0].ops)
    assert len(scffor_ops) == 5

    # First
    assert isinstance(scffor_ops[0], LoadOp)
    assert isinstance(scffor_ops[1], ApplyOp)
    assert isinstance(scffor_ops[2], StoreOp)
    assert isinstance(scffor_ops[3], LoadOp)

    # Yield
    assert isinstance(scffor_ops[4], Yield)

    assert type(ops[7] == Call)
    assert type(ops[8] == StoreOp)
    assert type(ops[9] == Return)
