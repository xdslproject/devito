from devito import Grid, TimeFunction, Eq, XDSLOperator, Operator, Function
from devito.ir.ietxdsl.xdsl_passes import transform_devito_xdsl_string
from devito.ir import ietxdsl
# flake8: noqa
from devito.operator.xdsl_operator import XDSLOperator
from xdsl.printer import Printer

def test_inc():

    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = Function(name='u', grid=grid)
    eq = Eq(u, u + 1)
    xdsl_op = XDSLOperator([eq])
    xdsl_op.__class__ = XDSLOperator
    xdsl_op.apply()

    op = Operator([eq])
    op.apply(time_M=5)

    print(xdsl_op.ccode)
    # print(op.ccode)

    # assert(str(op.ccode) == xdsl_op.ccode)

def test_inc_mlir():
    grid = Grid(shape=(3, 3))
    u = Function(name='u', grid=grid)
    eq = Eq(u, u + 1)
    op = Operator([eq])
    module = ietxdsl.transform_devito_to_iet_ssa(op)

    p = Printer(target=Printer.Target.MLIR)
    p.print(module)
    import pdb;pdb.set_trace()

    xdsl_op = XDSLOperator([eq])
    xdsl_op.__class__ = XDSLOperator
    print(xdsl_op.ccode)
    xdsl_op.apply()


def test_create_xdsl_operator():

    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    xdsl_op = XDSLOperator([eq])
    xdsl_op.__class__ = XDSLOperator
    xdsl_op.apply(time_M=5)

    op = Operator([eq])
    op.apply(time_M=5)

    print(xdsl_op.ccode)
    # print(op.ccode)

    # assert(str(op.ccode) == xdsl_op.ccode)
