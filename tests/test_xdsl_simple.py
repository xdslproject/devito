from devito import Grid, TimeFunction, Eq, XDSLOperator, Operator
# flake8: noqa
from devito.operator.xdsl_operator import XDSLOperator


def test_create_xdsl_operator():

    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u + 1)
    xdsl_op = XDSLOperator([eq])
    xdsl_op.__class__ = XDSLOperator
    xdsl_op.apply(time_M=5)

    op = Operator([eq])
    op.apply(time_M=5)

    # TOFIX to add proper test
    # assert(str(op.ccode) == xdsl_op.ccode)
