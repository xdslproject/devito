from devito import Grid, TimeFunction, Eq, Operator


def test_create_xdsl_operator():

    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    xdsl_op = Operator([eq], opt='xdsl')

    assert xdsl_op.name == 'Kernel'
    assert len(xdsl_op._module.regions[0].blocks[0].ops) == 3

    xdsl_op.apply(time_M=5)
