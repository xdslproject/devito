import pytest

from sympy import Add, cos, sin, sqrt  # noqa

from devito.core.autotuning import options  # noqa
from devito import TimeFunction, Grid, Operator, Eq # noqa
from devito.ir import Expression, Iteration, FindNodes


class TestCodeGenSkew(object):

    '''
    Test code generation with skewing, tests adapted from test_operator.py
    '''
    @pytest.mark.parametrize('expr, expected', [
        (['Eq(u.forward, u + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z-time+1],u[t0,x-time+1,y-time+1,z-time+1]+1)']),
        (['Eq(u.forward, v + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z-time+1],v[t0,x-time+1,y-time+1,z-time+1]+1)']),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x-time+1,y-time+1,z-time+1],v[t0,x-time+1,y-time+1,z-time+1]+1)']),
    ])
    def test_skewed_bounds(self, expr, expected):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('blocking', 'skewing'))

        iters = FindNodes(Iteration).visit(op)
        time_iter = [i for i in iters if i.dim.is_Time]
        assert len(time_iter) == 1
        time_iter = time_iter[0]

        for i in ['bf0']:
            assert i in op._func_table
            iters = FindNodes(Iteration).visit(op._func_table[i].root)
            assert len(iters) == 5
            assert iters[0].dim.parent is x
            assert iters[1].dim.parent is y
            assert iters[4].dim is z
            assert iters[2].dim.parent is iters[0].dim
            assert iters[3].dim.parent is iters[1].dim

            assert (iters[2].symbolic_min == (iters[0].dim + time))
            assert (iters[2].symbolic_max == (iters[0].dim + time +
                                              iters[0].dim.symbolic_incr - 1))
            assert (iters[3].symbolic_min == (iters[1].dim + time))
            assert (iters[3].symbolic_max == (iters[1].dim + time +
                                              iters[1].dim.symbolic_incr - 1))

            assert (iters[4].symbolic_min == (iters[4].dim.symbolic_min + time))
            assert (iters[4].symbolic_max == (iters[4].dim.symbolic_max + time))
            skewed = [i.expr for i in FindNodes(Expression).visit(op._func_table[i].root)]
            assert str(skewed[0]).replace(' ', '') == expected
