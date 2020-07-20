from itertools import product

import numpy as np
from sympy import And
import pytest

from conftest import skipif
from devito import (ConditionalDimension, Grid, Function, TimeFunction, SparseFunction,  # noqa
                    Eq, Operator, Constant, Dimension, SubDimension, switchconfig,
                    SubDomain)
from devito.ir.iet import Expression, Iteration, FindNodes, retrieve_iteration_tree
from devito.symbolics import indexify, retrieve_functions
from devito.types import Array, Lt, Le, Gt, Ge, Ne


@pytest.mark.parametrize('setup_rel, rhs, c1, c2, c3, c4', [
    # Relation, RHS, c1 to c4 used as indexes in assert
    (Lt, 3, 2, 4, 4, -1), (Le, 2, 2, 4, 4, -1), (Ge, 3, 4, 6, 1, 4),
    (Gt, 2, 4, 6, 1, 4), (Ne, 5, 2, 6, 1, 2)
])
def test_relational_classes(setup_rel, rhs, c1, c2, c3, c4):
    """
    Test ConditionalDimension using using conditions based on Relations.
    """

    class InnerDomain(SubDomain):
        name = 'inner'

        def define(self, dimensions):
            return {d: ('middle', 2, 2) for d in dimensions}

    inner_domain = InnerDomain()
    grid = Grid(shape=(8, 8), subdomains=(inner_domain,))
    g = Function(name='g', grid=grid)
    g2 = Function(name='g2', grid=grid)

    g.data[:4, :4] = 1
    g.data[4:, :4] = 2
    g.data[4:, 4:] = 3
    g.data[:4, 4:] = 4
    g2.data[:4, :4] = 1
    g2.data[4:, :4] = 2
    g2.data[4:, 4:] = 3
    g2.data[:4, 4:] = 4

    xi, yi = grid.subdomains['inner'].dimensions

    cond = setup_rel(0.4*g + 0.6*g2, rhs, subdomain=grid.subdomains['inner'])
    ci = ConditionalDimension(name='ci', parent=yi, condition=cond)
    f = Function(name='f', shape=grid.shape, dimensions=(xi, ci))

    eq1 = Eq(f, g + 0.01*g2)
    eq2 = Eq(f, 5)
#     eq3 = Eq(g2, g2)

    op = Operator([eq1, eq2])
    op.apply()
    import pdb; pdb.set_trace()

    assert np.all(f.data[2:6, c1:c2] == 5.)
    assert np.all(f.data[:, c3:c4] < 5.)
