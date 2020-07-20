from devito import Dimension, ConditionalDimension, Function, TimeFunction, Eq, Operator, Grid
from devito.types import Le
from devito.symbolics import CondEq
from sympy import And
import pdb
nx, ny = 4 , 4

shape = (nx, ny)
grid = Grid(shape=shape)

x, y = grid.dimensions

g = Function(name='g', grid=grid)
h = Function(name='h', grid=grid)

cond = Le(g[x, 3], 5)
ci = ConditionalDimension(name='ci', parent=x, condition=cond)

# eq0 = Eq(g[x, 3], 2)
eq1 = Eq(g, g + 1, implicit_dims=ci)
op = Operator([eq1])
print(op.ccode)
op.apply()