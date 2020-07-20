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
h = Function(name='h', grid=grid, dimensions=(x,), shape=(shape[0],))

cond = Le(h, 5)
ci = ConditionalDimension(name='ci', parent=x, condition=cond)

eq0 = Eq(h, g[x, 1])
eq1 = Eq(g, g + 1, implicit_dims=ci)
op = Operator([eq0, eq1])
print(op.ccode)
op.apply()