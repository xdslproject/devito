from devito import Dimension, ConditionalDimension, Function, TimeFunction, Eq, Operator, Grid
from devito.types import Le
from devito.symbolics import CondEq
from sympy import And
import pdb
nx, ny = 4 , 4

shape = (nx, ny)
grid = Grid(shape=shape)

g = TimeFunction(name='g', grid=grid)
h = TimeFunction(name='h', grid=grid)

assert g.grid == h.grid
assert g.grid == grid