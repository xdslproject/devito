from devito import Grid, ConditionalDimension, Function, Eq, Operator
from devito.symbolics import CondEq
import pdb

# Parameters
nx = 4
ny = 4

grid = Grid(shape=(nx, ny))  # Define grid

# Define Function
#g = Function(name='g', shape=grid.shape, dimensions=grid.dimensions)
g = Function(name='g', grid=grid)

x, y = grid.dimensions

ci = ConditionalDimension(name='ci', parent=y, condition=CondEq(g, 0))

f = Function(name='f', shape=grid.shape, dimensions=(x, ci))

op = Operator(Eq(f, g + 1, implicit_dims=ci))
print(op.ccode)
op.apply()
pdb.set_trace()