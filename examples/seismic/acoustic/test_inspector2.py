import numpy as np
from devito import TimeFunction, Function, Dimension, Eq, Grid, ConditionalDimension
from devito import Operator
from devito.types.basic import Scalar

# Some variable declarations
nx = 20
ny = 20

# Define a physical size
shape = (nx, ny)

grid = Grid(shape=shape)

f = TimeFunction(name="f", grid=grid, space_order=2, time_order=0)
f2 = Function(name="f2", grid=grid, space_order=2)
time = grid.time_dim
t, x, y = f.dimensions
assert(f2.dimensions == f.dimensions[1:])

z = Dimension(name="z")
usc = Function(name="usc", shape=(1,), dimensions=(z,), dtype=np.int32)

eq1 = Eq(usc[0], f, implicit_dims=(time, x, y, z))

op2 = Operator([eq1])
print(op2.ccode)
op2.apply(time_M=3)
import pdb; pdb.set_trace()
