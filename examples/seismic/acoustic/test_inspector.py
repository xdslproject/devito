import numpy as np
from devito import TimeFunction, Function, Dimension, Eq, Grid
from devito import Operator
from devito.types.basic import Scalar

# Some variable declarations
nx = 20
ny = 20
nz = 20
# Define a physical size
shape = (nx, ny, nz)

grid = Grid(shape=shape)

f = TimeFunction(name="f", grid=grid, space_order=2)

f2 = Function(name="f2", grid=grid, space_order=2)

t, x, y, z = f.dimensions
assert(f2.dimensions == f.dimensions[1:])

zi = Dimension(name="zi")
usc = Function(name="usc", shape=(1,), dimensions=(zi,), dtype=np.int32)

eq0 = Eq(f.forward, f)
eq1 = Eq(usc[0], f.forward[0, x, y, z])

op2 = Operator([eq0, eq1])
print(op2.ccode)
op2.apply(time_M=20)
# import pdb; pdb.set_trace()
