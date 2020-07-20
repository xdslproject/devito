import numpy as np

from devito import TimeFunction, Operator, Grid, Eq, Dimension

nx, ny = 4, 4

shape = (nx, ny)
grid = Grid(shape=shape)

x, y = grid.dimensions
time = grid.time_dim
id = Dimension(name='id')
t = grid.stepping_dim
f = TimeFunction(name='f', grid=grid, dtype=np.int32)
g = TimeFunction(name='g', grid=grid, shape=(40, 10), dimensions= (time, id),dtype=np.int32)

eq0 = Eq(f.forward, f + g)

op = Operator([eq0])

print(op.ccode)
import pdb; pdb.set_trace()
op.apply(time_M=7)
