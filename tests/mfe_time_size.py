import numpy as np

from devito import TimeFunction, Operator, Grid, Eq

nx, ny = 4, 4

shape = (nx, ny)
grid = Grid(shape=shape)

x, y = grid.dimensions
time = grid.time_dim

f = TimeFunction(name='f', grid=grid, dtype=np.int32)

eq0 = Eq(f.forward, f + 1)

op = Operator([eq0])

print(op.ccode)
op.apply(time_M=40)
