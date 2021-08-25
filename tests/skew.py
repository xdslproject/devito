from devito import Grid, Dimension, Eq, Function, TimeFunction, Operator, solve # noqa
from devito.ir import Iteration, FindNodes

from matplotlib.pyplot import pause # noqa
import matplotlib.pyplot as plt
import numpy as np

nx = 32
ny = 32
nz = 32
nt = 24
nu = .5
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)
dz = 2. / (nz - 1)
sigma = .25
dt = sigma * dx * dz * dy / nu

# Initialise u with hat function
init_value = 50

# Field initialization
grid = Grid(shape=(nx, ny, nz))
u = TimeFunction(name='u', grid=grid, space_order=4)
u.data[:, :, :] = init_value

# Create an equation with second-order derivatives
eq = Eq(u.dt, u.dx2 + u.dy2 + u.dz2)
x, y, z = grid.dimensions
stencil = solve(eq, u.forward)
eq0 = Eq(u.forward, stencil)
# eq0 = Eq(u.forward, u+1)
eq0
time_M = nt

# List comprehension would need explicit locals/globals mappings to eval
op = Operator(eq0, opt=('advanced', {'openmp': True,
                                     'wavefront': False, 'blocklevels': 1}))

op.apply(time_M=time_M, dt=dt)
print(np.linalg.norm(u.data))
