from devito import Grid, Dimension, Eq, Function, TimeFunction, Operator, solve # noqa
from devito.ir import Iteration, FindNodes

from matplotlib.pyplot import pause # noqa
import numpy as np

nx = 64
ny = 64
nz = 64
nt = 38
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
                                     'blocktime': False, 'blocklevels': 1}))

op.apply(time_M=time_M, dt=dt)
print(np.linalg.norm(u.data))
u.data[:] = init_value
op1 = Operator(eq0, opt=('advanced', {'skewing': True, 'openmp': True,
                         'blocktime': True}))

op1.apply(time_M=time_M, dt=dt)
print(np.linalg.norm(u.data))

u.data[:] = init_value
op2 = Operator(eq0, opt=('advanced', {'openmp': True, 'skewing': True,
                                      'blocktime': True, 'blocklevels': 2}))

op2.apply(time_M=time_M, dt=dt)

print(np.linalg.norm(u.data))
u.data[:] = init_value

# List comprehension would need explicit locals/globals mappings to eval
op3 = Operator(eq0, opt=('advanced', {'openmp': True, 'skewing': True,
                                      'blocktime': True, 'blocklevels': 3}))

op3.apply(time_M=time_M, dt=dt)
print(np.linalg.norm(u.data))
# assert np.isclose(norm(u), 2467.3872, atol=1e-3, rtol=0)
u.data[:] = init_value

# List comprehension would need explicit locals/globals mappings to eval
op4 = Operator(eq0, opt=('advanced', {'openmp': True, 'skewing': True,
                                      'blocktime': True, 'blocklevels': 4}))

op4.apply(time_M=time_M, dt=dt)
# assert np.isclose(norm(u), 2467.3872, atol=1e-3, rtol=0)
print(np.linalg.norm(u.data))


iters = FindNodes(Iteration).visit(op2)
time_iter = [i for i in iters if i.dim.is_Time]

assert len(time_iter) == 2
