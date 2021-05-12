from devito import *

grid = Grid(shape=(10, 10))
x, y = grid.dimensions
t = grid.stepping_dim

u = TimeFunction(name='u', grid=grid)
u1 = TimeFunction(name='u', grid=grid)

u.data[:] = 1.
u1.data[:] = 1.

eq = Eq(u.forward, u[t, x-1, y] + u[t, x+1, y] + u[t, x, y-1] + u[t, x, y+1])

op0 = Operator(eq, name='Jacobi')
op1 = Operator(eq, name='IsoFwdOperator', compiler='cuda')

op0.apply(time_M=2)
op1.apply(time_M=2, u=u1)
from IPython import embed; embed()
