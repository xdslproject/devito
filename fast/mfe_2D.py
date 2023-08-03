# A 2D heat diffusion using Devito
# BC modelling included
# PyVista plotting included

import argparse
import numpy as np

from devito import Grid, TimeFunction, Eq, solve, Operator, Constant, norm, XDSLOperator
from examples.seismic import plot_image
from examples.cfd import init_hat

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(11, 11), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=2,
                    type=int, help="Space order of the simulation")
parser.add_argument("-to", "--time_order", default=1,
                    type=int, help="Time order of the simulation")
parser.add_argument("-nt", "--nt", default=40,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-bls", "--blevels", default=2, type=int, nargs="+",
                    help="Block levels")
parser.add_argument("-plot", "--plot", default=False, type=bool, help="Plot3D")
args = parser.parse_args()

# Some variable declarations
nx, ny = args.shape
nt = args.nt
nu = .5
dx = 1. / (nx - 1)
dy = 1. / (ny - 1)
sigma = .25

dt = sigma * dx * dy / nu
so = args.space_order
to = args.time_order

print("dx %s, dy %s" % (dx, dy))

grid = Grid(shape=(nx, ny), extent=(2., 2.))
u = TimeFunction(name='u', grid=grid, space_order=so)

# Reset our data field and ICs
#init_hat(field=u.data[0], dx=dx, dy=dy, value=1.)
u.data[:, 2:3, 2:3] = 1

a = Constant(name='a')
# Create an equation with second-order derivatives
# eq = Eq(u.dt, a * u.laplace, subdomain=grid.interior)
eq = Eq(u.dt, a * u.laplace)
stencil = solve(eq, u.forward)
eq_stencil = Eq(u.forward, stencil)

# Create boundary condition expressions
x, y = grid.dimensions
t = grid.stepping_dim

initdata = u.data[:]
op = Operator([eq_stencil], name='DevitoOperator')
op.apply(time=nt, dt=dt, a=nu)
print(u.data[0, :])
print("Devito Field norm is:", norm(u))

u.data[:, : , :] = 0
u.data[:, 2:3 , 2:3] = 1
# Reset data and run XDSLOperator
#init_hat(field=u.data[0], dx=dx, dy=dy, value=1.)
xdslop = Operator([eq_stencil], name='XDSLOperator')
xdslop.apply(time=nt, dt=dt, a=nu)
print(u.data[0, :])

print("XDSL Field norm is:", norm(u))