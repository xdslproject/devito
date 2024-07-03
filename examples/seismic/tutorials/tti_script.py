# # TTI pure qP-wave equation implementation

# First of all, it is necessary to import some Devito modules and other packages that will be used in the implementation.
import argparse
import numpy as np
from devito import (Function, TimeFunction, cos, sin, solve,
                    Eq, Operator)
from examples.seismic import TimeAxis, RickerSource, demo_model
from devito import norm


parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(110, 110), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=2,
                    type=int, help="Space order of the simulation")
parser.add_argument("-nt", "--nt", default=40,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-devito", "--devito", default=False, type=bool, help="Devito run")
parser.add_argument("-xdsl", "--xdsl", default=False, type=bool, help="xDSL run")
parser.add_argument("-plot", "--plot", default=False, type=bool, help="Plot2D")

args = parser.parse_args()

# We will start with the definitions of the grid and the physical parameters $v_{p}, \theta, \epsilon, \delta$. For simplicity, we won't use any absorbing boundary conditions to avoid reflections of outgoing waves at the boundaries of the computational domain, but we will have boundary conditions (zero Dirichlet) at $x=0,nx$ and $z=0,nz$ for the solution of the Poisson equation. We use a homogeneous model. The model is discretized with a grid of $101 \times 101$ and spacing of 10 m. The $v_{p}, \epsilon, \delta$ and $\theta$ parameters of this model are 3600 m∕s, 0.23, 0.17, and 45°, respectively. 

# NBVAL_IGNORE_OUTPUT   

shape = args.shape  # Number of grid point (nx, nz)
spacing = (10., 10.)  # spacing of 10 meters
origin = (0., 0.)
nbl = 1  # number of pad layers

model = demo_model('layers-tti', spacing=spacing, space_order=4,
                   shape=shape, nbl=nbl, nlayers=nbl)

# Compute the dt and set time range
t0 = 0.  # Simulation time start
tn = 150.  # Simulation time end (0.15 second = 150 msec)

# dt = (dvalue/(np.pi*vmax))*np.sqrt(1/(1+etamax*(max_cos_sin)**2))  # eq. above (cell 3)
dt = 0.88

time_range = TimeAxis(start=t0, stop=tn, step=dt)
print("time_range; ", time_range)

# time stepping 
p = TimeFunction(name="p", grid=model.grid, time_order=1, space_order=4)
q = Function(name="q", grid=model.grid, space_order=4)

# Main equations
stencil_p = solve(p.dt, p.forward)
update_p = Eq(p.forward, stencil_p)

# Create stencil and boundary condition expressions
x, z = model.grid.dimensions
t = model.grid.stepping_dim

# set source and receivers
src = RickerSource(name='src', grid=model.grid, f0=0.02, npoint=1, time_range=time_range)

src.coordinates.data[:, 0] = model.domain_size[0] * .5
src.coordinates.data[:, 1] = model.domain_size[0] * .5
# Define the source injection
src_term = src.inject(field=p.forward, expr=src)

# optime = Operator([update_p], opt='xdsl')
optime = Operator([update_p] + src_term, opt='xdsl')
optime.apply(time=time_range.num-1, dt=dt)
print(norm(p))


p.data[:] = 0.

optime = Operator([update_p] + src_term)
optime.apply(time=time_range.num-1, dt=dt)
print(norm(p))

import pdb; pdb.set_trace()
