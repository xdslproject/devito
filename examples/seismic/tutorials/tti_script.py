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

shape = args.shape  # Number of grid point (nx, nz)
spacing = (10., 10.)  # spacing of 10 meters
origin = (0., 0.)
nbl = 1  # number of pad layers

model = demo_model('layers-tti', spacing=spacing, space_order=4,
                   shape=shape, nbl=nbl, nlayers=nbl)

# Compute the dt and set time range
t0 = 0.  # Simulation time start
tn = args.nt  # Simulation time end (0.15 second = 150 msec)
dt = model.critical_dt

time_range = TimeAxis(start=t0, stop=tn, step=dt)
print("time_range; ", time_range)

# time stepping 
p = TimeFunction(name="p", grid=model.grid, time_order=2, space_order=8)

# Main equations
stencil_p = solve(p.dt2, p.forward)
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

optime = Operator([update_p] + src_term)
optime.apply(time=time_range.num-1, dt=dt)
print(norm(p))


p.data[:] = 0.

optime = Operator([update_p] + src_term, opt='xdsl')
optime.apply(time=time_range.num-1, dt=dt)
print(norm(p))

import pdb; pdb.set_trace()
