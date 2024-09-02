# # TTI pure qP-wave equation implementation

# %%
import argparse
import numpy as np
from devito import Function, TimeFunction, solve, Eq, Operator, norm, Grid
from examples.seismic import TimeAxis, demo_model

# Create the argument parser
parser = argparse.ArgumentParser(description='TTI pure qP-wave equation simulation')

# Add the arguments
parser.add_argument('-tn', type=int, default=100, help='Number of timesteps')
parser.add_argument('-d', type=int, nargs=2, default=[10, 10], help='Grid shape')

# Parse the command line arguments
args = parser.parse_args()

# Get the values from the arguments
timesteps = args.tn
shape = tuple(args.d)

shape   = args.d # 101x101 grid
spacing = (10.,10.) # spacing of 10 meters
origin  = (0.,0.)  
nbl = 0  # number of pad points


so = 2 # space order

grid = Grid(shape=shape, extent=(100.0, 100.0),
            origin=origin, dtype=np.float32)

# %%
# Compute the dt and set time range
t0 = 0.  #  Simulation time start
tn = timesteps  #  Simulation time end (0.15 second = 150 msec)

dt = 0.88

time_range = TimeAxis(start=t0, stop=tn, step=dt)
print("time_range; ", time_range)

# time stepping 
p = TimeFunction(name="p", grid=grid, time_order=2, space_order=so)
q = Function(name="q", grid=grid, space_order=so)

# stencil_p = solve(m*p.dt2 - q.dx, p.forward)
stencil_p = solve(p.dt2 - 1.4*q.dx, p.forward)
# update_p = Eq(p.forward, stencil_p)
import pdb;pdb.set_trace()
update_p = Eq(p.forward, stencil_p)

in_value = 0.1

p.data[:] = in_value
opx = Operator([update_p], opt='xdsl')


print("Norm xdsl(p) before is ", norm(p))
opx.apply(time_m=0, time_M=time_range.num-2, dt=dt)
print("Norm xdsl(p) after is ", norm(p))

xnorm0 = np.linalg.norm(p.data[0])
xnorm1 = np.linalg.norm(p.data[1])
xnorm2 = np.linalg.norm(p.data[2])

print("Norm xdsl(p0) after is ", xnorm0)
print("Norm xdsl(p1) after is ", xnorm1)
print("Norm xdsl(p2) after is ", xnorm2)

p.data[:] = in_value

opd = Operator([update_p])
print("Norm devito(p) before is ", norm(p))
opd.apply(time_m=0, time_M=time_range.num-2, dt=dt)
print("Norm devito(p) after is ", norm(p))

dnorm0 = np.linalg.norm(p.data[0])
dnorm1 = np.linalg.norm(p.data[1])
dnorm2 = np.linalg.norm(p.data[2])

print("Norm devito(p0) after is ", dnorm0)
print("Norm devito(p1) after is ", dnorm1)
print("Norm devito(p2) after is ", dnorm2)

import pdb; pdb.set_trace()