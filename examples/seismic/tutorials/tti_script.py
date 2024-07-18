# # TTI pure qP-wave equation implementation

# %%
import argparse
import numpy as np
from devito import (Function, TimeFunction, cos, sin, solve,
                    Eq, Operator, configuration, norm)
from examples.seismic import TimeAxis, RickerSource, Receiver, demo_model
from matplotlib import pyplot as plt

# Create the argument parser
parser = argparse.ArgumentParser(description='TTI pure qP-wave equation simulation')

# Add the arguments
parser.add_argument('-tn', type=int, default=100, help='Number of timesteps')
parser.add_argument('-d', type=int, nargs=2, default=[101, 101], help='Grid shape')

# Parse the command line arguments
args = parser.parse_args()

# Get the values from the arguments
timesteps = args.tn
shape = tuple(args.d)

shape   = args.d # 101x101 grid
spacing = (10.,10.) # spacing of 10 meters
origin  = (0.,0.)  
nbl = 0  # number of pad points

model = demo_model('layers-tti', spacing=spacing, space_order=8,
                   shape=shape, nbl=nbl, nlayers=0)

# initialize Thomsem parameters to those used in Mu et al., (2020)
model.update('vp', np.ones(shape)*3.6) # km/s
model.update('epsilon', np.ones(shape)*0.23)
model.update('delta', np.ones(shape)*0.17)
model.update('theta', np.ones(shape)*(45.*(np.pi/180.))) # radians

# %% [markdown]
# In cell below, symbols used in the PDE definition are obtained from the `model` object. Note that trigonometric functions proper of Devito are exploited.

# %%
# Get symbols from model
theta = model.theta
delta = model.delta
epsilon = model.epsilon
m = model.m

#  Use trigonometric functions from Devito
costheta  = cos(theta)
sintheta  = sin(theta)
cos2theta = cos(2*theta)
sin2theta = sin(2*theta)
sin4theta = sin(4*theta)

# NBVAL_IGNORE_OUTPUT

# Values used to compute the time sampling
epsilonmax = np.max(np.abs(epsilon.data[:]))
deltamax = np.max(np.abs(delta.data[:]))
etamax = max(epsilonmax, deltamax)
vmax = model._max_vp
max_cos_sin = np.amax(np.abs(np.cos(theta.data[:]) - np.sin(theta.data[:])))
dvalue = min(spacing)

# %%
# Compute the dt and set time range
t0 = 0.   #  Simulation time start
tn = timesteps  #  Simulation time end (0.15 second = 150 msec)
dt = (dvalue/(np.pi*vmax))*np.sqrt(1/(1+etamax*(max_cos_sin)**2)) # eq. above (cell 3)
time_range = TimeAxis(start=t0, stop=tn, step=dt)
print("time_range; ", time_range)

# time stepping 
p = TimeFunction(name="p", grid=model.grid, time_order=2, space_order=8)
q = Function(name="q", grid=model.grid, space_order=8)

# Main equations

# Problem with the following line
term1_p = q.dx
# term2_p = (1 + 2*delta*(sintheta**2)*(costheta**2) + 2*epsilon*sintheta**4)*q.dy
# term3_p = (2-delta*(sin2theta)**2 + 3*epsilon*(sin2theta)**2 + 2*delta*(cos2theta)**2)*((q.dy2).dx2)

# term4_p = ( delta*sin4theta - 4*epsilon*sin2theta*costheta**2)*((q.dy).dx3)
# term5_p = (-delta*sin4theta - 4*epsilon*sin2theta*sintheta**2)*((q.dy3).dx)

stencil_p = solve(m*p.dt2 - term1_p, p.forward)
update_p = Eq(p.forward, stencil_p)

import pdb; pdb.set_trace()

p.data[:] = 0.1
optime = Operator([update_p], opt='xdsl')
optime(time_M=time_range.num-2, dt=dt)
print("Norm xdsl(p) is ", norm(p))

import pdb; pdb.set_trace()


p.data[:] = 0.1
optime = Operator([update_p])
optime(time_M=time_range.num-2, dt=dt)
print("Norm devito(p) is ", norm(p))

from devito import norm

print("Norm p is ", norm(p))

# import pdb; pdb.set_trace()