# # TTI pure qP-wave equation implementation

# First of all, it is necessary to import some Devito modules and other packages that will be used in the implementation.

import numpy as np
from devito import (Function, TimeFunction, cos, sin, solve,
                    Eq, Operator)
from examples.seismic import TimeAxis, RickerSource, Receiver, demo_model
from matplotlib import pyplot as plt
from devito import norm

# We will start with the definitions of the grid and the physical parameters $v_{p}, \theta, \epsilon, \delta$. For simplicity, we won't use any absorbing boundary conditions to avoid reflections of outgoing waves at the boundaries of the computational domain, but we will have boundary conditions (zero Dirichlet) at $x=0,nx$ and $z=0,nz$ for the solution of the Poisson equation. We use a homogeneous model. The model is discretized with a grid of $101 \times 101$ and spacing of 10 m. The $v_{p}, \epsilon, \delta$ and $\theta$ parameters of this model are 3600 m∕s, 0.23, 0.17, and 45°, respectively. 

# NBVAL_IGNORE_OUTPUT   

shape = (101, 101)  # 101x101 grid
spacing = (10., 10.)  # spacing of 10 meters
origin = (0., 0.)  
nbl = 1  # number of pad points

model = demo_model('layers-tti', spacing=spacing, space_order=8,
                   shape=shape, nbl=nbl, nlayers=1)

# initialize Thomsem parameters to those used in Mu et al., (2020)
model.update('vp', np.ones(shape)*3.6)  # km/s
model.update('epsilon', np.ones(shape)*0.23)
model.update('delta', np.ones(shape)*0.17)
model.update('theta', np.ones(shape)*(45.*(np.pi/180.)))  # radians

# In cell below, symbols used in the PDE definition are obtained from the `model` object.
# Note that trigonometric functions proper of Devito are exploited.

# Get symbols from model
theta = model.theta
delta = model.delta
epsilon = model.epsilon
m = model.m

# Use trigonometric functions from Devito
costheta = cos(theta)
sintheta = sin(theta)
cos2theta = cos(2*theta)
sin2theta = sin(2*theta)
sin4theta = sin(4*theta)

# Accordingly to [Mu et al., (2020)](https://library.seg.org/doi/10.1190/geo2019-0320.1), the time sampling can be chosen as 
# $$
# \Delta t < \frac{\Delta d}{\pi \cdot (v_{p})_{max}}\sqrt{\dfrac{1}{(1+\eta_{max}|\cos\theta-\sin\theta|_{max}^{2})}}
# $$,
# 
# where $\eta_{max}$ denotes the maximum value between $|\epsilon|_{max}$ and $|\delta|_{max}$, $|cos\theta-sin\theta|_{max}$ is the maximum value of $|cos\theta-sin\theta|$.


# Values used to compute the time sampling
epsilonmax = np.max(np.abs(epsilon.data[:]))
deltamax = np.max(np.abs(delta.data[:]))
etamax = max(epsilonmax, deltamax)
vmax = model._max_vp
max_cos_sin = np.amax(np.abs(np.cos(theta.data[:]) - np.sin(theta.data[:])))
dvalue = min(spacing)

# The next step is to define the simulation time. It has to be small enough to avoid reflections from borders. Note we will use the `dt` computed below rather than the one provided by the property() function `critical_dt` in the `SeismicModel` class, as the latter only works for the coupled pseudoacoustic equation.

# Compute the dt and set time range
t0 = 0.  # Simulation time start
tn = 150.  # Simulation time end (0.15 second = 150 msec)
dt = (dvalue/(np.pi*vmax))*np.sqrt(1/(1+etamax*(max_cos_sin)**2))  # eq. above (cell 3)

import pdb;pdb.set_trace()

time_range = TimeAxis(start=t0, stop=tn, step=dt)
print("time_range; ", time_range)

# In exactly the same form as in the [Cavity flow with Navier-Stokes]() tutorial, we will use two operators,
# one for solving the Poisson equation in pseudotime and one for advancing in time. But unlike what was done in such tutorial, in this case, we write the FD solution of the poisson equation in a manually way, without using the `laplace` shortcut and `solve` functionality (just to break up the routine and try to vary). The internal time loop can be controlled by supplying the number of pseudotime steps (`niter_poisson` iterations) as a `time` argument to the operator. A Ricker wavelet source with peak frequency of 20 Hz is located at center of the model.

# NBVAL_IGNORE_OUTPUT

# time stepping 
p = TimeFunction(name="p", grid=model.grid, time_order=2, space_order=8)
q = Function(name="q", grid=model.grid, space_order=8)

# Main equations
term1_p = (1 + 2*delta*(sintheta**2)*(costheta**2) + 2*epsilon*costheta**4)*q.dx4
term2_p = (1 + 2*delta*(sintheta**2)*(costheta**2) + 2*epsilon*sintheta**4)*q.dy4
term3_p = (2-delta*(sin2theta)**2 + 3*epsilon*(sin2theta)**2 + 2*delta*(cos2theta)**2)*((q.dy2).dx2)
term4_p = ( delta*sin4theta - 4*epsilon*sin2theta*costheta**2)*((q.dy).dx3)
term5_p = (-delta*sin4theta - 4*epsilon*sin2theta*sintheta**2)*((q.dy3).dx)

stencil_p = solve(m*p.dt2 - (term1_p + term2_p + term3_p + term4_p + term5_p), p.forward)
update_p = Eq(p.forward, stencil_p)

# Poisson eq. (following notebook 6 from CFD examples)
b = Function(name='b', grid=model.grid, space_order=8)
pp = TimeFunction(name='pp', grid=model.grid, space_order=8)

# Create stencil and boundary condition expressions
x, z = model.grid.dimensions
t = model.grid.stepping_dim

update_q = Eq(pp[t+1, x, z], ((pp[t, x+1, z] + pp[t, x-1, z])*z.spacing**2 + (pp[t, x, z + 1] + pp[t, x, z-1])*x.spacing**2 - 
               b[x, z]*x.spacing**2*z.spacing**2) / (2*(x.spacing**2 + z.spacing**2)))


# set source and receivers
src = RickerSource(name='src', grid=model.grid, f0=0.02, npoint=1, time_range=time_range)

src.coordinates.data[:, 0] = model.domain_size[0] * .5
src.coordinates.data[:, 1] = model.domain_size[0] * .5
# Define the source injection
src_term = src.inject(field=p.forward, expr=src)

# optime = Operator([update_p], opt='xdsl')
optime = Operator([update_p] + src_term, opt='xdsl')
# optime = Operator([update_p] + src_term)
optime.apply(time=time_range.num-1, dt=dt)
print(norm(p))

import pdb; pdb.set_trace()
