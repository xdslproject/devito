import numpy as np
from devito import TimeFunction, Function, configuration, Dimension, Eq, ConditionalDimension
from devito import Operator
from examples.seismic import RickerSource, TimeAxis
from examples.seismic import Model
from devito.symbolics import CondEq
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size

x = Dimension(name='x')
y1 = Dimension(name='y1')
y2 = Dimension(name='y2')
u1 = Function(name="u1", shape=(4, 4), dimensions=(x, y1), dtype=np.int32)
u2 = Function(name="u2", shape=(4, 4), dimensions=(x, y2), dtype=np.int32)

Eq1 = Eq(u2, u2[x, u1]) #  Not working
Eq2 = Eq(u2, u2[x, u1[x, y1]]) #  Working

op1 = Operator([Eq1])
op2 = Operator([Eq2])
import pdb; pdb.set_trace()

assert str(op1.ccode) == str(op2.ccode)
