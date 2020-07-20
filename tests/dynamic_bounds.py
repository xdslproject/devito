import numpy as np
from devito import Function, Eq, Operator, Dimension, Inc, TimeFunction, TimeDimension, Grid

x = Dimension(name='x')
y = Dimension(name='y')
time = TimeDimension(name='time')

g = TimeFunction(name='g', shape=(1,3), dimensions=(time, x), time_order=0, dtype=np.int32)
g.data[0, :] = [1, 2, 3]

h1 = TimeFunction(name='h1', shape=(1,3), dimensions=(time, y), time_order=0, dtype=np.float32)
h1.data[0, :] = [0]

eq0 = Eq(y.symbolic_max, g[0, x], implicit_dims=(time, x)) # TIME-INVARIANT

eq1 = Inc(h1[0,0], 1, implicit_dims= (time, x, y))

op = Operator([eq0, eq1])

print(op.ccode)
op.apply()
# assert(h1.data==9.)