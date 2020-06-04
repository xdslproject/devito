import numpy as np
from devito import ConditionalDimension, Function, Eq, Operator, Dimension
from devito.types import Lt

shape = (8, 8)
x = Dimension(name='x')
y = Dimension(name='y')
g = Function(name='g', shape=shape, dimensions=(x, y), dtype=np.float32)
cond = Lt(g, 2)
ci = ConditionalDimension(name='ci', parent=y, condition=cond)
f = Function(name='f', shape=shape, dimensions=(x, ci))
eq1 = Eq(f, 5)
op = Operator([eq1])
print(op.ccode)
import pdb; pdb.set_trace()
op.apply()
