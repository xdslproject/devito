import numpy as np
from devito import ConditionalDimension, Function, Eq, Operator, Dimension, dimensions
from devito.types import Lt

"""
Tests that deeply nested (depth > 2) functions used as indices are indexified.
"""
x, y1, y2, y3, y4 = dimensions('x y1 y2 y3 y4')

u1 = Function(name="u1", shape=(4, 4), dimensions=(x, y1), dtype=np.int32)
u2 = Function(name="u2", shape=(4, 4), dimensions=(x, y2), dtype=np.int32)
u3 = Function(name="u3", shape=(4, 4), dimensions=(x, y3), dtype=np.int32)
u4 = Function(name="u4", shape=(4, 4), dimensions=(x, y4), dtype=np.int32)

Eq1 = Eq(u1, u2[x, u3[x, u4]])
Eq2 = Eq(u1, u2[x, u3[x, u4[x, u4[x, u4[x, y4]]]]])

op1 = Operator([Eq1])
op2 = Operator([Eq2])

op1.apply()
op2.apply()

print(op1.ccode)
print(op2.ccode)

assert str(op1.ccode) == str(op2.ccode)
