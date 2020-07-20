import numpy as np
from devito import Grid, ConditionalDimension, Function, Eq, Operator, Dimension, dimensions
from devito.types import Lt

"""
Tests that deeply nested (depth > 2) functions are lowered and indexified.
"""

grid = Grid(shape=(4, 4))
x, y = grid.dimensions

u1 = Function(name="u1", grid=grid, dtype=np.int32)
u2 = Function(name="u2", grid=grid, dtype=np.int32)
u3 = Function(name="u3", grid=grid, dtype=np.int32)
u4 = Function(name="u4", grid=grid, dtype=np.int32)
u5 = Function(name="u5", grid=grid, dtype=np.int32)

Eq1 = Eq(u1, u2[x, u3[x, u4[x, u5[x, u1]]]])
Eq2 = Eq(u1, u2[x, u3[x, u4[x, u5[x, u1[x + 1, y + 1]]]]])

op1 = Operator([Eq1])
op2 = Operator([Eq2])

op1.apply()
op2.apply()

print(op1.ccode)
print(op2.ccode)

assert ('u1[x + 1][y + 1] = u2[x + 1][u3[x][u4[x][u5[x]'
        '[u1[x + 1][y + 1]]]] + 1]') in str(op2.ccode)

assert str(op2.ccode) == str(op1.ccode)
