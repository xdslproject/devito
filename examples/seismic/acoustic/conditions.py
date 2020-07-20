from devito import Dimension, ConditionalDimension, Function, Eq, Operator
from sympy import And
size = 16
i = Dimension(name='i')
k = Dimension(name='k')
g = Function(name='g', shape=(int(size),int(size)), dimensions=(i,k))
ci = ConditionalDimension(name='ci', parent=k, condition=And(g[i,k] > 0, g[i,k] < 4, evaluate=False))
f = Function(name='f', shape=(int(size),int(size)), dimensions=(i,ci))
op = Operator(Eq(g, f + 1)).apply()
import pdb;pdb.set_trace()