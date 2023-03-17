import numpy as np
from devito import (Grid, TimeFunction, Eq, Operator, XDSLOperator, norm, configuration)

from devito.ir import ietxdsl # noqa

from xdsl.printer import Printer # noqa
from xdsl.pattern_rewriter import PatternRewriteWalker, GreedyRewritePatternApplier # noqa

configuration['log-level'] = 'DEBUG'

grid = Grid(shape=(300, 300, 30))
steps = 1

u = TimeFunction(name='u', grid=grid)
u.data[:, :, :] = 5
print("Original data norm is {}".format(norm(u)))

print(norm(u))

eq = Eq(u.forward, u + 0.1)
xop = XDSLOperator([eq])
xop.apply(time_M=steps)
xdsl_data: np.array = u.data_with_halo.copy()
xdsl_norm = norm(u)

u.data[:,:] = 5
op = Operator([eq])
op.apply(time_M=steps)
orig_data: np.array = u.data_with_halo.copy() 
orig_norm = norm(u)


print("orig={}, \nxdsl={}".format(xdsl_norm, orig_norm))
assert np.isclose(xdsl_data, orig_data, rtol=1e-06).all()


# module = ietxdsl.transform_devito_to_iet_ssa(op)

# p = Printer(target=Printer.Target.MLIR)
# p.print(module)

# print("\n\nAFTER REWRITE:\n")

# ietxdsl.iet_to_standard_mlir(module)

# p = Printer(target=Printer.Target.MLIR)
# p.print(module)