from devito.ir import ietxdsl
from devito import Grid, Function, TimeFunction, Eq, Operator, Constant

from xdsl.printer import Printer

from xdsl.pattern_rewriter import PatternRewriteWalker, GreedyRewritePatternApplier

if __name__ == '__main__':

    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    op = Operator([eq])
    module = ietxdsl.transform_devito_to_iet_ssa(op)

    p = Printer(target=Printer.Target.MLIR)
    p.print(module)

    print("\n\nAFTER REWRITE:\n")

    walk = PatternRewriteWalker(GreedyRewritePatternApplier([
        ietxdsl.LowerIetForToScfParallel(),
        ietxdsl.LowerIetForToScfFor()
    ]))
    walk.rewrite_module(module)

    p = Printer(target=Printer.Target.MLIR)
    p.print(module)
