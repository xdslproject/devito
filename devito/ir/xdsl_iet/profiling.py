
from dataclasses import dataclass, field

from devito.ir.xdsl_iet import iet_ssa
from xdsl.dialects import builtin, func, llvm

from xdsl.pattern_rewriter import (RewritePattern, op_type_rewrite_pattern,
                                   GreedyRewritePatternApplier, PatternRewriter,
                                   PatternRewriteWalker, InsertPoint)


class TimerRewritePattern(RewritePattern):
    """
    Base class for time benchmarking related rewrite patterns
    """
    pass


@dataclass
class MakeFunctionTimed(TimerRewritePattern):
    """
    Populate the section0 devito timer with the total runtime of the function
    """
    func_name: str
    seen_ops: set[func.Func] = field(default_factory=set)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if op.sym_name.data != self.func_name or op in self.seen_ops:
            return

        # Add the op to the set of seen operations
        self.seen_ops.add(op)

        # Insert timer start and end calls
        rewriter.insert_op([
            t0 := func.Call('timer_start', [], [builtin.f64])
        ], InsertPoint.at_start(op.body.block))

        ret = op.get_return_op()
        assert ret is not None

        rewriter.insert_op([
            timers := iet_ssa.LoadSymbolic.get('timers', llvm.LLVMPointerType.opaque()),
            t1 := func.Call('timer_end', [t0], [builtin.f64]),
            llvm.StoreOp(t1, timers),
        ], InsertPoint.before(ret))

        rewriter.insert_op([
            func.FuncOp.external('timer_start', [], [builtin.f64]),
            func.FuncOp.external('timer_end', [builtin.f64], [builtin.f64]),
        ], InsertPoint.after(rewriter.current_operation))


def apply_timers(module, **kwargs):
    """
    Apply timers to a module
    """

    name = kwargs.get("name", "Kernel")
    grpa = GreedyRewritePatternApplier([MakeFunctionTimed(name)])
    PatternRewriteWalker(grpa, walk_regions_first=True).rewrite_module(module)
