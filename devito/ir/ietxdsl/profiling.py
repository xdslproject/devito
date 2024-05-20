
from dataclasses import dataclass, field

from devito.ir.ietxdsl import iet_ssa
from xdsl.dialects import builtin, func, llvm

from xdsl.pattern_rewriter import (RewritePattern, op_type_rewrite_pattern,
                                   GreedyRewritePatternApplier, PatternRewriter,
                                   PatternRewriteWalker)


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

        # only apply once
        self.seen_ops.add(op)

        # Insert timer start and end calls
        rewriter.insert_op_at_start([
            t0 := func.Call('timer_start', [], [builtin.f64])
        ], op.body.block)

        ret = op.get_return_op()
        assert ret is not None

        rewriter.insert_op_before([
            timers := iet_ssa.LoadSymbolic.get('timers', llvm.LLVMPointerType.opaque()),
            t1 := func.Call('timer_end', [t0], [builtin.f64]),
            llvm.StoreOp(t1, timers),
        ], ret)

        rewriter.insert_op_after_matched_op([
            func.FuncOp.external('timer_start', [], [builtin.f64]),
            func.FuncOp.external('timer_end', [builtin.f64], [builtin.f64])
        ])


def apply_timers(module, **kwargs):
    """
    Apply timers to a module
    """
    name = kwargs.get("name", "Kernel")
    grpa = GreedyRewritePatternApplier([MakeFunctionTimed(name)])
    PatternRewriteWalker(grpa, walk_regions_first=True).rewrite_module(module)
