from devito.ir.ietxdsl import Iteration
from xdsl.pattern_rewriter import RewritePattern, GreedyRewritePatternApplier, \
    op_type_rewrite_pattern, PatternRewriteWalker, PatternRewriter
from xdsl.dialects.builtin import ModuleOp, StringAttr
from dataclasses import dataclass


# NOTE: this is WIP and needs refactoring ;)
@dataclass
class MakeSimdPattern(RewritePattern):
    """
    This pattern reproduces the behaviour of PragmaSimdTransformer
    """

    def is_parallel_relaxed(self, iteration: Iteration) -> bool:
        return any([
            prop.data
            in ["parallel", "parallel_if_private", "parallel_if_private"]
            for prop in iteration.properties.data
        ])

    @op_type_rewrite_pattern
    def match_and_rewrite(self, iteration: Iteration,
                          rewriter: PatternRewriter):

        if (not self.is_parallel_relaxed(iteration)):
            return

        # check if parent is parallel as well
        parent_op = iteration.parent.parent.parent
        if (not self.is_parallel_relaxed(parent_op)):
            return

        children_ops = iteration.body.blocks[0].ops

        # TODO this is a bit expensive, is there a way to only match on the lowest
        #   level operations
        # check if children is parallel as well
        if any([
            isinstance(op, Iteration) and self.is_parallel_relaxed(op)
            for op in children_ops
        ]):
            return

        # TODO: insert additional checks
        iteration.pragmas.data.append(StringAttr.from_str("simd-for"))


def construct_walker() -> PatternRewriteWalker:
    applier = GreedyRewritePatternApplier([MakeSimdPattern()])

    return PatternRewriteWalker(applier,
                                walk_regions_first=False,
                                apply_recursively=False)


def make_simd(ctx, op: ModuleOp):
    walker = construct_walker()
    walker.rewrite_module(op)
