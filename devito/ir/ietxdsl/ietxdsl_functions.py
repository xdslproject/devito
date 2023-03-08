# definitions pulled out from GenerateXDSL jupyter notebook
from sympy import Indexed, Integer, Symbol, Add, Eq, Mod, Pow, Mul, Float
import cgen

import devito.ir.iet.nodes as nodes
from devito.ir.iet import MetaCall
import devito.mpi.routines as routines  # noqa

from devito import ModuloDimension, SpaceDimension
from devito.passes.iet.languages.openmp import OmpRegion

from devito.ir.ietxdsl import (MLContext, IET, Constant, Modi, Idx,
                               Assign, Block, Iteration, IterationWithSubIndices,
                               Statement, PointerCast, Powi, Initialise,
                               StructDecl, Call)
from devito.tools.utils import as_tuple
from devito.types.basic import IndexedData

from xdsl.irdl import AnyOf
from xdsl.ir import SSAValue
from xdsl.dialects.builtin import (ContainerOf, Float16Type, Float32Type,
                                   Float64Type, Builtin, i32, f32, StringAttr)

from xdsl.dialects.arith import Muli, Addi
from xdsl.dialects.scf import For

floatingPointLike = ContainerOf(AnyOf([Float16Type, Float32Type, Float64Type]))

ctx = MLContext()
Builtin(ctx)
iet = IET(ctx)


def printHeaders(cgen, header_str, headers):
    for header in headers:
        cgen.printOperation(Statement.get(createStatement(header_str, header)))
    cgen.printOperation(Statement.get(createStatement('')))


def printIncludes(cgen, header_str, headers):
    for header in headers:
        cgen.printOperation(Statement.get(
                            createStatement(header_str, '"' + header + '"')))
    cgen.printOperation(Statement.get(createStatement('')))


def printStructs(cgen, struct_decs):
    for struct in struct_decs:
        cgen.printOperation(
            StructDecl.get(struct.tpname, struct.fields, struct.declname,
                           struct.pad_bytes))


def createStatement(string="", val=None):
    for t in as_tuple(val):
        string = string + " " + t

    return string


def collectStructs(parameters):
    struct_decs = []
    struct_strs = []
    for i in parameters:
        # Bypass a struct decl if it has te same _C_typename
        if (i._C_typedecl is not None and str(i._C_typename) not in struct_strs):
            struct_decs.append(i._C_typedecl)
            struct_strs.append(i._C_typename)
    return struct_decs


def calculateAddArguments(arguments):
    # Get an input of arguments that are added. In case only one argument remains,
    # return the argument.
    # In case more, return expression by breaking down args.
    if len(arguments) == 1:
        return arguments[0]
    else:
        return Add(arguments[0], calculateAddArguments(arguments[1:len(arguments)]))


def add_to_block(expr, arg_by_expr, result):
    if expr in arg_by_expr:
        return

    if isinstance(expr, IndexedData):
        # Only index first bit of IndexedData
        add_to_block(expr.args[0], arg_by_expr, result)
        arg_by_expr[expr] = arg_by_expr[expr.args[0]]
        return

    if isinstance(expr, Symbol):
        # All symbols must be passed in at the start

        my_expr = Symbol(expr.name)
        assert my_expr in arg_by_expr, f'Symbol with name {expr.name} not found ' \
                                       f'in {arg_by_expr}'
        arg_by_expr[expr] = arg_by_expr[my_expr]
        return

    if isinstance(expr, Integer):
        constant = int(expr.evalf())
        arg = Constant.from_int_and_width(constant, i32)
        arg_by_expr[expr] = arg
        result.append(arg)
        return

    if isinstance(expr, Float):
        constant = float(expr.evalf())
        arg = Constant.from_float_and_width(constant, f32)
        arg_by_expr[expr] = arg
        result.append(arg)
        return

    for child_expr in expr.args:
        add_to_block(child_expr, arg_by_expr, result)

    if isinstance(expr, Add):
        # workaround for large additions (TODO: should this be handled earlier?)
        len_args = len(expr.args)
        if len_args > 2:
            # this works for 3 arguments:
            first_arg = expr.args[0]
            second_arg = calculateAddArguments(expr.args[1:len_args])
            add_to_block(second_arg, arg_by_expr, result)
        else:
            first_arg = expr.args[0]
            second_arg = expr.args[1]
            # Mostly additions in indexing
            if isinstance(second_arg, SpaceDimension) | isinstance(second_arg, Indexed):
                tmp = first_arg
                first_arg = second_arg
                second_arg = tmp
        lhs = arg_by_expr[first_arg]
        rhs = arg_by_expr[second_arg]
        sum = Addi.get(lhs, rhs)
        arg_by_expr[expr] = sum
        result.append(sum)
        return

    if isinstance(expr, Mul):
        # Convert a sympy.core.mul.Mul to xdsl.dialects.arith.Muli
        lhs = arg_by_expr[expr.args[0]]
        rhs = arg_by_expr[expr.args[1]]
        sum = Muli.get(lhs, rhs)
        arg_by_expr[expr] = sum
        result.append(sum)
        return

    if isinstance(expr, nodes.Return):
        # Covert a Return node
        # import pdb;pdb.set_trace()
        return

    if isinstance(expr, Mod):
        # To update docstring
        lhs = arg_by_expr[expr.args[0]]
        rhs = arg_by_expr[expr.args[1]]
        sum = Modi.get(lhs, rhs)
        arg_by_expr[expr] = sum
        result.append(sum)
        return

    if isinstance(expr, Pow):
        # Convert sympy.core.power.Pow to devito.ir.ietxdsl.operations.Powi
        base = arg_by_expr[expr.args[0]]
        exponent = arg_by_expr[expr.args[1]]
        pow = Powi.get(base, exponent)
        arg_by_expr[expr] = pow
        result.append(pow)
        return

    if isinstance(expr, Indexed):
        add_to_block(expr.args[0], arg_by_expr, result)
        prev = arg_by_expr[expr.args[0]]
        for child_expr in expr.args[1:]:
            add_to_block(child_expr, arg_by_expr, result)
            child_arg = arg_by_expr[child_expr]
            idx = Idx.get(prev, child_arg)
            result.append(idx)
            prev = idx
        arg_by_expr[expr] = prev
        return

    if isinstance(expr, Eq):
        # Convert devito.ir.equations.equation.ClusterizedEq to devito.ir.ietxdsl.operations.Assign
        add_to_block(expr.args[0], arg_by_expr, result)
        lhs = arg_by_expr[expr.args[0]]
        add_to_block(expr.args[1], arg_by_expr, result)
        rhs = arg_by_expr[expr.args[1]]
        assign = Assign.build([lhs, rhs])
        arg_by_expr[expr] = assign
        result.append(assign)
        return

    assert False, f'unsupported expr {expr} of type {expr.func}'


def myVisit(node, block=None, ctx={}):
    try:
        print("asserted!")
        bool_node = isinstance(
            node, nodes.Node), f'Argument must be subclass of Node, found: {node}'
        comment_node = isinstance(
            node, cgen.Comment), f'Argument must be subclass of Node, found: {node}'
        statement_node = isinstance(
            node, cgen.Statement), f'Argument must be subclass of Node, found: {node}'
        assert bool_node or comment_node or statement_node
    except:
        print("fail!")

    if hasattr(node, 'is_Callable') and node.is_Callable:
        return

    if isinstance(node, nodes.CallableBody):
        return

    if isinstance(node, nodes.Expression):
        b = Block.from_arg_types([i32])
        r = []
        expr = node.expr
        if node.init:
            expr_name = expr.args[0]
            add_to_block(expr.args[1], {Symbol(s): a for s, a in ctx.items()}, r)
            init = Initialise.get(r[-1].results[0], r[-1].results[0], str(expr_name))
            block.add_ops([init])
        else:
            add_to_block(expr, {Symbol(s): a for s, a in ctx.items()}, r)
            block.add_ops(r)
        return

    if isinstance(node, nodes.ExpressionBundle):
        assert len(node.children) == 1
        for idx in range(len(node.children[0])):
            child = node.children[0][idx]
            myVisit(child, block, ctx)
        return

    if isinstance(node, nodes.Iteration):
        assert len(node.children) == 1
        assert len(node.children[0]) == 1
        index = node.index
        b = Block.from_arg_types([i32])
        ctx = {**ctx, index: b.args[0]}
        # check if there are subindices
        hasSubIndices = False
        if len(node.uindices) > 0:
            uindices_names = []
            uindices_symbmins = []
            for uindex in list(node.uindices):
                # currently only want to deal with a very specific subindex!
                if isinstance(uindex, ModuloDimension):
                    hasSubIndices = True
                    uindices_names.append(uindex.name)
                    uindices_symbmins.append(uindex.symbolic_min)
            if hasSubIndices:
                myVisit(node.children[0][0], b, ctx)
                if len(node.pragmas) > 0:
                    for p in node.pragmas:
                        prag = Statement.get(p)
                        block.add_ops([prag])
                iteration = IterationWithSubIndices.get(
                    node.properties, node.limits, uindices_names,
                    uindices_symbmins, node.index, b)
                block.add_ops([iteration])
                return
        myVisit(node.children[0][0], b, ctx)
        if len(node.pragmas) > 0:
            for p in node.pragmas:
                prag = Statement.get(p)
                block.add_ops([prag])
        # import pdb;pdb.set_trace()
        # lb = SSAValue.get(StringAttr(str(node.limits[0])))
        # ub = SSAValue.get(node.limits[1])
        # step = SSAValue.get(node.limits[2])
        # iteration = For.get(lb, ub, step, [], b)
        iteration = Iteration.get(node.properties, node.limits, node.index, b)
        block.add_ops([iteration])
        return

    if isinstance(node, nodes.Section):
        assert len(node.children) == 1
        assert len(node.children[0]) == 1
        for content in node.ccode.contents:
            if isinstance(content, cgen.Comment):
                comment = Statement.get(content)
                block.add_ops([comment])
            else:
                myVisit(node.children[0][0], block, ctx)
        return

    if isinstance(node, nodes.HaloSpot):
        assert len(node.children) == 1
        try:
            assert isinstance(node.children[0], nodes.Iteration)
        except:
            assert isinstance(node.children[0], OmpRegion)

        myVisit(node.children[0], block, ctx)
        return

    if isinstance(node, nodes.TimedList):
        assert len(node.children) == 1
        assert len(node.children[0]) == 1
        header = Statement.get(node.header[0])
        block.add_ops([header])
        myVisit(node.children[0][0], block, ctx)
        footer = Statement.get(node.footer[0])
        block.add_ops([footer])
        return

    if isinstance(node, nodes.PointerCast):
        statement = node.ccode
        pointer_cast = PointerCast.get(statement)
        block.add_ops([pointer_cast])
        return

    if isinstance(node, nodes.List):
        # Problem: When a List is ecountered with only body, but no header or footer
        # we have a problem
        for h in node.header:
            myVisit(h, block, ctx)

        for b in node.body:
            myVisit(b, block, ctx)

        for f in node.footer:
            myVisit(f, block, ctx)

        return

    if isinstance(node, nodes.Call):
        # Those parameters without associated types aren't printed in the Kernel header
        call_name = str(node.name)
        try:
            call_args = [str(i._C_name) for i in node.arguments]
        except:
            return

        call = Call.get(call_name, call_args, 'void')

        block.add_ops([call])

        print("Constructing a call")

        return

    if isinstance(node, nodes.Conditional):
        # Those parameters without associated types aren't printed in the Kernel header
        print("Conditional placement skipping")
        return

    if isinstance(node, cgen.Comment):
        # cgen.Comment as Statement
        comment = Statement.get(node)
        block.add_ops([comment])
        return

    if isinstance(node, cgen.Statement):
        comment = Statement.get(node)
        block.add_ops([comment])
        return

    if isinstance(node, cgen.Line):
        comment = Statement.get(node)
        block.add_ops([comment])
        return

    raise TypeError(f'Unsupported type of node: {type(node)}, {vars(node)}')
