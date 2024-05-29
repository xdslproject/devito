# ------------- General imports -------------#

from typing import Any, Iterable
from dataclasses import dataclass, field
from sympy import Add, And, Expr, Float, GreaterThan, Indexed, Integer, LessThan, Mod, Mul, Number, Pow, StrictGreaterThan, StrictLessThan, Symbol, floor
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction
from devito.ir.equations.equation import OpInc
from devito.operations.interpolators import Injection
from devito.operator.operator import Operator
from devito.symbolics.search import retrieve_dimensions, retrieve_functions
from devito.symbolics.extended_sympy import INT
from devito.tools.data_structures import OrderedSet
from devito.tools.utils import as_tuple
from devito.types.basic import Scalar
from devito.types.dense import DiscreteFunction, Function, TimeFunction
from devito.types.dimension import SpaceDimension, TimeDimension
from devito.types.equation import Eq

# ------------- xdsl imports -------------#
from xdsl.dialects import (arith, builtin, func, memref, scf,
                           stencil, gpu)
from xdsl.dialects.experimental import math
from xdsl.ir import Block, Operation, OpResult, Region, SSAValue
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.builder import ImplicitBuilder
from xdsl.transforms.experimental.convert_stencil_to_ll_mlir import StencilToMemRefType

# ------------- devito imports -------------#
from devito import Grid, SteppingDimension
from devito.ir.equations import LoweredEq
from devito.symbolics import retrieve_function_carriers
from devito.tools.data_structures import OrderedSet
from devito.types.dense import DiscreteFunction, Function, TimeFunction
from devito.types.equation import Eq
from devito.types.mlir_types import dtype_to_xdsltype

# ------------- devito-xdsl SSA imports -------------#
from devito.ir.ietxdsl import iet_ssa
from devito.ir.ietxdsl.utils import is_int, is_float

# flake8: noqa


def field_from_function(f: DiscreteFunction) -> stencil.FieldType:
    # import pdb;pdb.set_trace()
    halo = [f.halo[d] for d in f.dimensions]
    shape = f.shape
    bounds = [(-h[0], s+h[1]) for h, s in zip(halo, shape)]
    if isinstance(f, TimeFunction):
        bounds = bounds[1:]
    return stencil.FieldType(bounds, element_type=dtypes_to_xdsltypes[f.dtype])


class ExtractDevitoStencilConversion:
    """
    Lower Devito equations to the stencil dialect
    """

    operator: type[Operator]
    eqs: list[LoweredEq]
    block: Block
    temps: dict[tuple[DiscreteFunction, int], SSAValue]
    symbol_values: dict[str, SSAValue]
    time_offs: int

    def __init__(self):
        self.temps = dict()
        self.symbol_values = dict()

    time_offs: int

    def __init__(self, operator: type[Operator]):
        self.temps = dict()
        self.operator = operator

    def convert_function_eq(self, eq: LoweredEq, **kwargs):
        # Read the grid containing necessary discretization information
        # (size, halo width, ...)
        write_function = eq.lhs.function
        grid: Grid = write_function.grid
        step_dim = grid.stepping_dim

        if isinstance(write_function, TimeFunction):
            time_size = write_function.time_size
            output_time_offset = (eq.lhs.indices[step_dim] - step_dim) % time_size
            self.out_time_buffer = (write_function, output_time_offset)
        elif isinstance(write_function, Function):
            self.out_time_buffer = (write_function, 0)
        else:
            raise NotImplementedError(f"Function of type {type(write_function)} not supported")

        dims = retrieve_dimensions(eq.lhs.indices)
        if not all(isinstance(d, (SteppingDimension, SpaceDimension)) for d in dims):
            self.build_generic_step(step_dim, eq)
        else:# Get the function carriers of the equation
            self.build_stencil_step(step_dim, eq)

    def convert_symbol_eq(self, symbol: Symbol, rhs: LoweredEq, **kwargs):
        """
        Convert a symbol equation to xDSL.
        """
        self.symbol_values[symbol.name] = self._visit_math_nodes(None, rhs, None)
        self.symbol_values[symbol.name].name_hint = symbol.name

    def _convert_eq(self, eq: LoweredEq, **kwargs):
        """
        # Docs here Need rewriting

        This converts a Devito LoweredEq to IR implementing it.
        e.g.
        ```python
        Eq(u[t + 1, x + 1], u[t, x + 1] + 1)
        ```
        with a grid of
        ```python
        Grid[extent=(1.0,), shape=(3,), dimensions=(x,)]
        ```

        1. Create a stencil.apply op to implement the equation, with a classical AST
        translation.

        The example above would be translated to:
        ```mlir

        %u_t0_temp = "stencil.load"(%u_t0) : (!stencil.field<[-1,4]xf32>) -> !stencil.temp<?xf32>
        %4 = "stencil.apply"(%u_t0_temp) ({
        ^0(%u_t0_blk : !stencil.temp<?xf32>):
        %5 = arith.constant 1 : i64
        %6 = "stencil.access"(%u_t0_blk) {"offset" = #stencil.index<0>} : (!stencil.temp<?xf32>) -> f32
        %7 = "arith.sitofp"(%5) : (i64) -> f32
        %8 = arith.addf %7, %6 : f32
        "stencil.return"(%8) : (f32) -> ()
        }) : (!stencil.temp<?xf32>) -> !stencil.temp<?xf32>
        "stencil.store"(%4, %u_t1) {"lb" = #stencil.index<0>, "ub" = #stencil.index<3>} : (!stencil.temp<?xf32>, !stencil.field<[-1,4]xf32>) -> ()
        ```
        """
        # Get the left hand side, called "output function" here because it tells us
        # Where to write the results of each step.
        write_function = eq.lhs

        match write_function:
            case Indexed():
                match write_function.function:
                    case TimeFunction() | Function():
                        self.convert_function_eq(eq, **kwargs)
                    case _:
                        raise NotImplementedError(
                            f"Function of type {type(write_function.function)} not supported"  # noqa
                        )
            case Symbol():
                self.convert_symbol_eq(write_function, eq.rhs, **kwargs)
            case _:
                raise NotImplementedError(f"LHS of type {type(write_function)} not supported")  # noqa

    def _visit_math_nodes(self, dim: SteppingDimension, node: Expr,
                          output_indexed: Indexed) -> SSAValue:
        # Handle Indexeds
        if isinstance(node, Indexed):
            # If we have a time function, we compute its time offset
            if isinstance(node.function, TimeFunction):
                time_offset = (node.indices[dim] - dim) % node.function.time_size
            elif isinstance(node.function, (Function, PointSource)):
                time_offset = 0
            else:
                raise NotImplementedError(f"reading function of type {type(node.func)} not supported")
            # If we are in a stencil (encoded by having the output_indexed passed), we
            # compute the relative space offsets and make it a stencil offset
            if output_indexed is not None:
                space_offsets = [node.indices[d] - output_indexed.indices[d] for d in node.function.space_dimensions]
                temp = self.function_values[(node.function, time_offset)]
                access = stencil.AccessOp.get(temp, space_offsets)
                return access.res
            # Otherwise, generate a load op
            else:
                temp = self.function_values[(node.function, time_offset)]
                memtemp = builtin.UnrealizedConversionCastOp.get(temp, StencilToMemRefType(temp.type)).results[0]
                memtemp.name_hint = temp.name_hint + "_mem"
                indices = node.indices
                if isinstance(node.function, TimeFunction):
                    indices = indices[1:]
                ssa_indices = [self._visit_math_nodes(dim, i, output_indexed) for i in node.indices]
                for i, ssa_i in enumerate(ssa_indices):
                    if isinstance(ssa_i.type, builtin.IntegerType):
                        ssa_indices[i] = arith.IndexCastOp(ssa_i, builtin.IndexType())
                return memref.Load.get(memtemp, ssa_indices).res

            import pdb; pdb.set_trace()
        # Handle Integers
        elif isinstance(node, Integer):
            cst = arith.Constant.from_int_and_width(int(node), builtin.i64)
            return cst.result
        # Handle Floats
        elif isinstance(node, Float):
            cst = arith.Constant(builtin.FloatAttr(float(node), builtin.f32))
            return cst.result
        # Handle Symbols
        elif isinstance(node, Symbol):
            if node.name in self.symbol_values:
                return self.symbol_values[node.name]
            else: 
                mlir_dtype = dtypes_to_xdsltypes[node.dtype]
                symb = iet_ssa.LoadSymbolic.get(node.name, mlir_dtype)
                return symb.result     
        # Handle Add Mul
        elif isinstance(node, (Add, Mul)):
            args = [self._visit_math_nodes(dim, arg, output_indexed) for arg in node.args]
            # Add casts when necessary
            # get first element out, store the rest in args
            # this makes the reduction easier
            carry, *args = self._ensure_same_type(*args)
            # select the correct op from arith.addi, arith.addf, arith.muli, arith.mulf
            if is_int(carry):
                op_cls = arith.Addi if isinstance(node, Add) else arith.Muli
            elif isinstance(carry.type, builtin.Float32Type):
                op_cls = arith.Addf if isinstance(node, Add) else arith.Mulf
            else:
                raise NotImplementedError(f"Add support for another type {carry.type}")
            for arg in args:
                op = op_cls(carry, arg)
                carry = op.result
            return carry

        # Handle Pow
        elif isinstance(node, Pow):
            args = [self._visit_math_nodes(dim, arg, output_indexed) for arg in node.args]
            assert len(args) == 2, "can't pow with != 2 args!"
            base, ex = args
            if is_int(base):
                if is_int(ex):
                    op_cls = math.IpowIOP
                else:
                    raise ValueError("no IPowFOp yet!")
            elif is_float(base):
                if is_float(ex):
                    op_cls = math.PowFOp
                elif is_int(ex):
                    op_cls = math.FPowIOp
                else:
                    raise ValueError("Expected float or int as pow args!")
            else:
                raise ValueError("Expected float or int as pow args!")

            op = op_cls(base, ex)
            return op.result
        # Handle Mod
        elif isinstance(node, INT):
            assert len(node.args) == 1, "Expected single argument for integer cast."
            return arith.FPToSIOp(self._visit_math_nodes(dim, node.args[0], output_indexed), builtin.i64).result
        elif isinstance(node, floor):
            assert len(node.args) == 1, "Expected single argument for floor."
            return math.FloorOp(self._visit_math_nodes(dim, node.args[0], output_indexed)).result
        elif isinstance(node, And):
            SSAargs = (self._visit_math_nodes(dim, arg, output_indexed) for arg in node.args)
            return reduce(lambda x,y : arith.AndI(x,y).result, SSAargs)
        elif isinstance(node, Relational):
            if isinstance(node, GreaterThan):
                mnemonic = "sge"
            elif isinstance(node, LessThan):
                mnemonic = "sle"
            elif isinstance(node, StrictGreaterThan):
                mnemonic = "sgt"
            elif isinstance(node, StrictLessThan):
                mnemonic = "slt"
            else:
                raise NotImplementedError(f"Unimplemented comparison {type(node)}")
                
            # import pdb;
            # pdb.set_trace()
            SSAargs = (self._visit_math_nodes(dim, arg, output_indexed) for arg in node.args)
            # Operands must have the same type
            # TODO: look at here if index stuff does not make sense
            # The s in sgt means *signed* greater than
            # Writer has no clue if this should rather be u for unsigned
            return arith.Cmpi(*self._ensure_same_type(*SSAargs), mnemonic).result

        else:
            raise NotImplementedError(f"Unknown math:{type(node)} {node}", node)

    def build_stencil_step(self, dim: SteppingDimension, eq:LoweredEq) -> None:
        """
        Builds the body of the step function for a given dimension and equation.

        Args:
            dim (SteppingDimension): The stepping dimension for the equation.
            eq (LoweredEq): The equation to build the step body for.

        Returns:
            None
        """
        read_functions = set()
        for f in retrieve_function_carriers(eq.rhs):
            if isinstance(f.function, PointSource):
                time_offset = 0
            elif isinstance(f.function, TimeFunction):
                time_offset = (f.indices[dim]-dim) % f.function.time_size
            elif isinstance(f.function, Function):
                time_offset = 0
            else:
                import pdb;pdb.set_trace()
                raise NotImplementedError(f"reading function of type {type(f.function)} not supported")
            read_functions.add((f.function, time_offset))

        for f, t in read_functions:
            if (f, t) not in self.temps:
                self.temps[(f, t)] = stencil.LoadOp.get(self.function_values[(f, t)]).res
                self.temps[(f, t)].name_hint = f"{f.name}_t{t}_temp"

        apply_args = [self.temps[f] for f in read_functions]

        write_function = self.out_time_buffer[0]
        shape = write_function.grid.shape_local
        apply = stencil.ApplyOp.get(
            apply_args,
            Block(arg_types=[a.type for a in apply_args]),
            result_types=[stencil.TempType(len(shape),
                          element_type=dtype_to_xdsltype(write_function.dtype))]
        )

        # Adding temp as suffix to the apply result name
        apply.res[0].name_hint = f"{write_function.name}_t{self.out_time_buffer[1]}_temp"

        # Give names to stencil.apply's block arguments
        for apply_arg, apply_op in zip(apply.region.block.args, apply.operands):
            # Just reuse the corresponding operand name
            # i.e. %v_t1_temp -> %v_t1_blk
            assert "temp" in apply_op.name_hint
            apply_arg.name_hint = apply_op.name_hint.replace("temp", "blk")

        self.apply_temps = {k:v for k,v in zip(read_functions, apply.region.block.args)}
        self.function_values |= self.apply_temps

        with ImplicitBuilder(apply.region.block):
            stencil.ReturnOp.get([self._visit_math_nodes(dim, eq.rhs, eq.lhs)])

        lb = stencil.IndexAttr.get(*([0] * len(shape)))
        ub = stencil.IndexAttr.get(*shape)

        store = stencil.StoreOp.get(
            apply.res[0],
            self.function_values[self.out_time_buffer],
            stencil.StencilBoundsAttr(zip(lb, ub)),
            stencil.TempType(len(shape),
                             element_type=dtype_to_xdsltype(write_function.dtype))
        )

        store.temp_with_halo.name_hint = f"{write_function.name}_t{self.out_time_buffer[1]}_temp"  # noqa
        self.temps[self.out_time_buffer] = store.temp_with_halo

    def build_generic_step_expression(self, dim: SteppingDimension, eq: LoweredEq):
        value = self._visit_math_nodes(dim, eq.rhs, None)
        temp = self.function_values[self.out_time_buffer]
        memtemp = builtin.UnrealizedConversionCastOp.get([temp], [StencilToMemRefType(temp.type)]).results[0]
        memtemp.name_hint = temp.name_hint + "_mem"
        indices = eq.lhs.indices
        if isinstance(eq.lhs.function, TimeFunction):
            indices = indices[1:]
        ssa_indices = [self._visit_math_nodes(dim, i, None) for i in indices]
        for i, ssa_i in enumerate(ssa_indices):
            if isinstance(ssa_i.type, builtin.IntegerType):
                ssa_indices[i] = arith.IndexCastOp(ssa_i, builtin.IndexType())

        match eq.operation:
            case None:
                pass
            case OpInc:
                value = arith.Addf(value, memref.Load.get(memtemp, ssa_indices).res)
        memref.Store.get(value, memtemp, ssa_indices)

    def build_condition(self, dim: SteppingDimension, eq: BooleanFunction):
        return self._visit_math_nodes(dim, eq, None)

    def build_generic_step(self, dim: SteppingDimension, eq: LoweredEq):
        if eq.conditionals:
            condition = And(*eq.conditionals.values(), evaluate=False)
            cond = self.build_condition(dim, condition)
            if_ = scf.If(cond, (), Region(Block()))
            with ImplicitBuilder(if_.true_region.block):
                self.build_generic_step_expression(dim, eq)
                scf.Yield()
        else:
            self.build_generic_step_expression(dim, eq)

    def build_time_loop(
        self, eqs: list[Any], step_dim: SteppingDimension, **kwargs
    ):
        # Bounds and step boilerpalte
        lb = iet_ssa.LoadSymbolic.get(
            step_dim.symbolic_min._C_name, builtin.IndexType()
        )
        ub = iet_ssa.LoadSymbolic.get(
            step_dim.symbolic_max._C_name, builtin.IndexType()
        )
        one = arith.Constant.from_int_and_width(1, builtin.IndexType())
        # Devito iterates from time_m to time_M *inclusive*, MLIR only takes
        # exclusive upper bounds, so we increment here.
        ub = arith.Addi(ub, one)

        # Take the exact time_step from Devito
        try:
            step = arith.Constant.from_int_and_width(
                int(step_dim.symbolic_incr), builtin.IndexType()
            )

            step.result.name_hint = "step"
        except:
            raise ValueError("step must be int!")

        # The iteration arguments
        # Those are the buffers we will be swapping through the time loop.
        # This is our SSA implementation of DeVito's buffer swapping as
        # u[(t+k)%n].
        iter_args = list(
            v for (f, t), v in self.function_args.items() if isinstance(f, TimeFunction)
        )
        # Create the for loop
        loop = scf.For(
            lb,
            ub,
            step,
            iter_args,
            Block(arg_types=[builtin.IndexType(), *(a.type for a in iter_args)]),
        )

        # Name the 'time' step iterator
        loop.body.block.args[0].name_hint = step_dim.root.name  # 'time'

        # Store a mapping from time_buffers to their corresponding block
        # arguments for easier access later.
        self.block_args = {
            (f, t): loop.body.block.args[1 + i]
            for i, (f, t) in enumerate(self.time_buffers)
        }
        self.function_values |= self.block_args
        # Name the block argument for debugging.
        for (f, t), arg in self.block_args.items():
            arg.name_hint = f"{f.name}_t{t}"

        with ImplicitBuilder(loop.body.block):
            self.generate_equations(eqs, **kwargs)
            # Swap buffers through scf.yield
            yield_args = [
                self.block_args[(f, (t + 1) % f.time_size)]
                for (f, t) in self.block_args.keys()
            ]
            scf.Yield(*yield_args)

    def generate_equations(self, eqs: list[Any], **kwargs):
        # Lower equations to their xDSL equivalent
        for eq in eqs:
            if isinstance(eq, Eq):
                lowered = self.operator._lower_exprs(as_tuple(eq), **kwargs)[0]
                self._convert_eq(lowered, **kwargs)
            elif isinstance(eq, Injection):
                lowered = self.operator._lower_exprs(as_tuple(eq), **kwargs)
                self._lower_injection(lowered)
            else:
                raise NotImplementedError(f"Expression {eq} of type {type(eq)} not supported")

    def _lower_injection(self, eqs: list[LoweredEq]):
        """
        Lower an injection to xDSL.
        """
        # We assert that all equations of one Injection share the same iteration space!
        ispaces = [e.ispace for e in eqs]
        assert all(ispaces[0] == isp for isp in ispaces[1:])
        ispace = ispaces[0]
        assert isinstance(ispace.dimensions[0], TimeDimension)

        lbs = []
        ubs = []
        for interval in ispace[1:]:
            lower = interval.symbolic_min
            if isinstance(lower, Scalar):
                lb = iet_ssa.LoadSymbolic.get(lower._C_name, builtin.IndexType())
            elif isinstance(lower, (Number, int)):
                lb = arith.Constant.from_int_and_width(int(lower), builtin.IndexType())
            else:
                raise NotImplementedError(f"Lower bound of type {type(lower)} not supported")
            lb.result.name_hint = f"{interval.dim.name}_m"

            upper = interval.symbolic_max
            if isinstance(upper, Scalar):
                ub = iet_ssa.LoadSymbolic.get(upper._C_name, builtin.IndexType())
            elif isinstance(upper, (Number, int)):
                ub = arith.Constant.from_int_and_width(int(upper), builtin.IndexType())
            else:
                raise NotImplementedError(
                    f"Upper bound of type {type(upper)} not supported"
                )
            ub.result.name_hint = f"{interval.dim.name}_M"
            lbs.append(lb)
            ubs.append(ub)

        steps = [arith.Constant.from_int_and_width(1, builtin.IndexType()).result]*len(ubs)
        ubs = [arith.Addi(ub, steps[0]) for ub in ubs]

        with ImplicitBuilder(scf.ParallelOp(lbs, ubs, steps, [pblock := Block(arg_types=[builtin.IndexType()]*len(ubs))]).body):
            for arg, interval in zip(pblock.args, ispace[1:], strict=True):
                arg.name_hint = interval.dim.name
                self.symbol_values[interval.dim.name] = arg
            for eq in eqs:
                self._convert_eq(eq)
            scf.Yield()
            # raise NotImplementedError("Injections not supported yet")

    def convert(self, eqs: Iterable[Eq], **kwargs) -> builtin.ModuleOp:
        """
        This converts a Devito Operator, represented here by a list of LoweredEqs, to
        an xDSL module defining a function implementing it.
        ```python
        [eq0 := Eq(...), eq1 := Eq(...), ...]
        ```
        with a grid of
        ```python
        Grid[extent=(1.0,), shape=(3,), dimensions=(x,)]
        ```
        and one TimeFunction u with 2 time buffers, would be converted to:

        1. Create a function signature corresponding to all used functions and
        their time sizes. Their sizes are deduced from the Grid.
        2. Create a time iteration loop, swapping buffers to implement time buffering.

        NB: This needs to be converted to a Cluster conversion soon, which will be more sound.

        ```mlir
        func.func @apply_kernel(%u_vec_0 : !stencil.field<[-1,4]xf32>, %u_vec_1 : !stencil.field<[-1,4]xf32>) {
        %time_m = "devito.load_symbolic"() {"symbol_name" = "time_m"} : () -> index
        %time_M = "devito.load_symbolic"() {"symbol_name" = "time_M"} : () -> index
        %0 = arith.constant 1 : index
        %step = arith.constant 1 : index
        %1 = arith.addi %time_M, %0 : index
        %2, %3 = scf.for %time = %time_m to %1 step %step iter_args(%u_t0 = %u_vec_0, %u_t1 = %u_vec_1) -> (!stencil.field<[-1,4]xf32>, !stencil.field<[-1,4]xf32>) {
            # eq0
            # eq1
            # ...
            scf.yield %u_t1, %u_t0 : !stencil.field<[-1,4]xf32>, !stencil.field<[-1,4]xf32>
        }
        func.return
        }
        ```

        The code generated by this step still contains "devito.load_symbolic" ops.
        Those represents runtime values not yet known that will be JIT-compiled when
        calling the operator.
        """
        # Instantiate the module.
        self.function_values : dict[tuple[Function, int], SSAValue] = {}
        self.symbol_values : dict[str, SSAValue] = {}
        module = builtin.ModuleOp(Region([block := Block([])]))
        with ImplicitBuilder(block):
            # Get all functions used in the equations
            functions = OrderedSet()
            for eq in eqs:
                if isinstance(eq, Eq):
                    for f in retrieve_functions(eq):
                        functions.add(f.function)
                elif isinstance(eq, Injection):
                    # import pdb; pdb.set_trace()
                    functions.add(eq.field.function)
                    for f in retrieve_functions(eq.expr):
                        if isinstance(f, PointSource):
                            functions.add(f._coordinates)
                        functions.add(f.function)

                else:
                    raise NotImplementedError(f"Expression {eq} of type {type(eq)} not supported")
            self.time_buffers : list[TimeFunction] = []
            self.functions : list[Function] = []
            for f in functions:
                match f:
                    case TimeFunction():
                        for i in range(f.time_size):
                            self.time_buffers.append((f, i))
                    case Function():
                        self.functions.append(f)
                    case PointSource():
                        self.functions.append(f.coordinates)
                        self.functions.append(f)
                    case _:
                        raise NotImplementedError(f"Function of type {type(f)} not supported")

            # For each used time_buffer, define a stencil.field type for the function.
            # Those represent DeVito's buffers in xDSL/stencil terms.
            fields_types = [field_from_function(f) for (f, _) in self.time_buffers]
            fields_types += [field_from_function(f) for f in self.functions]
            # Get the operator name to name the function accordingly.
            name = kwargs.get("name", "Kernel")
            # Create a function with the fields as arguments.
            xdsl_func = func.FuncOp(name, (fields_types, []))

            # Store in self.function_args a mapping from time_buffers to their
            # corresponding function arguments, for easier access later.
            self.function_args = {}
            for i, (f, t) in enumerate(self.time_buffers):
                # Also define argument names to help with debugging
                xdsl_func.body.block.args[i].name_hint = f._C_name + str(t)
                self.function_args[(f, t)] = xdsl_func.body.block.args[i]
            for i, f in enumerate(self.functions):
                xdsl_func.body.block.args[len(self.time_buffers)+i].name_hint = f._C_name
                self.function_args[(f, 0)] = xdsl_func.body.block.args[len(self.time_buffers)+i]

            # Union operation?
            self.function_values |= self.function_args

            # Move on to generate the function body
            with ImplicitBuilder(xdsl_func.body.block):

                # Get the stepping dimension, if there is any in the whole input
                time_functions = [f for (f,_) in self.time_buffers]
                dimensions = {
                    d for f in (self.functions + time_functions) for d in f.dimensions
                }

                step_dim = next((d for d in dimensions if isinstance(d, SteppingDimension)), None)
                if step_dim is not None:
                    self.build_time_loop(eqs, step_dim, **kwargs)
                else:
                    self.generate_equations(eqs, **kwargs)

                # func wants a return
                func.Return()

        return module

    def _ensure_same_type(self, *vals: SSAValue):
        if all(isinstance(val.type, builtin.IntegerAttr) for val in vals):
            return vals
        if all(isinstance(val.type, builtin.IndexType) for val in vals):
            return vals
        if all(is_float(val) for val in vals):
            return vals
        # not everything homogeneous
        cast_to_floats = True
        if all(is_int(val) for val in vals):
            cast_to_floats = False
        processed = []
        for val in vals:
            if cast_to_floats and is_float(val):
                processed.append(val)
                continue
            if (not cast_to_floats) and isinstance(val.type, builtin.IndexType):
                processed.append(val)
                continue
            # if the val is the result of a arith.constant with no uses,
            # we change the type of the arith.constant to our desired type
            if (
                isinstance(val, OpResult)
                and isinstance(val.op, arith.Constant)
                and val.uses == 0
            ):
                if cast_to_floats:
                    val.type = builtin.f32
                    val.op.attributes["value"] = builtin.FloatAttr(
                        float(val.op.value.value.data), builtin.f32
                    )
                else:
                    val.type = builtin.IndexType()
                    val.op.value.type = builtin.IndexType()
                processed.append(val)
                continue
            # insert a cast op
            if cast_to_floats:
                if val.type == builtin.IndexType():
                    val = arith.IndexCastOp(val, builtin.i64).result
                conv = arith.SIToFPOp(val, builtin.f32)
            else:
                conv = arith.IndexCastOp(val, builtin.IndexType())
            processed.append(conv.result)
        return processed

# -------------------------------------------------------- ####
#                                                          ####
#           devito.stencil  ---> stencil dialect           ####
#                                                          ####
# -------------------------------------------------------- ####


class GPURewritePattern(RewritePattern):
    """
    Base class for GPU rewrite patterns
    """
    pass

@dataclass
class WrapFunctionWithTransfers(GPURewritePattern):
    func_name: str
    done: bool = field(default=False)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if op.sym_name.data != self.func_name or self.done:
            return
        self.done = True

        op.sym_name = builtin.StringAttr("gpu_kernel")
        print("Doing GPU STUFF")
        # GPU STUFF
        wrapper = func.FuncOp(self.func_name, op.function_type, Region(Block([func.Return()], arg_types=op.function_type.inputs)))
        body = wrapper.body.block
        wrapper.body.block.insert_op_before(func.Call("gpu_kernel", body.args, []), body.last_op)
        for arg in wrapper.args:
            shapetype = arg.type
            if isinstance(shapetype, stencil.FieldType):
                memref_type = memref.MemRefType.from_element_type_and_shape(shapetype.get_element_type(), shapetype.get_shape())
                alloc = gpu.AllocOp(memref.MemRefType.from_element_type_and_shape(shapetype.get_element_type(), shapetype.get_shape()))
                outcast = builtin.UnrealizedConversionCastOp.get(alloc, shapetype)
                arg.replace_by(outcast.results[0])
                incast = builtin.UnrealizedConversionCastOp.get(arg, memref_type)
                copy = gpu.MemcpyOp(source=incast, destination=alloc)
                body.insert_ops_before([alloc, outcast, incast, copy], body.ops.first)

                copy_out = gpu.MemcpyOp(source=alloc, destination=incast)
                dealloc = gpu.DeallocOp(alloc)
                body.insert_ops_before([copy_out, dealloc], body.ops.last)
        rewriter.insert_op_after_matched_op(wrapper)


def get_containing_func(op: Operation) -> func.FuncOp | None:
    while op is not None and not isinstance(op, func.FuncOp):
        op = op.parent_op()
    return op


@dataclass
class _InsertSymbolicConstants(RewritePattern):
    """
    Replace LoadSymbolic ops with their constant values. copilot: done
    """
    known_symbols: dict[str, int | float]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.LoadSymbolic, rewriter: PatternRewriter, /):
        symb_name = op.symbol_name.data
        if symb_name not in self.known_symbols:
            return

        if isinstance(op.result.type, (builtin.Float32Type, builtin.Float64Type)):
            rewriter.replace_matched_op(
                arith.Constant(builtin.FloatAttr
                    (
                        float(self.known_symbols[symb_name]), op.result.type
                    )
                )
            )
        elif isinstance(op.result.type, (builtin.IntegerType, builtin.IndexType)):
            rewriter.replace_matched_op(
                arith.Constant.from_int_and_width(
                    int(self.known_symbols[symb_name]), op.result.type
                )
            )


class _LowerLoadSymbolicToFuncArgs(RewritePattern):

    func_to_args: dict[func.FuncOp, dict[str, SSAValue]]

    def __init__(self):
        from collections import defaultdict

        self.func_to_args = defaultdict(dict)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.LoadSymbolic, rewriter: PatternRewriter, /):
        parent: func.FuncOp | None = get_containing_func(op)
        assert parent is not None
        args = list(parent.body.block.args)
        symb_name = op.symbol_name.data

        try:
            arg_index = [a.name_hint for a in args].index(symb_name)
        except ValueError:
            arg_index = -1

        if arg_index == -1:
            body = parent.body.block
            args.append(body.insert_arg(op.result.type, len(body.args)))
            arg_index = len(args) - 1

        op.result.replace_by(args[arg_index])

        rewriter.erase_matched_op()
        parent.update_function_type()


def finalize_module_with_globals(module: builtin.ModuleOp, known_symbols: dict[str, Any],
                                 gpu_boilerplate):
    """
    This function finalizes a module by replacing all symbolic constants with their
    values in the module. This is necessary to have a complete module that can be
    executed. [copilot: done]
    """
    patterns = [
        _InsertSymbolicConstants(known_symbols),
        _LowerLoadSymbolicToFuncArgs(),
    ]
    rewriter = GreedyRewritePatternApplier(patterns)
    PatternRewriteWalker(rewriter).rewrite_module(module)

    # GPU boilerplate
    if gpu_boilerplate:
        walker = PatternRewriteWalker(GreedyRewritePatternApplier([WrapFunctionWithTransfers('apply_kernel')]))  # noqa
        walker.rewrite_module(module)
