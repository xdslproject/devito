import numpy as np

from xdsl.dialects import builtin
from xdsl.ir import SSAValue


def is_int(val: SSAValue):
    return isinstance(val.type, (builtin.IntegerType, builtin.IndexType))


def is_float(val: SSAValue):
    return val.type in (builtin.f32, builtin.f64)


dtypes_to_xdsltypes = {
    np.float32: builtin.f32,
    np.float64: builtin.f64,
    np.int32: builtin.i32,
    np.int64: builtin.i64,
}
