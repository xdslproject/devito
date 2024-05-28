import ctypes
import numpy as np

from xdsl.dialects import builtin


f32 = ctypes.c_float
f64 = ctypes.c_double

i32 = ctypes.c_int32
i64 = ctypes.c_int64

index = ctypes.c_size_t

ptr_of = ctypes.POINTER

__all__ = ['dtype_to_xdsltype']


def dtype_to_xdsltype(dtype):
    """Map numpy types to xdsl datatypes."""

    return {
        np.float32: builtin.f32,
        np.float64: builtin.f64,
        np.int32: builtin.i32,
        np.int64: builtin.i64,
    }[dtype]
