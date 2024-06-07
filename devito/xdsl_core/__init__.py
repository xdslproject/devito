from devito.arch import Cpu64, Device

from devito.xdsl_core.xdsl_cpu import XdslnoopOperator, XdslAdvOperator

from devito.xdsl_core.xdsl_gpu import XdslAdvDeviceOperator
from devito.operator.registry import operator_registry

# Register XDSL Operators
operator_registry.add(XdslnoopOperator, Cpu64, 'xdsl-noop', 'C')
operator_registry.add(XdslAdvOperator, Cpu64, 'xdsl-noop', 'openmp')

operator_registry.add(XdslAdvOperator, Cpu64, 'xdsl', 'C')
operator_registry.add(XdslAdvOperator, Cpu64, 'xdsl', 'openmp')
operator_registry.add(XdslAdvDeviceOperator, Device, 'xdsl', 'openacc')
