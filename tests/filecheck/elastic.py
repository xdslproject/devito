# RUN: python %s | filecheck %s

# Test dynamically registering a dialect from an IRDL file

from devito import (Grid, TensorTimeFunction, VectorTimeFunction, div, grad, diag, solve,
                    Operator, Eq, Constant, SpaceDimension)
from devito.tools import as_tuple
from examples.seismic.source import RickerSource, TimeAxis

import numpy as np

if __name__ == "__main__":

    shape = (101, 101)

    so = 4
    nt = 20

    # Initial grid: km x km, with spacing
    shape = shape  # Number of grid point (nx, nz)
    spacing = as_tuple(10.0 for _ in range(len(shape)))
    extent = tuple([s*(n-1) for s, n in zip(spacing, shape)])

    x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))  # noqa
    z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[1]/(shape[1]-1)))  # noqa
    grid = Grid(extent=extent, shape=shape, dimensions=(x, z))

    # Timestep size from Eq. 7 with V_p=6000. and dx=100
    t0, tn = 0., nt
    dt = (10. / np.sqrt(2.)) / 6.
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    src = RickerSource(name='src', grid=grid, f0=0.01, time_range=time_range)
    src.coordinates.data[:] = [250., 250.]

    # Now we create the velocity and pressure fields
    v = VectorTimeFunction(name='v', grid=grid, space_order=so, time_order=1)
    tau = TensorTimeFunction(name='t', grid=grid, space_order=so, time_order=1)

    # We need some initial conditions
    V_p = 2.0
    V_s = 1.0
    density = 1.8

    # The source injection term
    src_xx = src.inject(field=tau.forward[0, 0], expr=src)
    src_zz = src.inject(field=tau.forward[1, 1], expr=src)

    # Thorbecke's parameter notation
    cp2 = V_p*V_p
    cs2 = V_s*V_s
    ro = 1/density

    mu = cs2*density
    l = (cp2*density - 2*mu)

    # First order elastic wave equation
    pde_v = v.dt - ro * div(tau)
    pde_tau = (tau.dt - l * diag(div(v.forward)) - mu * (grad(v.forward) +
               grad(v.forward).transpose(inner=False)))

    # Time update
    u_v = Eq(v.forward, solve(pde_v, v.forward))
    u_t = Eq(tau.forward, solve(pde_tau, tau.forward))

    # Inject sources. We use it to preinject data
    # Up to here, let's only use Devito
    op = Operator([u_v] + [u_t] + src_xx + src_zz)
    op(dt=dt)

    opx = Operator([u_v] + [u_t], opt='xdsl')
    opx(dt=dt, time_M=nt)

    opx._module.verify()
    print(opx._module)


# CHECK:      builtin.module {
# CHECK-NEXT:   func.func @Kernel(%v_x_vec0 : !stencil.field<[-4,105]x[-4,105]xf32>, %v_x_vec1 : !stencil.field<[-4,105]x[-4,105]xf32>, %v_z_vec0 : !stencil.field<[-4,105]x[-4,105]xf32>, %v_z_vec1 : !stencil.field<[-4,105]x[-4,105]xf32>, %t_xz_vec0 : !stencil.field<[-4,105]x[-4,105]xf32>, %t_xz_vec1 : !stencil.field<[-4,105]x[-4,105]xf32>, %t_xx_vec0 : !stencil.field<[-4,105]x[-4,105]xf32>, %t_xx_vec1 : !stencil.field<[-4,105]x[-4,105]xf32>, %t_zz_vec0 : !stencil.field<[-4,105]x[-4,105]xf32>, %t_zz_vec1 : !stencil.field<[-4,105]x[-4,105]xf32>, %timers : !llvm.ptr) {
# CHECK-NEXT:     %0 = func.call @timer_start() : () -> f64
# CHECK-NEXT:     %time_m = arith.constant 0 : index
# CHECK-NEXT:     %time_M = arith.constant 20 : index
# CHECK-NEXT:     %1 = arith.constant 1 : index
# CHECK-NEXT:     %2 = arith.addi %time_M, %1 : index
# CHECK-NEXT:     %step = arith.constant 1 : index
# CHECK-NEXT:     %3, %4, %5, %6, %7, %8, %9, %10, %11, %12 = scf.for %time = %time_m to %2 step %step iter_args(%v_x_t0 = %v_x_vec0, %v_x_t1 = %v_x_vec1, %v_z_t0 = %v_z_vec0, %v_z_t1 = %v_z_vec1, %t_xz_t0 = %t_xz_vec0, %t_xz_t1 = %t_xz_vec1, %t_xx_t0 = %t_xx_vec0, %t_xx_t1 = %t_xx_vec1, %t_zz_t0 = %t_zz_vec0, %t_zz_t1 = %t_zz_vec1) -> (!stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>) {
# CHECK-NEXT:       %t_xx_t0_temp = stencil.load %t_xx_t0 : !stencil.field<[-4,105]x[-4,105]xf32> -> !stencil.temp<?x?xf32>
# CHECK-NEXT:       %t_xz_t0_temp = stencil.load %t_xz_t0 : !stencil.field<[-4,105]x[-4,105]xf32> -> !stencil.temp<?x?xf32>
# CHECK-NEXT:       %v_x_t0_temp = stencil.load %v_x_t0 : !stencil.field<[-4,105]x[-4,105]xf32> -> !stencil.temp<?x?xf32>
# CHECK-NEXT:       %v_x_t1_temp = stencil.apply(%t_xx_t0_blk = %t_xx_t0_temp : !stencil.temp<?x?xf32>, %t_xz_t0_blk = %t_xz_t0_temp : !stencil.temp<?x?xf32>, %v_x_t0_blk = %v_x_t0_temp : !stencil.temp<?x?xf32>) -> (!stencil.temp<?x?xf32>) {
# CHECK-NEXT:         %13 = arith.constant 5.555556e-01 : f32
# CHECK-NEXT:         %14 = arith.constant 1.125000e+00 : f32
# CHECK-NEXT:         %h_x = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %15 = arith.constant -1 : i64
# CHECK-NEXT:         %16 = "math.fpowi"(%h_x, %15) : (f32, i64) -> f32
# CHECK-NEXT:         %17 = stencil.access %t_xx_t0_blk[1, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %18 = arith.mulf %14, %16 : f32
# CHECK-NEXT:         %19 = arith.mulf %18, %17 : f32
# CHECK-NEXT:         %20 = arith.constant 4.166667e-02 : f32
# CHECK-NEXT:         %h_x_1 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %21 = arith.constant -1 : i64
# CHECK-NEXT:         %22 = "math.fpowi"(%h_x_1, %21) : (f32, i64) -> f32
# CHECK-NEXT:         %23 = stencil.access %t_xx_t0_blk[-1, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %24 = arith.mulf %20, %22 : f32
# CHECK-NEXT:         %25 = arith.mulf %24, %23 : f32
# CHECK-NEXT:         %26 = arith.constant -1.125000e+00 : f32
# CHECK-NEXT:         %h_x_2 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %27 = arith.constant -1 : i64
# CHECK-NEXT:         %28 = "math.fpowi"(%h_x_2, %27) : (f32, i64) -> f32
# CHECK-NEXT:         %29 = stencil.access %t_xx_t0_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %30 = arith.mulf %26, %28 : f32
# CHECK-NEXT:         %31 = arith.mulf %30, %29 : f32
# CHECK-NEXT:         %32 = arith.constant -4.166667e-02 : f32
# CHECK-NEXT:         %h_x_3 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %33 = arith.constant -1 : i64
# CHECK-NEXT:         %34 = "math.fpowi"(%h_x_3, %33) : (f32, i64) -> f32
# CHECK-NEXT:         %35 = stencil.access %t_xx_t0_blk[2, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %36 = arith.mulf %32, %34 : f32
# CHECK-NEXT:         %37 = arith.mulf %36, %35 : f32
# CHECK-NEXT:         %38 = arith.addf %19, %25 : f32
# CHECK-NEXT:         %39 = arith.addf %38, %31 : f32
# CHECK-NEXT:         %40 = arith.addf %39, %37 : f32
# CHECK-NEXT:         %41 = arith.mulf %13, %40 : f32
# CHECK-NEXT:         %42 = arith.constant 5.555556e-01 : f32
# CHECK-NEXT:         %43 = arith.constant 1.125000e+00 : f32
# CHECK-NEXT:         %h_z = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %44 = arith.constant -1 : i64
# CHECK-NEXT:         %45 = "math.fpowi"(%h_z, %44) : (f32, i64) -> f32
# CHECK-NEXT:         %46 = stencil.access %t_xz_t0_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %47 = arith.mulf %43, %45 : f32
# CHECK-NEXT:         %48 = arith.mulf %47, %46 : f32
# CHECK-NEXT:         %49 = arith.constant 4.166667e-02 : f32
# CHECK-NEXT:         %h_z_1 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %50 = arith.constant -1 : i64
# CHECK-NEXT:         %51 = "math.fpowi"(%h_z_1, %50) : (f32, i64) -> f32
# CHECK-NEXT:         %52 = stencil.access %t_xz_t0_blk[0, -2] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %53 = arith.mulf %49, %51 : f32
# CHECK-NEXT:         %54 = arith.mulf %53, %52 : f32
# CHECK-NEXT:         %55 = arith.constant -1.125000e+00 : f32
# CHECK-NEXT:         %h_z_2 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %56 = arith.constant -1 : i64
# CHECK-NEXT:         %57 = "math.fpowi"(%h_z_2, %56) : (f32, i64) -> f32
# CHECK-NEXT:         %58 = stencil.access %t_xz_t0_blk[0, -1] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %59 = arith.mulf %55, %57 : f32
# CHECK-NEXT:         %60 = arith.mulf %59, %58 : f32
# CHECK-NEXT:         %61 = arith.constant -4.166667e-02 : f32
# CHECK-NEXT:         %h_z_3 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %62 = arith.constant -1 : i64
# CHECK-NEXT:         %63 = "math.fpowi"(%h_z_3, %62) : (f32, i64) -> f32
# CHECK-NEXT:         %64 = stencil.access %t_xz_t0_blk[0, 1] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %65 = arith.mulf %61, %63 : f32
# CHECK-NEXT:         %66 = arith.mulf %65, %64 : f32
# CHECK-NEXT:         %67 = arith.addf %48, %54 : f32
# CHECK-NEXT:         %68 = arith.addf %67, %60 : f32
# CHECK-NEXT:         %69 = arith.addf %68, %66 : f32
# CHECK-NEXT:         %70 = arith.mulf %42, %69 : f32
# CHECK-NEXT:         %dt = arith.constant 1.178511e+00 : f32
# CHECK-NEXT:         %71 = arith.constant -1 : i64
# CHECK-NEXT:         %72 = "math.fpowi"(%dt, %71) : (f32, i64) -> f32
# CHECK-NEXT:         %73 = stencil.access %v_x_t0_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %74 = arith.mulf %72, %73 : f32
# CHECK-NEXT:         %75 = arith.addf %41, %70 : f32
# CHECK-NEXT:         %76 = arith.addf %75, %74 : f32
# CHECK-NEXT:         %dt_1 = arith.constant 1.178511e+00 : f32
# CHECK-NEXT:         %77 = arith.mulf %76, %dt_1 : f32
# CHECK-NEXT:         stencil.return %77 : f32
# CHECK-NEXT:       }
# CHECK-NEXT:       %v_x_t1_temp_1 = stencil.store %v_x_t1_temp to %v_x_t1 ([0, 0] : [101, 101]) : !stencil.temp<?x?xf32> to !stencil.field<[-4,105]x[-4,105]xf32> with_halo : !stencil.temp<?x?xf32>
# CHECK-NEXT:       %t_zz_t0_temp = stencil.load %t_zz_t0 : !stencil.field<[-4,105]x[-4,105]xf32> -> !stencil.temp<?x?xf32>
# CHECK-NEXT:       %v_z_t0_temp = stencil.load %v_z_t0 : !stencil.field<[-4,105]x[-4,105]xf32> -> !stencil.temp<?x?xf32>
# CHECK-NEXT:       %v_z_t1_temp = stencil.apply(%t_xz_t0_blk = %t_xz_t0_temp : !stencil.temp<?x?xf32>, %t_zz_t0_blk = %t_zz_t0_temp : !stencil.temp<?x?xf32>, %v_z_t0_blk = %v_z_t0_temp : !stencil.temp<?x?xf32>) -> (!stencil.temp<?x?xf32>) {
# CHECK-NEXT:         %13 = arith.constant 5.555556e-01 : f32
# CHECK-NEXT:         %14 = arith.constant 1.125000e+00 : f32
# CHECK-NEXT:         %h_x = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %15 = arith.constant -1 : i64
# CHECK-NEXT:         %16 = "math.fpowi"(%h_x, %15) : (f32, i64) -> f32
# CHECK-NEXT:         %17 = stencil.access %t_xz_t0_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %18 = arith.mulf %14, %16 : f32
# CHECK-NEXT:         %19 = arith.mulf %18, %17 : f32
# CHECK-NEXT:         %20 = arith.constant 4.166667e-02 : f32
# CHECK-NEXT:         %h_x_1 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %21 = arith.constant -1 : i64
# CHECK-NEXT:         %22 = "math.fpowi"(%h_x_1, %21) : (f32, i64) -> f32
# CHECK-NEXT:         %23 = stencil.access %t_xz_t0_blk[-2, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %24 = arith.mulf %20, %22 : f32
# CHECK-NEXT:         %25 = arith.mulf %24, %23 : f32
# CHECK-NEXT:         %26 = arith.constant -1.125000e+00 : f32
# CHECK-NEXT:         %h_x_2 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %27 = arith.constant -1 : i64
# CHECK-NEXT:         %28 = "math.fpowi"(%h_x_2, %27) : (f32, i64) -> f32
# CHECK-NEXT:         %29 = stencil.access %t_xz_t0_blk[-1, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %30 = arith.mulf %26, %28 : f32
# CHECK-NEXT:         %31 = arith.mulf %30, %29 : f32
# CHECK-NEXT:         %32 = arith.constant -4.166667e-02 : f32
# CHECK-NEXT:         %h_x_3 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %33 = arith.constant -1 : i64
# CHECK-NEXT:         %34 = "math.fpowi"(%h_x_3, %33) : (f32, i64) -> f32
# CHECK-NEXT:         %35 = stencil.access %t_xz_t0_blk[1, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %36 = arith.mulf %32, %34 : f32
# CHECK-NEXT:         %37 = arith.mulf %36, %35 : f32
# CHECK-NEXT:         %38 = arith.addf %19, %25 : f32
# CHECK-NEXT:         %39 = arith.addf %38, %31 : f32
# CHECK-NEXT:         %40 = arith.addf %39, %37 : f32
# CHECK-NEXT:         %41 = arith.mulf %13, %40 : f32
# CHECK-NEXT:         %42 = arith.constant 5.555556e-01 : f32
# CHECK-NEXT:         %43 = arith.constant 1.125000e+00 : f32
# CHECK-NEXT:         %h_z = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %44 = arith.constant -1 : i64
# CHECK-NEXT:         %45 = "math.fpowi"(%h_z, %44) : (f32, i64) -> f32
# CHECK-NEXT:         %46 = stencil.access %t_zz_t0_blk[0, 1] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %47 = arith.mulf %43, %45 : f32
# CHECK-NEXT:         %48 = arith.mulf %47, %46 : f32
# CHECK-NEXT:         %49 = arith.constant 4.166667e-02 : f32
# CHECK-NEXT:         %h_z_1 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %50 = arith.constant -1 : i64
# CHECK-NEXT:         %51 = "math.fpowi"(%h_z_1, %50) : (f32, i64) -> f32
# CHECK-NEXT:         %52 = stencil.access %t_zz_t0_blk[0, -1] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %53 = arith.mulf %49, %51 : f32
# CHECK-NEXT:         %54 = arith.mulf %53, %52 : f32
# CHECK-NEXT:         %55 = arith.constant -1.125000e+00 : f32
# CHECK-NEXT:         %h_z_2 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %56 = arith.constant -1 : i64
# CHECK-NEXT:         %57 = "math.fpowi"(%h_z_2, %56) : (f32, i64) -> f32
# CHECK-NEXT:         %58 = stencil.access %t_zz_t0_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %59 = arith.mulf %55, %57 : f32
# CHECK-NEXT:         %60 = arith.mulf %59, %58 : f32
# CHECK-NEXT:         %61 = arith.constant -4.166667e-02 : f32
# CHECK-NEXT:         %h_z_3 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %62 = arith.constant -1 : i64
# CHECK-NEXT:         %63 = "math.fpowi"(%h_z_3, %62) : (f32, i64) -> f32
# CHECK-NEXT:         %64 = stencil.access %t_zz_t0_blk[0, 2] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %65 = arith.mulf %61, %63 : f32
# CHECK-NEXT:         %66 = arith.mulf %65, %64 : f32
# CHECK-NEXT:         %67 = arith.addf %48, %54 : f32
# CHECK-NEXT:         %68 = arith.addf %67, %60 : f32
# CHECK-NEXT:         %69 = arith.addf %68, %66 : f32
# CHECK-NEXT:         %70 = arith.mulf %42, %69 : f32
# CHECK-NEXT:         %dt = arith.constant 1.178511e+00 : f32
# CHECK-NEXT:         %71 = arith.constant -1 : i64
# CHECK-NEXT:         %72 = "math.fpowi"(%dt, %71) : (f32, i64) -> f32
# CHECK-NEXT:         %73 = stencil.access %v_z_t0_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %74 = arith.mulf %72, %73 : f32
# CHECK-NEXT:         %75 = arith.addf %41, %70 : f32
# CHECK-NEXT:         %76 = arith.addf %75, %74 : f32
# CHECK-NEXT:         %dt_1 = arith.constant 1.178511e+00 : f32
# CHECK-NEXT:         %77 = arith.mulf %76, %dt_1 : f32
# CHECK-NEXT:         stencil.return %77 : f32
# CHECK-NEXT:       }
# CHECK-NEXT:       %v_z_t1_temp_1 = stencil.store %v_z_t1_temp to %v_z_t1 ([0, 0] : [101, 101]) : !stencil.temp<?x?xf32> to !stencil.field<[-4,105]x[-4,105]xf32> with_halo : !stencil.temp<?x?xf32>
# CHECK-NEXT:       %t_xx_t1_temp = stencil.apply(%v_z_t1_blk = %v_z_t1_temp_1 : !stencil.temp<?x?xf32>, %v_x_t1_blk = %v_x_t1_temp_1 : !stencil.temp<?x?xf32>, %t_xx_t0_blk = %t_xx_t0_temp : !stencil.temp<?x?xf32>) -> (!stencil.temp<?x?xf32>) {
# CHECK-NEXT:         %13 = arith.constant 3.600000e+00 : f32
# CHECK-NEXT:         %14 = arith.constant 1.125000e+00 : f32
# CHECK-NEXT:         %h_z = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %15 = arith.constant -1 : i64
# CHECK-NEXT:         %16 = "math.fpowi"(%h_z, %15) : (f32, i64) -> f32
# CHECK-NEXT:         %17 = stencil.access %v_z_t1_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %18 = arith.mulf %14, %16 : f32
# CHECK-NEXT:         %19 = arith.mulf %18, %17 : f32
# CHECK-NEXT:         %20 = arith.constant 4.166667e-02 : f32
# CHECK-NEXT:         %h_z_1 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %21 = arith.constant -1 : i64
# CHECK-NEXT:         %22 = "math.fpowi"(%h_z_1, %21) : (f32, i64) -> f32
# CHECK-NEXT:         %23 = stencil.access %v_z_t1_blk[0, -2] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %24 = arith.mulf %20, %22 : f32
# CHECK-NEXT:         %25 = arith.mulf %24, %23 : f32
# CHECK-NEXT:         %26 = arith.constant -1.125000e+00 : f32
# CHECK-NEXT:         %h_z_2 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %27 = arith.constant -1 : i64
# CHECK-NEXT:         %28 = "math.fpowi"(%h_z_2, %27) : (f32, i64) -> f32
# CHECK-NEXT:         %29 = stencil.access %v_z_t1_blk[0, -1] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %30 = arith.mulf %26, %28 : f32
# CHECK-NEXT:         %31 = arith.mulf %30, %29 : f32
# CHECK-NEXT:         %32 = arith.constant -4.166667e-02 : f32
# CHECK-NEXT:         %h_z_3 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %33 = arith.constant -1 : i64
# CHECK-NEXT:         %34 = "math.fpowi"(%h_z_3, %33) : (f32, i64) -> f32
# CHECK-NEXT:         %35 = stencil.access %v_z_t1_blk[0, 1] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %36 = arith.mulf %32, %34 : f32
# CHECK-NEXT:         %37 = arith.mulf %36, %35 : f32
# CHECK-NEXT:         %38 = arith.addf %19, %25 : f32
# CHECK-NEXT:         %39 = arith.addf %38, %31 : f32
# CHECK-NEXT:         %40 = arith.addf %39, %37 : f32
# CHECK-NEXT:         %41 = arith.mulf %13, %40 : f32
# CHECK-NEXT:         %42 = arith.constant 7.200000e+00 : f32
# CHECK-NEXT:         %43 = arith.constant 1.125000e+00 : f32
# CHECK-NEXT:         %h_x = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %44 = arith.constant -1 : i64
# CHECK-NEXT:         %45 = "math.fpowi"(%h_x, %44) : (f32, i64) -> f32
# CHECK-NEXT:         %46 = stencil.access %v_x_t1_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %47 = arith.mulf %43, %45 : f32
# CHECK-NEXT:         %48 = arith.mulf %47, %46 : f32
# CHECK-NEXT:         %49 = arith.constant 4.166667e-02 : f32
# CHECK-NEXT:         %h_x_1 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %50 = arith.constant -1 : i64
# CHECK-NEXT:         %51 = "math.fpowi"(%h_x_1, %50) : (f32, i64) -> f32
# CHECK-NEXT:         %52 = stencil.access %v_x_t1_blk[-2, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %53 = arith.mulf %49, %51 : f32
# CHECK-NEXT:         %54 = arith.mulf %53, %52 : f32
# CHECK-NEXT:         %55 = arith.constant -1.125000e+00 : f32
# CHECK-NEXT:         %h_x_2 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %56 = arith.constant -1 : i64
# CHECK-NEXT:         %57 = "math.fpowi"(%h_x_2, %56) : (f32, i64) -> f32
# CHECK-NEXT:         %58 = stencil.access %v_x_t1_blk[-1, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %59 = arith.mulf %55, %57 : f32
# CHECK-NEXT:         %60 = arith.mulf %59, %58 : f32
# CHECK-NEXT:         %61 = arith.constant -4.166667e-02 : f32
# CHECK-NEXT:         %h_x_3 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %62 = arith.constant -1 : i64
# CHECK-NEXT:         %63 = "math.fpowi"(%h_x_3, %62) : (f32, i64) -> f32
# CHECK-NEXT:         %64 = stencil.access %v_x_t1_blk[1, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %65 = arith.mulf %61, %63 : f32
# CHECK-NEXT:         %66 = arith.mulf %65, %64 : f32
# CHECK-NEXT:         %67 = arith.addf %48, %54 : f32
# CHECK-NEXT:         %68 = arith.addf %67, %60 : f32
# CHECK-NEXT:         %69 = arith.addf %68, %66 : f32
# CHECK-NEXT:         %70 = arith.mulf %42, %69 : f32
# CHECK-NEXT:         %dt = arith.constant 1.178511e+00 : f32
# CHECK-NEXT:         %71 = arith.constant -1 : i64
# CHECK-NEXT:         %72 = "math.fpowi"(%dt, %71) : (f32, i64) -> f32
# CHECK-NEXT:         %73 = stencil.access %t_xx_t0_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %74 = arith.mulf %72, %73 : f32
# CHECK-NEXT:         %75 = arith.addf %41, %70 : f32
# CHECK-NEXT:         %76 = arith.addf %75, %74 : f32
# CHECK-NEXT:         %dt_1 = arith.constant 1.178511e+00 : f32
# CHECK-NEXT:         %77 = arith.mulf %76, %dt_1 : f32
# CHECK-NEXT:         stencil.return %77 : f32
# CHECK-NEXT:       }
# CHECK-NEXT:       %t_xx_t1_temp_1 = stencil.store %t_xx_t1_temp to %t_xx_t1 ([0, 0] : [101, 101]) : !stencil.temp<?x?xf32> to !stencil.field<[-4,105]x[-4,105]xf32> with_halo : !stencil.temp<?x?xf32>
# CHECK-NEXT:       %t_xz_t1_temp = stencil.apply(%v_z_t1_blk = %v_z_t1_temp_1 : !stencil.temp<?x?xf32>, %v_x_t1_blk = %v_x_t1_temp_1 : !stencil.temp<?x?xf32>, %t_xz_t0_blk = %t_xz_t0_temp : !stencil.temp<?x?xf32>) -> (!stencil.temp<?x?xf32>) {
# CHECK-NEXT:         %13 = arith.constant 1.800000e+00 : f32
# CHECK-NEXT:         %14 = arith.constant 1.125000e+00 : f32
# CHECK-NEXT:         %h_x = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %15 = arith.constant -1 : i64
# CHECK-NEXT:         %16 = "math.fpowi"(%h_x, %15) : (f32, i64) -> f32
# CHECK-NEXT:         %17 = stencil.access %v_z_t1_blk[1, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %18 = arith.mulf %14, %16 : f32
# CHECK-NEXT:         %19 = arith.mulf %18, %17 : f32
# CHECK-NEXT:         %20 = arith.constant 4.166667e-02 : f32
# CHECK-NEXT:         %h_x_1 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %21 = arith.constant -1 : i64
# CHECK-NEXT:         %22 = "math.fpowi"(%h_x_1, %21) : (f32, i64) -> f32
# CHECK-NEXT:         %23 = stencil.access %v_z_t1_blk[-1, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %24 = arith.mulf %20, %22 : f32
# CHECK-NEXT:         %25 = arith.mulf %24, %23 : f32
# CHECK-NEXT:         %26 = arith.constant -1.125000e+00 : f32
# CHECK-NEXT:         %h_x_2 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %27 = arith.constant -1 : i64
# CHECK-NEXT:         %28 = "math.fpowi"(%h_x_2, %27) : (f32, i64) -> f32
# CHECK-NEXT:         %29 = stencil.access %v_z_t1_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %30 = arith.mulf %26, %28 : f32
# CHECK-NEXT:         %31 = arith.mulf %30, %29 : f32
# CHECK-NEXT:         %32 = arith.constant -4.166667e-02 : f32
# CHECK-NEXT:         %h_x_3 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %33 = arith.constant -1 : i64
# CHECK-NEXT:         %34 = "math.fpowi"(%h_x_3, %33) : (f32, i64) -> f32
# CHECK-NEXT:         %35 = stencil.access %v_z_t1_blk[2, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %36 = arith.mulf %32, %34 : f32
# CHECK-NEXT:         %37 = arith.mulf %36, %35 : f32
# CHECK-NEXT:         %38 = arith.addf %19, %25 : f32
# CHECK-NEXT:         %39 = arith.addf %38, %31 : f32
# CHECK-NEXT:         %40 = arith.addf %39, %37 : f32
# CHECK-NEXT:         %41 = arith.mulf %13, %40 : f32
# CHECK-NEXT:         %42 = arith.constant 1.800000e+00 : f32
# CHECK-NEXT:         %43 = arith.constant 1.125000e+00 : f32
# CHECK-NEXT:         %h_z = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %44 = arith.constant -1 : i64
# CHECK-NEXT:         %45 = "math.fpowi"(%h_z, %44) : (f32, i64) -> f32
# CHECK-NEXT:         %46 = stencil.access %v_x_t1_blk[0, 1] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %47 = arith.mulf %43, %45 : f32
# CHECK-NEXT:         %48 = arith.mulf %47, %46 : f32
# CHECK-NEXT:         %49 = arith.constant 4.166667e-02 : f32
# CHECK-NEXT:         %h_z_1 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %50 = arith.constant -1 : i64
# CHECK-NEXT:         %51 = "math.fpowi"(%h_z_1, %50) : (f32, i64) -> f32
# CHECK-NEXT:         %52 = stencil.access %v_x_t1_blk[0, -1] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %53 = arith.mulf %49, %51 : f32
# CHECK-NEXT:         %54 = arith.mulf %53, %52 : f32
# CHECK-NEXT:         %55 = arith.constant -1.125000e+00 : f32
# CHECK-NEXT:         %h_z_2 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %56 = arith.constant -1 : i64
# CHECK-NEXT:         %57 = "math.fpowi"(%h_z_2, %56) : (f32, i64) -> f32
# CHECK-NEXT:         %58 = stencil.access %v_x_t1_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %59 = arith.mulf %55, %57 : f32
# CHECK-NEXT:         %60 = arith.mulf %59, %58 : f32
# CHECK-NEXT:         %61 = arith.constant -4.166667e-02 : f32
# CHECK-NEXT:         %h_z_3 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %62 = arith.constant -1 : i64
# CHECK-NEXT:         %63 = "math.fpowi"(%h_z_3, %62) : (f32, i64) -> f32
# CHECK-NEXT:         %64 = stencil.access %v_x_t1_blk[0, 2] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %65 = arith.mulf %61, %63 : f32
# CHECK-NEXT:         %66 = arith.mulf %65, %64 : f32
# CHECK-NEXT:         %67 = arith.addf %48, %54 : f32
# CHECK-NEXT:         %68 = arith.addf %67, %60 : f32
# CHECK-NEXT:         %69 = arith.addf %68, %66 : f32
# CHECK-NEXT:         %70 = arith.mulf %42, %69 : f32
# CHECK-NEXT:         %dt = arith.constant 1.178511e+00 : f32
# CHECK-NEXT:         %71 = arith.constant -1 : i64
# CHECK-NEXT:         %72 = "math.fpowi"(%dt, %71) : (f32, i64) -> f32
# CHECK-NEXT:         %73 = stencil.access %t_xz_t0_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %74 = arith.mulf %72, %73 : f32
# CHECK-NEXT:         %75 = arith.addf %41, %70 : f32
# CHECK-NEXT:         %76 = arith.addf %75, %74 : f32
# CHECK-NEXT:         %dt_1 = arith.constant 1.178511e+00 : f32
# CHECK-NEXT:         %77 = arith.mulf %76, %dt_1 : f32
# CHECK-NEXT:         stencil.return %77 : f32
# CHECK-NEXT:       }
# CHECK-NEXT:       %t_xz_t1_temp_1 = stencil.store %t_xz_t1_temp to %t_xz_t1 ([0, 0] : [101, 101]) : !stencil.temp<?x?xf32> to !stencil.field<[-4,105]x[-4,105]xf32> with_halo : !stencil.temp<?x?xf32>
# CHECK-NEXT:       %t_zz_t1_temp = stencil.apply(%v_x_t1_blk = %v_x_t1_temp_1 : !stencil.temp<?x?xf32>, %v_z_t1_blk = %v_z_t1_temp_1 : !stencil.temp<?x?xf32>, %t_zz_t0_blk = %t_zz_t0_temp : !stencil.temp<?x?xf32>) -> (!stencil.temp<?x?xf32>) {
# CHECK-NEXT:         %13 = arith.constant 3.600000e+00 : f32
# CHECK-NEXT:         %14 = arith.constant 1.125000e+00 : f32
# CHECK-NEXT:         %h_x = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %15 = arith.constant -1 : i64
# CHECK-NEXT:         %16 = "math.fpowi"(%h_x, %15) : (f32, i64) -> f32
# CHECK-NEXT:         %17 = stencil.access %v_x_t1_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %18 = arith.mulf %14, %16 : f32
# CHECK-NEXT:         %19 = arith.mulf %18, %17 : f32
# CHECK-NEXT:         %20 = arith.constant 4.166667e-02 : f32
# CHECK-NEXT:         %h_x_1 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %21 = arith.constant -1 : i64
# CHECK-NEXT:         %22 = "math.fpowi"(%h_x_1, %21) : (f32, i64) -> f32
# CHECK-NEXT:         %23 = stencil.access %v_x_t1_blk[-2, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %24 = arith.mulf %20, %22 : f32
# CHECK-NEXT:         %25 = arith.mulf %24, %23 : f32
# CHECK-NEXT:         %26 = arith.constant -1.125000e+00 : f32
# CHECK-NEXT:         %h_x_2 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %27 = arith.constant -1 : i64
# CHECK-NEXT:         %28 = "math.fpowi"(%h_x_2, %27) : (f32, i64) -> f32
# CHECK-NEXT:         %29 = stencil.access %v_x_t1_blk[-1, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %30 = arith.mulf %26, %28 : f32
# CHECK-NEXT:         %31 = arith.mulf %30, %29 : f32
# CHECK-NEXT:         %32 = arith.constant -4.166667e-02 : f32
# CHECK-NEXT:         %h_x_3 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %33 = arith.constant -1 : i64
# CHECK-NEXT:         %34 = "math.fpowi"(%h_x_3, %33) : (f32, i64) -> f32
# CHECK-NEXT:         %35 = stencil.access %v_x_t1_blk[1, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %36 = arith.mulf %32, %34 : f32
# CHECK-NEXT:         %37 = arith.mulf %36, %35 : f32
# CHECK-NEXT:         %38 = arith.addf %19, %25 : f32
# CHECK-NEXT:         %39 = arith.addf %38, %31 : f32
# CHECK-NEXT:         %40 = arith.addf %39, %37 : f32
# CHECK-NEXT:         %41 = arith.mulf %13, %40 : f32
# CHECK-NEXT:         %42 = arith.constant 7.200000e+00 : f32
# CHECK-NEXT:         %43 = arith.constant 1.125000e+00 : f32
# CHECK-NEXT:         %h_z = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %44 = arith.constant -1 : i64
# CHECK-NEXT:         %45 = "math.fpowi"(%h_z, %44) : (f32, i64) -> f32
# CHECK-NEXT:         %46 = stencil.access %v_z_t1_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %47 = arith.mulf %43, %45 : f32
# CHECK-NEXT:         %48 = arith.mulf %47, %46 : f32
# CHECK-NEXT:         %49 = arith.constant 4.166667e-02 : f32
# CHECK-NEXT:         %h_z_1 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %50 = arith.constant -1 : i64
# CHECK-NEXT:         %51 = "math.fpowi"(%h_z_1, %50) : (f32, i64) -> f32
# CHECK-NEXT:         %52 = stencil.access %v_z_t1_blk[0, -2] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %53 = arith.mulf %49, %51 : f32
# CHECK-NEXT:         %54 = arith.mulf %53, %52 : f32
# CHECK-NEXT:         %55 = arith.constant -1.125000e+00 : f32
# CHECK-NEXT:         %h_z_2 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %56 = arith.constant -1 : i64
# CHECK-NEXT:         %57 = "math.fpowi"(%h_z_2, %56) : (f32, i64) -> f32
# CHECK-NEXT:         %58 = stencil.access %v_z_t1_blk[0, -1] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %59 = arith.mulf %55, %57 : f32
# CHECK-NEXT:         %60 = arith.mulf %59, %58 : f32
# CHECK-NEXT:         %61 = arith.constant -4.166667e-02 : f32
# CHECK-NEXT:         %h_z_3 = arith.constant 1.000000e+01 : f32
# CHECK-NEXT:         %62 = arith.constant -1 : i64
# CHECK-NEXT:         %63 = "math.fpowi"(%h_z_3, %62) : (f32, i64) -> f32
# CHECK-NEXT:         %64 = stencil.access %v_z_t1_blk[0, 1] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %65 = arith.mulf %61, %63 : f32
# CHECK-NEXT:         %66 = arith.mulf %65, %64 : f32
# CHECK-NEXT:         %67 = arith.addf %48, %54 : f32
# CHECK-NEXT:         %68 = arith.addf %67, %60 : f32
# CHECK-NEXT:         %69 = arith.addf %68, %66 : f32
# CHECK-NEXT:         %70 = arith.mulf %42, %69 : f32
# CHECK-NEXT:         %dt = arith.constant 1.178511e+00 : f32
# CHECK-NEXT:         %71 = arith.constant -1 : i64
# CHECK-NEXT:         %72 = "math.fpowi"(%dt, %71) : (f32, i64) -> f32
# CHECK-NEXT:         %73 = stencil.access %t_zz_t0_blk[0, 0] : !stencil.temp<?x?xf32>
# CHECK-NEXT:         %74 = arith.mulf %72, %73 : f32
# CHECK-NEXT:         %75 = arith.addf %41, %70 : f32
# CHECK-NEXT:         %76 = arith.addf %75, %74 : f32
# CHECK-NEXT:         %dt_1 = arith.constant 1.178511e+00 : f32
# CHECK-NEXT:         %77 = arith.mulf %76, %dt_1 : f32
# CHECK-NEXT:         stencil.return %77 : f32
# CHECK-NEXT:       }
# CHECK-NEXT:       %t_zz_t1_temp_1 = stencil.store %t_zz_t1_temp to %t_zz_t1 ([0, 0] : [101, 101]) : !stencil.temp<?x?xf32> to !stencil.field<[-4,105]x[-4,105]xf32> with_halo : !stencil.temp<?x?xf32>
# CHECK-NEXT:       scf.yield %v_x_t1, %v_x_t0, %v_z_t1, %v_z_t0, %t_xz_t1, %t_xz_t0, %t_xx_t1, %t_xx_t0, %t_zz_t1, %t_zz_t0 : !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>, !stencil.field<[-4,105]x[-4,105]xf32>
# CHECK-NEXT:     }
# CHECK-NEXT:     %13 = func.call @timer_end(%0) : (f64) -> f64
# CHECK-NEXT:     "llvm.store"(%13, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func private @timer_start() -> f64
# CHECK-NEXT:   func.func private @timer_end(f64) -> f64
# CHECK-NEXT: }
