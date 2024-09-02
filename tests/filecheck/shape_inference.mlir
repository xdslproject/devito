// RUN: xdsl-opt -p shape-inference %s | filecheck %s

builtin.module {
  func.func @Kernel(%f2_vec0 : !stencil.field<[-2,5]x[-2,5]xf32>, %f2_vec1 : !stencil.field<[-2,5]x[-2,5]xf32>, %timers : !llvm.ptr) {
    %0 = func.call @timer_start() : () -> f64
    %time_m = arith.constant 0 : index
    %time_M = arith.constant 1 : index
    %1 = arith.constant 1 : index
    %2 = arith.addi %time_M, %1 : index
    %step = arith.constant 1 : index
    %3, %4 = scf.for %time = %time_m to %2 step %step iter_args(%f2_t0 = %f2_vec0, %f2_t1 = %f2_vec1) -> (!stencil.field<[-2,5]x[-2,5]xf32>, !stencil.field<[-2,5]x[-2,5]xf32>) {
      %f2_t0_temp = stencil.load %f2_t0 : !stencil.field<[-2,5]x[-2,5]xf32> -> !stencil.temp<?x?xf32>
      %f2_t1_temp = stencil.apply(%f2_t0_blk = %f2_t0_temp : !stencil.temp<?x?xf32>) -> (!stencil.temp<?x?xf32>) {
        %5 = arith.constant 5.000000e-01 : f32
        %h_x = arith.constant 5.000000e-01 : f32
        %6 = arith.constant -2 : i64
        %7 = "math.fpowi"(%h_x, %6) : (f32, i64) -> f32
        %8 = stencil.access %f2_t0_blk[-1, 0] : !stencil.temp<?x?xf32>
        %9 = arith.mulf %7, %8 : f32
        %h_x_1 = arith.constant 5.000000e-01 : f32
        %10 = arith.constant -2 : i64
        %11 = "math.fpowi"(%h_x_1, %10) : (f32, i64) -> f32
        %12 = stencil.access %f2_t0_blk[1, 0] : !stencil.temp<?x?xf32>
        %13 = arith.mulf %11, %12 : f32
        %14 = arith.constant -2.000000e+00 : f32
        %h_x_2 = arith.constant 5.000000e-01 : f32
        %15 = arith.constant -2 : i64
        %16 = "math.fpowi"(%h_x_2, %15) : (f32, i64) -> f32
        %17 = stencil.access %f2_t0_blk[0, 0] : !stencil.temp<?x?xf32>
        %18 = arith.mulf %14, %16 : f32
        %19 = arith.mulf %18, %17 : f32
        %20 = arith.addf %9, %13 : f32
        %21 = arith.addf %20, %19 : f32
        %22 = arith.mulf %5, %21 : f32
        %23 = arith.constant 5.000000e-01 : f32
        %h_y = arith.constant 5.000000e-01 : f32
        %24 = arith.constant -2 : i64
        %25 = "math.fpowi"(%h_y, %24) : (f32, i64) -> f32
        %26 = stencil.access %f2_t0_blk[0, -1] : !stencil.temp<?x?xf32>
        %27 = arith.mulf %25, %26 : f32
        %h_y_1 = arith.constant 5.000000e-01 : f32
        %28 = arith.constant -2 : i64
        %29 = "math.fpowi"(%h_y_1, %28) : (f32, i64) -> f32
        %30 = stencil.access %f2_t0_blk[0, 1] : !stencil.temp<?x?xf32>
        %31 = arith.mulf %29, %30 : f32
        %32 = arith.constant -2.000000e+00 : f32
        %h_y_2 = arith.constant 5.000000e-01 : f32
        %33 = arith.constant -2 : i64
        %34 = "math.fpowi"(%h_y_2, %33) : (f32, i64) -> f32
        %35 = stencil.access %f2_t0_blk[0, 0] : !stencil.temp<?x?xf32>
        %36 = arith.mulf %32, %34 : f32
        %37 = arith.mulf %36, %35 : f32
        %38 = arith.addf %27, %31 : f32
        %39 = arith.addf %38, %37 : f32
        %40 = arith.mulf %23, %39 : f32
        %dt = arith.constant 1.000000e-01 : f32
        %41 = arith.constant -1 : i64
        %42 = "math.fpowi"(%dt, %41) : (f32, i64) -> f32
        %43 = stencil.access %f2_t0_blk[0, 0] : !stencil.temp<?x?xf32>
        %44 = arith.mulf %42, %43 : f32
        %45 = arith.addf %22, %40 : f32
        %46 = arith.addf %45, %44 : f32
        %dt_1 = arith.constant 1.000000e-01 : f32
        %47 = arith.mulf %46, %dt_1 : f32
        stencil.return %47 : f32
      }
      stencil.store %f2_t1_temp to %f2_t1(<[0, 0], [3, 3]>)  : !stencil.temp<?x?xf32> to !stencil.field<[-2,5]x[-2,5]xf32>
      scf.yield %f2_t1, %f2_t0 : !stencil.field<[-2,5]x[-2,5]xf32>, !stencil.field<[-2,5]x[-2,5]xf32>
    }
    %5 = func.call @timer_end(%0) : (f64) -> f64
    "llvm.store"(%5, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
    func.return
  }
  func.func private @timer_start() -> f64
  func.func private @timer_end(f64) -> f64
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @Kernel(%f2_vec0 : !stencil.field<[-2,5]x[-2,5]xf32>, %f2_vec1 : !stencil.field<[-2,5]x[-2,5]xf32>, %timers : !llvm.ptr) {
// CHECK-NEXT:      %0 = func.call @timer_start() : () -> f64
// CHECK-NEXT:      %time_m = arith.constant 0 : index
// CHECK-NEXT:      %time_M = arith.constant 1 : index
// CHECK-NEXT:      %1 = arith.constant 1 : index
// CHECK-NEXT:      %2 = arith.addi %time_M, %1 : index
// CHECK-NEXT:      %step = arith.constant 1 : index
// CHECK-NEXT:      %3, %4 = scf.for %time = %time_m to %2 step %step iter_args(%f2_t0 = %f2_vec0, %f2_t1 = %f2_vec1) -> (!stencil.field<[-2,5]x[-2,5]xf32>, !stencil.field<[-2,5]x[-2,5]xf32>) {
// CHECK-NEXT:        %f2_t0_temp = stencil.load %f2_t0 : !stencil.field<[-2,5]x[-2,5]xf32> -> !stencil.temp<[-1,4]x[-1,4]xf32>
// CHECK-NEXT:        %f2_t1_temp = stencil.apply(%f2_t0_blk = %f2_t0_temp : !stencil.temp<[-1,4]x[-1,4]xf32>) -> (!stencil.temp<[0,3]x[0,3]xf32>) {
// CHECK-NEXT:          %5 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:          %h_x = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:          %6 = arith.constant -2 : i64
// CHECK-NEXT:          %7 = "math.fpowi"(%h_x, %6) : (f32, i64) -> f32
// CHECK-NEXT:          %8 = stencil.access %f2_t0_blk[-1, 0] : !stencil.temp<[-1,4]x[-1,4]xf32>
// CHECK-NEXT:          %9 = arith.mulf %7, %8 : f32
// CHECK-NEXT:          %h_x_1 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:          %10 = arith.constant -2 : i64
// CHECK-NEXT:          %11 = "math.fpowi"(%h_x_1, %10) : (f32, i64) -> f32
// CHECK-NEXT:          %12 = stencil.access %f2_t0_blk[1, 0] : !stencil.temp<[-1,4]x[-1,4]xf32>
// CHECK-NEXT:          %13 = arith.mulf %11, %12 : f32
// CHECK-NEXT:          %14 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:          %h_x_2 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:          %15 = arith.constant -2 : i64
// CHECK-NEXT:          %16 = "math.fpowi"(%h_x_2, %15) : (f32, i64) -> f32
// CHECK-NEXT:          %17 = stencil.access %f2_t0_blk[0, 0] : !stencil.temp<[-1,4]x[-1,4]xf32>
// CHECK-NEXT:          %18 = arith.mulf %14, %16 : f32
// CHECK-NEXT:          %19 = arith.mulf %18, %17 : f32
// CHECK-NEXT:          %20 = arith.addf %9, %13 : f32
// CHECK-NEXT:          %21 = arith.addf %20, %19 : f32
// CHECK-NEXT:          %22 = arith.mulf %5, %21 : f32
// CHECK-NEXT:          %23 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:          %h_y = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:          %24 = arith.constant -2 : i64
// CHECK-NEXT:          %25 = "math.fpowi"(%h_y, %24) : (f32, i64) -> f32
// CHECK-NEXT:          %26 = stencil.access %f2_t0_blk[0, -1] : !stencil.temp<[-1,4]x[-1,4]xf32>
// CHECK-NEXT:          %27 = arith.mulf %25, %26 : f32
// CHECK-NEXT:          %h_y_1 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:          %28 = arith.constant -2 : i64
// CHECK-NEXT:          %29 = "math.fpowi"(%h_y_1, %28) : (f32, i64) -> f32
// CHECK-NEXT:          %30 = stencil.access %f2_t0_blk[0, 1] : !stencil.temp<[-1,4]x[-1,4]xf32>
// CHECK-NEXT:          %31 = arith.mulf %29, %30 : f32
// CHECK-NEXT:          %32 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:          %h_y_2 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:          %33 = arith.constant -2 : i64
// CHECK-NEXT:          %34 = "math.fpowi"(%h_y_2, %33) : (f32, i64) -> f32
// CHECK-NEXT:          %35 = stencil.access %f2_t0_blk[0, 0] : !stencil.temp<[-1,4]x[-1,4]xf32>
// CHECK-NEXT:          %36 = arith.mulf %32, %34 : f32
// CHECK-NEXT:          %37 = arith.mulf %36, %35 : f32
// CHECK-NEXT:          %38 = arith.addf %27, %31 : f32
// CHECK-NEXT:          %39 = arith.addf %38, %37 : f32
// CHECK-NEXT:          %40 = arith.mulf %23, %39 : f32
// CHECK-NEXT:          %dt = arith.constant 1.000000e-01 : f32
// CHECK-NEXT:          %41 = arith.constant -1 : i64
// CHECK-NEXT:          %42 = "math.fpowi"(%dt, %41) : (f32, i64) -> f32
// CHECK-NEXT:          %43 = stencil.access %f2_t0_blk[0, 0] : !stencil.temp<[-1,4]x[-1,4]xf32>
// CHECK-NEXT:          %44 = arith.mulf %42, %43 : f32
// CHECK-NEXT:          %45 = arith.addf %22, %40 : f32
// CHECK-NEXT:          %46 = arith.addf %45, %44 : f32
// CHECK-NEXT:          %dt_1 = arith.constant 1.000000e-01 : f32
// CHECK-NEXT:          %47 = arith.mulf %46, %dt_1 : f32
// CHECK-NEXT:          stencil.return %47 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        stencil.store %f2_t1_temp to %f2_t1(<[0, 0], [3, 3]>)  : !stencil.temp<[0,3]x[0,3]xf32> to !stencil.field<[-2,5]x[-2,5]xf32>
// CHECK-NEXT:        scf.yield %f2_t1, %f2_t0 : !stencil.field<[-2,5]x[-2,5]xf32>, !stencil.field<[-2,5]x[-2,5]xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %5 = func.call @timer_end(%0) : (f64) -> f64
// CHECK-NEXT:      "llvm.store"(%5, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @timer_start() -> f64
// CHECK-NEXT:    func.func private @timer_end(f64) -> f64
// CHECK-NEXT:  }

