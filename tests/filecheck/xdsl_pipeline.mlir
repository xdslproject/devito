// RUN: xdsl-opt -p stencil-shape-inference,convert-stencil-to-ll-mlir,scf-parallel-loop-tiling{parallel-loop-tile-sizes=64,0},printf-to-llvm,canonicalize %s | filecheck %s

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
      %f2_t1_temp_1 = stencil.store %f2_t1_temp to %f2_t1 ([0, 0] : [3, 3]) : !stencil.temp<?x?xf32> to !stencil.field<[-2,5]x[-2,5]xf32> with_halo : !stencil.temp<?x?xf32>
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
// CHECK-NEXT:    func.func @Kernel(%f2_vec0 : memref<7x7xf32>, %f2_vec1 : memref<7x7xf32>, %timers : !llvm.ptr) {
// CHECK-NEXT:      %0 = func.call @timer_start() : () -> f64
// CHECK-NEXT:      %time_m = arith.constant 0 : index
// CHECK-NEXT:      %time_M = arith.constant 1 : index
// CHECK-NEXT:      %1 = arith.constant 1 : index
// CHECK-NEXT:      %2 = arith.addi %time_M, %1 : index
// CHECK-NEXT:      %step = arith.constant 1 : index
// CHECK-NEXT:      %3, %4 = scf.for %time = %time_m to %2 step %step iter_args(%f2_t0 = %f2_vec0, %f2_t1 = %f2_vec1) -> (memref<7x7xf32>, memref<7x7xf32>) {
// CHECK-NEXT:        %f2_t1_storeview = "memref.subview"(%f2_t1) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 3, 3>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<7x7xf32>) -> memref<3x3xf32, strided<[7, 1], offset: 16>>
// CHECK-NEXT:        %f2_t0_loadview = "memref.subview"(%f2_t0) <{"static_offsets" = array<i64: 2, 2>, "static_sizes" = array<i64: 5, 5>, "static_strides" = array<i64: 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<7x7xf32>) -> memref<5x5xf32, strided<[7, 1], offset: 16>>
// CHECK-NEXT:        %5 = arith.constant 0 : index
// CHECK-NEXT:        %6 = arith.constant 0 : index
// CHECK-NEXT:        %7 = arith.constant 1 : index
// CHECK-NEXT:        %8 = arith.constant 1 : index
// CHECK-NEXT:        %9 = arith.constant 3 : index
// CHECK-NEXT:        %10 = arith.constant 3 : index
// CHECK-NEXT:        %11 = arith.constant 0 : index
// CHECK-NEXT:        %12 = arith.constant 64 : index
// CHECK-NEXT:        %13 = arith.muli %7, %12 : index
// CHECK-NEXT:        "scf.parallel"(%5, %9, %13) <{"operandSegmentSizes" = array<i32: 1, 1, 1, 0>}> ({
// CHECK-NEXT:        ^0(%14 : index):
// CHECK-NEXT:          %15 = "affine.min"(%12, %9, %14) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:          "scf.parallel"(%11, %6, %15, %10, %7, %8) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:          ^1(%16 : index, %17 : index):
// CHECK-NEXT:            %18 = arith.addi %14, %16 : index
// CHECK-NEXT:            %19 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:            %h_x = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:            %20 = arith.constant -2 : i64
// CHECK-NEXT:            %21 = "math.fpowi"(%h_x, %20) : (f32, i64) -> f32
// CHECK-NEXT:            %22 = arith.constant -1 : index
// CHECK-NEXT:            %23 = arith.addi %18, %22 : index
// CHECK-NEXT:            %24 = memref.load %f2_t0_loadview[%23, %17] : memref<5x5xf32, strided<[7, 1], offset: 16>>
// CHECK-NEXT:            %25 = arith.mulf %21, %24 : f32
// CHECK-NEXT:            %h_x_1 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:            %26 = arith.constant -2 : i64
// CHECK-NEXT:            %27 = "math.fpowi"(%h_x_1, %26) : (f32, i64) -> f32
// CHECK-NEXT:            %28 = arith.constant 1 : index
// CHECK-NEXT:            %29 = arith.addi %18, %28 : index
// CHECK-NEXT:            %30 = memref.load %f2_t0_loadview[%29, %17] : memref<5x5xf32, strided<[7, 1], offset: 16>>
// CHECK-NEXT:            %31 = arith.mulf %27, %30 : f32
// CHECK-NEXT:            %32 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:            %h_x_2 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:            %33 = arith.constant -2 : i64
// CHECK-NEXT:            %34 = "math.fpowi"(%h_x_2, %33) : (f32, i64) -> f32
// CHECK-NEXT:            %35 = memref.load %f2_t0_loadview[%18, %17] : memref<5x5xf32, strided<[7, 1], offset: 16>>
// CHECK-NEXT:            %36 = arith.mulf %32, %34 : f32
// CHECK-NEXT:            %37 = arith.mulf %36, %35 : f32
// CHECK-NEXT:            %38 = arith.addf %25, %31 : f32
// CHECK-NEXT:            %39 = arith.addf %38, %37 : f32
// CHECK-NEXT:            %40 = arith.mulf %19, %39 : f32
// CHECK-NEXT:            %41 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:            %h_y = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:            %42 = arith.constant -2 : i64
// CHECK-NEXT:            %43 = "math.fpowi"(%h_y, %42) : (f32, i64) -> f32
// CHECK-NEXT:            %44 = arith.constant -1 : index
// CHECK-NEXT:            %45 = arith.addi %17, %44 : index
// CHECK-NEXT:            %46 = memref.load %f2_t0_loadview[%18, %45] : memref<5x5xf32, strided<[7, 1], offset: 16>>
// CHECK-NEXT:            %47 = arith.mulf %43, %46 : f32
// CHECK-NEXT:            %h_y_1 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:            %48 = arith.constant -2 : i64
// CHECK-NEXT:            %49 = "math.fpowi"(%h_y_1, %48) : (f32, i64) -> f32
// CHECK-NEXT:            %50 = arith.constant 1 : index
// CHECK-NEXT:            %51 = arith.addi %17, %50 : index
// CHECK-NEXT:            %52 = memref.load %f2_t0_loadview[%18, %51] : memref<5x5xf32, strided<[7, 1], offset: 16>>
// CHECK-NEXT:            %53 = arith.mulf %49, %52 : f32
// CHECK-NEXT:            %54 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:            %h_y_2 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:            %55 = arith.constant -2 : i64
// CHECK-NEXT:            %56 = "math.fpowi"(%h_y_2, %55) : (f32, i64) -> f32
// CHECK-NEXT:            %57 = memref.load %f2_t0_loadview[%18, %17] : memref<5x5xf32, strided<[7, 1], offset: 16>>
// CHECK-NEXT:            %58 = arith.mulf %54, %56 : f32
// CHECK-NEXT:            %59 = arith.mulf %58, %57 : f32
// CHECK-NEXT:            %60 = arith.addf %47, %53 : f32
// CHECK-NEXT:            %61 = arith.addf %60, %59 : f32
// CHECK-NEXT:            %62 = arith.mulf %41, %61 : f32
// CHECK-NEXT:            %dt = arith.constant 1.000000e-01 : f32
// CHECK-NEXT:            %63 = arith.constant -1 : i64
// CHECK-NEXT:            %64 = "math.fpowi"(%dt, %63) : (f32, i64) -> f32
// CHECK-NEXT:            %65 = memref.load %f2_t0_loadview[%18, %17] : memref<5x5xf32, strided<[7, 1], offset: 16>>
// CHECK-NEXT:            %66 = arith.mulf %64, %65 : f32
// CHECK-NEXT:            %67 = arith.addf %40, %62 : f32
// CHECK-NEXT:            %68 = arith.addf %67, %66 : f32
// CHECK-NEXT:            %dt_1 = arith.constant 1.000000e-01 : f32
// CHECK-NEXT:            %69 = arith.mulf %68, %dt_1 : f32
// CHECK-NEXT:            memref.store %69, %f2_t1_storeview[%18, %17] : memref<3x3xf32, strided<[7, 1], offset: 16>>
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (index, index, index) -> ()
// CHECK-NEXT:        scf.yield %f2_t1, %f2_t0 : memref<7x7xf32>, memref<7x7xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %70 = func.call @timer_end(%0) : (f64) -> f64
// CHECK-NEXT:      "llvm.store"(%70, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @timer_start() -> f64
// CHECK-NEXT:    func.func private @timer_end(f64) -> f64
// CHECK-NEXT:  }
// CHECK-NEXT:  
