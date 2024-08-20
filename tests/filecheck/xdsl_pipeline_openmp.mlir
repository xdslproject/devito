// RUN: xdsl-opt -p "canonicalize,cse" %s | filecheck %s

builtin.module {
  func.func @xDSLDiffusionOperator(%u_vec0 : memref<158x158x158xf32>, %u_vec1 : memref<158x158x158xf32>, %timers : !llvm.ptr) {
    %0 = func.call @timer_start() : () -> f64
    %time_m = arith.constant 0 : index
    %time_M = arith.constant 250 : index
    %1 = arith.constant 1 : index
    %2 = arith.addi %time_M, %1 : index
    %step = arith.constant 1 : index
    %3, %4 = scf.for %time = %time_m to %2 step %step iter_args(%u_t0 = %u_vec0, %u_t1 = %u_vec1) -> (memref<158x158x158xf32>, memref<158x158x158xf32>) {
      %u_t1_storeview = "memref.subview"(%u_t1) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 150, 150, 150>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<158x158x158xf32>) -> memref<150x150x150xf32, strided<[24964, 158, 1], offset: 100492>>
      %u_t0_loadview = "memref.subview"(%u_t0) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 154, 154, 154>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<158x158x158xf32>) -> memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
      %5 = arith.constant 0 : index
      %6 = arith.constant 0 : index
      %7 = arith.constant 0 : index
      %8 = arith.constant 1 : index
      %9 = arith.constant 1 : index
      %10 = arith.constant 1 : index
      %11 = arith.constant 150 : index
      %12 = arith.constant 150 : index
      %13 = arith.constant 150 : index
      %14 = arith.constant 0 : index
      %15 = arith.constant 64 : index
      %16 = arith.constant 64 : index
      %17 = arith.muli %8, %15 : index
      %18 = arith.muli %9, %16 : index
      "scf.parallel"(%5, %6, %11, %12, %17, %18) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
      ^0(%19 : index, %20 : index):
        %21 = "affine.min"(%15, %11, %19) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
        %22 = "affine.min"(%16, %12, %20) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
        "scf.parallel"(%14, %14, %7, %21, %22, %13, %8, %9, %10) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
        ^1(%23 : index, %24 : index, %25 : index):
          %26 = arith.addi %19, %23 : index
          %27 = arith.addi %20, %24 : index
          %dt = arith.constant 6.717825e-07 : f32
          %28 = arith.constant -1 : i64
          %29 = "math.fpowi"(%dt, %28) : (f32, i64) -> f32
          %30 = memref.load %u_t0_loadview[%26, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %31 = arith.mulf %29, %30 : f32
          %32 = arith.constant 1.333333e+00 : f32
          %h_x = arith.constant 1.342282e-02 : f32
          %33 = arith.constant -2 : i64
          %34 = "math.fpowi"(%h_x, %33) : (f32, i64) -> f32
          %35 = arith.constant -1 : index
          %36 = arith.addi %26, %35 : index
          %37 = memref.load %u_t0_loadview[%36, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %38 = arith.mulf %32, %34 : f32
          %39 = arith.mulf %38, %37 : f32
          %40 = arith.constant 1.333333e+00 : f32
          %h_x_1 = arith.constant 1.342282e-02 : f32
          %41 = arith.constant -2 : i64
          %42 = "math.fpowi"(%h_x_1, %41) : (f32, i64) -> f32
          %43 = arith.constant 1 : index
          %44 = arith.addi %26, %43 : index
          %45 = memref.load %u_t0_loadview[%44, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %46 = arith.mulf %40, %42 : f32
          %47 = arith.mulf %46, %45 : f32
          %48 = arith.constant -2.500000e+00 : f32
          %h_x_2 = arith.constant 1.342282e-02 : f32
          %49 = arith.constant -2 : i64
          %50 = "math.fpowi"(%h_x_2, %49) : (f32, i64) -> f32
          %51 = memref.load %u_t0_loadview[%26, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %52 = arith.mulf %48, %50 : f32
          %53 = arith.mulf %52, %51 : f32
          %54 = arith.constant -8.333333e-02 : f32
          %h_x_3 = arith.constant 1.342282e-02 : f32
          %55 = arith.constant -2 : i64
          %56 = "math.fpowi"(%h_x_3, %55) : (f32, i64) -> f32
          %57 = arith.constant -2 : index
          %58 = arith.addi %26, %57 : index
          %59 = memref.load %u_t0_loadview[%58, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %60 = arith.mulf %54, %56 : f32
          %61 = arith.mulf %60, %59 : f32
          %62 = arith.constant -8.333333e-02 : f32
          %h_x_4 = arith.constant 1.342282e-02 : f32
          %63 = arith.constant -2 : i64
          %64 = "math.fpowi"(%h_x_4, %63) : (f32, i64) -> f32
          %65 = arith.constant 2 : index
          %66 = arith.addi %26, %65 : index
          %67 = memref.load %u_t0_loadview[%66, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %68 = arith.mulf %62, %64 : f32
          %69 = arith.mulf %68, %67 : f32
          %70 = arith.addf %39, %47 : f32
          %71 = arith.addf %70, %53 : f32
          %72 = arith.addf %71, %61 : f32
          %73 = arith.addf %72, %69 : f32
          %74 = arith.constant 1.333333e+00 : f32
          %h_y = arith.constant 1.342282e-02 : f32
          %75 = arith.constant -2 : i64
          %76 = "math.fpowi"(%h_y, %75) : (f32, i64) -> f32
          %77 = arith.constant -1 : index
          %78 = arith.addi %27, %77 : index
          %79 = memref.load %u_t0_loadview[%26, %78, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %80 = arith.mulf %74, %76 : f32
          %81 = arith.mulf %80, %79 : f32
          %82 = arith.constant 1.333333e+00 : f32
          %h_y_1 = arith.constant 1.342282e-02 : f32
          %83 = arith.constant -2 : i64
          %84 = "math.fpowi"(%h_y_1, %83) : (f32, i64) -> f32
          %85 = arith.constant 1 : index
          %86 = arith.addi %27, %85 : index
          %87 = memref.load %u_t0_loadview[%26, %86, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %88 = arith.mulf %82, %84 : f32
          %89 = arith.mulf %88, %87 : f32
          %90 = arith.constant -2.500000e+00 : f32
          %h_y_2 = arith.constant 1.342282e-02 : f32
          %91 = arith.constant -2 : i64
          %92 = "math.fpowi"(%h_y_2, %91) : (f32, i64) -> f32
          %93 = memref.load %u_t0_loadview[%26, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %94 = arith.mulf %90, %92 : f32
          %95 = arith.mulf %94, %93 : f32
          %96 = arith.constant -8.333333e-02 : f32
          %h_y_3 = arith.constant 1.342282e-02 : f32
          %97 = arith.constant -2 : i64
          %98 = "math.fpowi"(%h_y_3, %97) : (f32, i64) -> f32
          %99 = arith.constant -2 : index
          %100 = arith.addi %27, %99 : index
          %101 = memref.load %u_t0_loadview[%26, %100, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %102 = arith.mulf %96, %98 : f32
          %103 = arith.mulf %102, %101 : f32
          %104 = arith.constant -8.333333e-02 : f32
          %h_y_4 = arith.constant 1.342282e-02 : f32
          %105 = arith.constant -2 : i64
          %106 = "math.fpowi"(%h_y_4, %105) : (f32, i64) -> f32
          %107 = arith.constant 2 : index
          %108 = arith.addi %27, %107 : index
          %109 = memref.load %u_t0_loadview[%26, %108, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %110 = arith.mulf %104, %106 : f32
          %111 = arith.mulf %110, %109 : f32
          %112 = arith.addf %81, %89 : f32
          %113 = arith.addf %112, %95 : f32
          %114 = arith.addf %113, %103 : f32
          %115 = arith.addf %114, %111 : f32
          %116 = arith.constant 1.333333e+00 : f32
          %h_z = arith.constant 1.342282e-02 : f32
          %117 = arith.constant -2 : i64
          %118 = "math.fpowi"(%h_z, %117) : (f32, i64) -> f32
          %119 = arith.constant -1 : index
          %120 = arith.addi %25, %119 : index
          %121 = memref.load %u_t0_loadview[%26, %27, %120] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %122 = arith.mulf %116, %118 : f32
          %123 = arith.mulf %122, %121 : f32
          %124 = arith.constant 1.333333e+00 : f32
          %h_z_1 = arith.constant 1.342282e-02 : f32
          %125 = arith.constant -2 : i64
          %126 = "math.fpowi"(%h_z_1, %125) : (f32, i64) -> f32
          %127 = arith.constant 1 : index
          %128 = arith.addi %25, %127 : index
          %129 = memref.load %u_t0_loadview[%26, %27, %128] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %130 = arith.mulf %124, %126 : f32
          %131 = arith.mulf %130, %129 : f32
          %132 = arith.constant -2.500000e+00 : f32
          %h_z_2 = arith.constant 1.342282e-02 : f32
          %133 = arith.constant -2 : i64
          %134 = "math.fpowi"(%h_z_2, %133) : (f32, i64) -> f32
          %135 = memref.load %u_t0_loadview[%26, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %136 = arith.mulf %132, %134 : f32
          %137 = arith.mulf %136, %135 : f32
          %138 = arith.constant -8.333333e-02 : f32
          %h_z_3 = arith.constant 1.342282e-02 : f32
          %139 = arith.constant -2 : i64
          %140 = "math.fpowi"(%h_z_3, %139) : (f32, i64) -> f32
          %141 = arith.constant -2 : index
          %142 = arith.addi %25, %141 : index
          %143 = memref.load %u_t0_loadview[%26, %27, %142] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %144 = arith.mulf %138, %140 : f32
          %145 = arith.mulf %144, %143 : f32
          %146 = arith.constant -8.333333e-02 : f32
          %h_z_4 = arith.constant 1.342282e-02 : f32
          %147 = arith.constant -2 : i64
          %148 = "math.fpowi"(%h_z_4, %147) : (f32, i64) -> f32
          %149 = arith.constant 2 : index
          %150 = arith.addi %25, %149 : index
          %151 = memref.load %u_t0_loadview[%26, %27, %150] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
          %152 = arith.mulf %146, %148 : f32
          %153 = arith.mulf %152, %151 : f32
          %154 = arith.addf %123, %131 : f32
          %155 = arith.addf %154, %137 : f32
          %156 = arith.addf %155, %145 : f32
          %157 = arith.addf %156, %153 : f32
          %158 = arith.addf %73, %115 : f32
          %159 = arith.addf %158, %157 : f32
          %a = arith.constant 9.000000e-01 : f32
          %160 = arith.mulf %159, %a : f32
          %161 = arith.addf %31, %160 : f32
          %dt_1 = arith.constant 6.717825e-07 : f32
          %162 = arith.mulf %161, %dt_1 : f32
          memref.store %162, %u_t1_storeview[%26, %27, %25] : memref<150x150x150xf32, strided<[24964, 158, 1], offset: 100492>>
          scf.yield
        }) : (index, index, index, index, index, index, index, index, index) -> ()
        scf.yield
      }) : (index, index, index, index, index, index) -> ()
      scf.yield %u_t1, %u_t0 : memref<158x158x158xf32>, memref<158x158x158xf32>
    }
    %163 = func.call @timer_end(%0) : (f64) -> f64
    "llvm.store"(%163, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
    func.return
  }
  func.func private @timer_start() -> f64
  func.func private @timer_end(f64) -> f64
}


// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @xDSLDiffusionOperator(%u_vec0 : memref<158x158x158xf32>, %u_vec1 : memref<158x158x158xf32>, %timers : !llvm.ptr) {
// CHECK-NEXT:      %0 = func.call @timer_start() : () -> f64
// CHECK-NEXT:      %time_m = arith.constant 0 : index
// CHECK-NEXT:      %time_M = arith.constant 250 : index
// CHECK-NEXT:      %1 = arith.constant 1 : index
// CHECK-NEXT:      %2 = arith.addi %time_M, %1 : index
// CHECK-NEXT:      %step = arith.constant 1 : index
// CHECK-NEXT:      %3, %4 = scf.for %time = %time_m to %2 step %step iter_args(%u_t0 = %u_vec0, %u_t1 = %u_vec1) -> (memref<158x158x158xf32>, memref<158x158x158xf32>) {
// CHECK-NEXT:        %u_t1_storeview = "memref.subview"(%u_t1) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 150, 150, 150>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<158x158x158xf32>) -> memref<150x150x150xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:        %u_t0_loadview = "memref.subview"(%u_t0) <{"static_offsets" = array<i64: 4, 4, 4>, "static_sizes" = array<i64: 154, 154, 154>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<158x158x158xf32>) -> memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:        %5 = arith.constant 0 : index
// CHECK-NEXT:        %6 = arith.constant 0 : index
// CHECK-NEXT:        %7 = arith.constant 0 : index
// CHECK-NEXT:        %8 = arith.constant 1 : index
// CHECK-NEXT:        %9 = arith.constant 1 : index
// CHECK-NEXT:        %10 = arith.constant 1 : index
// CHECK-NEXT:        %11 = arith.constant 150 : index
// CHECK-NEXT:        %12 = arith.constant 150 : index
// CHECK-NEXT:        %13 = arith.constant 150 : index
// CHECK-NEXT:        %14 = arith.constant 0 : index
// CHECK-NEXT:        %15 = arith.constant 64 : index
// CHECK-NEXT:        %16 = arith.constant 64 : index
// CHECK-NEXT:        %17 = arith.muli %8, %15 : index
// CHECK-NEXT:        %18 = arith.muli %9, %16 : index
// CHECK-NEXT:        "scf.parallel"(%5, %6, %11, %12, %17, %18) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:        ^0(%19 : index, %20 : index):
// CHECK-NEXT:          %21 = "affine.min"(%15, %11, %19) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:          %22 = "affine.min"(%16, %12, %20) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:          "scf.parallel"(%14, %14, %7, %21, %22, %13, %8, %9, %10) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:          ^1(%23 : index, %24 : index, %25 : index):
// CHECK-NEXT:            %26 = arith.addi %19, %23 : index
// CHECK-NEXT:            %27 = arith.addi %20, %24 : index
// CHECK-NEXT:            %dt = arith.constant 6.717825e-07 : f32
// CHECK-NEXT:            %28 = arith.constant -1 : i64
// CHECK-NEXT:            %29 = "math.fpowi"(%dt, %28) : (f32, i64) -> f32
// CHECK-NEXT:            %30 = memref.load %u_t0_loadview[%26, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %31 = arith.mulf %29, %30 : f32
// CHECK-NEXT:            %32 = arith.constant 1.333333e+00 : f32
// CHECK-NEXT:            %h_x = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %33 = arith.constant -2 : i64
// CHECK-NEXT:            %34 = "math.fpowi"(%h_x, %33) : (f32, i64) -> f32
// CHECK-NEXT:            %35 = arith.constant -1 : index
// CHECK-NEXT:            %36 = arith.addi %26, %35 : index
// CHECK-NEXT:            %37 = memref.load %u_t0_loadview[%36, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %38 = arith.mulf %32, %34 : f32
// CHECK-NEXT:            %39 = arith.mulf %38, %37 : f32
// CHECK-NEXT:            %40 = arith.constant 1.333333e+00 : f32
// CHECK-NEXT:            %h_x_1 = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %41 = arith.constant -2 : i64
// CHECK-NEXT:            %42 = "math.fpowi"(%h_x_1, %41) : (f32, i64) -> f32
// CHECK-NEXT:            %43 = arith.constant 1 : index
// CHECK-NEXT:            %44 = arith.addi %26, %43 : index
// CHECK-NEXT:            %45 = memref.load %u_t0_loadview[%44, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %46 = arith.mulf %40, %42 : f32
// CHECK-NEXT:            %47 = arith.mulf %46, %45 : f32
// CHECK-NEXT:            %48 = arith.constant -2.500000e+00 : f32
// CHECK-NEXT:            %h_x_2 = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %49 = arith.constant -2 : i64
// CHECK-NEXT:            %50 = "math.fpowi"(%h_x_2, %49) : (f32, i64) -> f32
// CHECK-NEXT:            %51 = memref.load %u_t0_loadview[%26, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %52 = arith.mulf %48, %50 : f32
// CHECK-NEXT:            %53 = arith.mulf %52, %51 : f32
// CHECK-NEXT:            %54 = arith.constant -8.333333e-02 : f32
// CHECK-NEXT:            %h_x_3 = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %55 = arith.constant -2 : i64
// CHECK-NEXT:            %56 = "math.fpowi"(%h_x_3, %55) : (f32, i64) -> f32
// CHECK-NEXT:            %57 = arith.constant -2 : index
// CHECK-NEXT:            %58 = arith.addi %26, %57 : index
// CHECK-NEXT:            %59 = memref.load %u_t0_loadview[%58, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %60 = arith.mulf %54, %56 : f32
// CHECK-NEXT:            %61 = arith.mulf %60, %59 : f32
// CHECK-NEXT:            %62 = arith.constant -8.333333e-02 : f32
// CHECK-NEXT:            %h_x_4 = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %63 = arith.constant -2 : i64
// CHECK-NEXT:            %64 = "math.fpowi"(%h_x_4, %63) : (f32, i64) -> f32
// CHECK-NEXT:            %65 = arith.constant 2 : index
// CHECK-NEXT:            %66 = arith.addi %26, %65 : index
// CHECK-NEXT:            %67 = memref.load %u_t0_loadview[%66, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %68 = arith.mulf %62, %64 : f32
// CHECK-NEXT:            %69 = arith.mulf %68, %67 : f32
// CHECK-NEXT:            %70 = arith.addf %39, %47 : f32
// CHECK-NEXT:            %71 = arith.addf %70, %53 : f32
// CHECK-NEXT:            %72 = arith.addf %71, %61 : f32
// CHECK-NEXT:            %73 = arith.addf %72, %69 : f32
// CHECK-NEXT:            %74 = arith.constant 1.333333e+00 : f32
// CHECK-NEXT:            %h_y = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %75 = arith.constant -2 : i64
// CHECK-NEXT:            %76 = "math.fpowi"(%h_y, %75) : (f32, i64) -> f32
// CHECK-NEXT:            %77 = arith.constant -1 : index
// CHECK-NEXT:            %78 = arith.addi %27, %77 : index
// CHECK-NEXT:            %79 = memref.load %u_t0_loadview[%26, %78, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %80 = arith.mulf %74, %76 : f32
// CHECK-NEXT:            %81 = arith.mulf %80, %79 : f32
// CHECK-NEXT:            %82 = arith.constant 1.333333e+00 : f32
// CHECK-NEXT:            %h_y_1 = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %83 = arith.constant -2 : i64
// CHECK-NEXT:            %84 = "math.fpowi"(%h_y_1, %83) : (f32, i64) -> f32
// CHECK-NEXT:            %85 = arith.constant 1 : index
// CHECK-NEXT:            %86 = arith.addi %27, %85 : index
// CHECK-NEXT:            %87 = memref.load %u_t0_loadview[%26, %86, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %88 = arith.mulf %82, %84 : f32
// CHECK-NEXT:            %89 = arith.mulf %88, %87 : f32
// CHECK-NEXT:            %90 = arith.constant -2.500000e+00 : f32
// CHECK-NEXT:            %h_y_2 = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %91 = arith.constant -2 : i64
// CHECK-NEXT:            %92 = "math.fpowi"(%h_y_2, %91) : (f32, i64) -> f32
// CHECK-NEXT:            %93 = memref.load %u_t0_loadview[%26, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %94 = arith.mulf %90, %92 : f32
// CHECK-NEXT:            %95 = arith.mulf %94, %93 : f32
// CHECK-NEXT:            %96 = arith.constant -8.333333e-02 : f32
// CHECK-NEXT:            %h_y_3 = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %97 = arith.constant -2 : i64
// CHECK-NEXT:            %98 = "math.fpowi"(%h_y_3, %97) : (f32, i64) -> f32
// CHECK-NEXT:            %99 = arith.constant -2 : index
// CHECK-NEXT:            %100 = arith.addi %27, %99 : index
// CHECK-NEXT:            %101 = memref.load %u_t0_loadview[%26, %100, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %102 = arith.mulf %96, %98 : f32
// CHECK-NEXT:            %103 = arith.mulf %102, %101 : f32
// CHECK-NEXT:            %104 = arith.constant -8.333333e-02 : f32
// CHECK-NEXT:            %h_y_4 = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %105 = arith.constant -2 : i64
// CHECK-NEXT:            %106 = "math.fpowi"(%h_y_4, %105) : (f32, i64) -> f32
// CHECK-NEXT:            %107 = arith.constant 2 : index
// CHECK-NEXT:            %108 = arith.addi %27, %107 : index
// CHECK-NEXT:            %109 = memref.load %u_t0_loadview[%26, %108, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %110 = arith.mulf %104, %106 : f32
// CHECK-NEXT:            %111 = arith.mulf %110, %109 : f32
// CHECK-NEXT:            %112 = arith.addf %81, %89 : f32
// CHECK-NEXT:            %113 = arith.addf %112, %95 : f32
// CHECK-NEXT:            %114 = arith.addf %113, %103 : f32
// CHECK-NEXT:            %115 = arith.addf %114, %111 : f32
// CHECK-NEXT:            %116 = arith.constant 1.333333e+00 : f32
// CHECK-NEXT:            %h_z = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %117 = arith.constant -2 : i64
// CHECK-NEXT:            %118 = "math.fpowi"(%h_z, %117) : (f32, i64) -> f32
// CHECK-NEXT:            %119 = arith.constant -1 : index
// CHECK-NEXT:            %120 = arith.addi %25, %119 : index
// CHECK-NEXT:            %121 = memref.load %u_t0_loadview[%26, %27, %120] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %122 = arith.mulf %116, %118 : f32
// CHECK-NEXT:            %123 = arith.mulf %122, %121 : f32
// CHECK-NEXT:            %124 = arith.constant 1.333333e+00 : f32
// CHECK-NEXT:            %h_z_1 = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %125 = arith.constant -2 : i64
// CHECK-NEXT:            %126 = "math.fpowi"(%h_z_1, %125) : (f32, i64) -> f32
// CHECK-NEXT:            %127 = arith.constant 1 : index
// CHECK-NEXT:            %128 = arith.addi %25, %127 : index
// CHECK-NEXT:            %129 = memref.load %u_t0_loadview[%26, %27, %128] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %130 = arith.mulf %124, %126 : f32
// CHECK-NEXT:            %131 = arith.mulf %130, %129 : f32
// CHECK-NEXT:            %132 = arith.constant -2.500000e+00 : f32
// CHECK-NEXT:            %h_z_2 = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %133 = arith.constant -2 : i64
// CHECK-NEXT:            %134 = "math.fpowi"(%h_z_2, %133) : (f32, i64) -> f32
// CHECK-NEXT:            %135 = memref.load %u_t0_loadview[%26, %27, %25] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %136 = arith.mulf %132, %134 : f32
// CHECK-NEXT:            %137 = arith.mulf %136, %135 : f32
// CHECK-NEXT:            %138 = arith.constant -8.333333e-02 : f32
// CHECK-NEXT:            %h_z_3 = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %139 = arith.constant -2 : i64
// CHECK-NEXT:            %140 = "math.fpowi"(%h_z_3, %139) : (f32, i64) -> f32
// CHECK-NEXT:            %141 = arith.constant -2 : index
// CHECK-NEXT:            %142 = arith.addi %25, %141 : index
// CHECK-NEXT:            %143 = memref.load %u_t0_loadview[%26, %27, %142] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %144 = arith.mulf %138, %140 : f32
// CHECK-NEXT:            %145 = arith.mulf %144, %143 : f32
// CHECK-NEXT:            %146 = arith.constant -8.333333e-02 : f32
// CHECK-NEXT:            %h_z_4 = arith.constant 1.342282e-02 : f32
// CHECK-NEXT:            %147 = arith.constant -2 : i64
// CHECK-NEXT:            %148 = "math.fpowi"(%h_z_4, %147) : (f32, i64) -> f32
// CHECK-NEXT:            %149 = arith.constant 2 : index
// CHECK-NEXT:            %150 = arith.addi %25, %149 : index
// CHECK-NEXT:            %151 = memref.load %u_t0_loadview[%26, %27, %150] : memref<154x154x154xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            %152 = arith.mulf %146, %148 : f32
// CHECK-NEXT:            %153 = arith.mulf %152, %151 : f32
// CHECK-NEXT:            %154 = arith.addf %123, %131 : f32
// CHECK-NEXT:            %155 = arith.addf %154, %137 : f32
// CHECK-NEXT:            %156 = arith.addf %155, %145 : f32
// CHECK-NEXT:            %157 = arith.addf %156, %153 : f32
// CHECK-NEXT:            %158 = arith.addf %73, %115 : f32
// CHECK-NEXT:            %159 = arith.addf %158, %157 : f32
// CHECK-NEXT:            %a = arith.constant 9.000000e-01 : f32
// CHECK-NEXT:            %160 = arith.mulf %159, %a : f32
// CHECK-NEXT:            %161 = arith.addf %31, %160 : f32
// CHECK-NEXT:            %dt_1 = arith.constant 6.717825e-07 : f32
// CHECK-NEXT:            %162 = arith.mulf %161, %dt_1 : f32
// CHECK-NEXT:            memref.store %162, %u_t1_storeview[%26, %27, %25] : memref<150x150x150xf32, strided<[24964, 158, 1], offset: 100492>>
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:        scf.yield %u_t1, %u_t0 : memref<158x158x158xf32>, memref<158x158x158xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %163 = func.call @timer_end(%0) : (f64) -> f64
// CHECK-NEXT:      "llvm.store"(%163, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @timer_start() -> f64
// CHECK-NEXT:    func.func private @timer_end(f64) -> f64
// CHECK-NEXT:  }