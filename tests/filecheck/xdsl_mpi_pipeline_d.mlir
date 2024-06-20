// RUN: xdsl-opt -p "distribute-stencil{strategy=3d-grid slices=2,1,1 restrict_domain=false},canonicalize-dmp,convert-stencil-to-ll-mlir,scf-parallel-loop-tiling{parallel-loop-tile-sizes=64,64,0}" %s | filecheck %s

builtin.module {
  func.func @Kernel(%u_vec0 : !stencil.field<[-2,53]x[-2,103]x[-2,103]xf32>, %u_vec1 : !stencil.field<[-2,53]x[-2,103]x[-2,103]xf32>, %u_vec2 : !stencil.field<[-2,53]x[-2,103]x[-2,103]xf32>, %timers : !llvm.ptr) {
    %0 = func.call @timer_start() : () -> f64
    %time_m = arith.constant 1 : index
    %time_M = arith.constant 20 : index
    %1 = arith.constant 1 : index
    %2 = arith.addi %time_M, %1 : index
    %step = arith.constant 1 : index
    %3, %4, %5 = scf.for %time = %time_m to %2 step %step iter_args(%u_t0 = %u_vec0, %u_t1 = %u_vec1, %u_t2 = %u_vec2) -> (!stencil.field<[-2,53]x[-2,103]x[-2,103]xf32>, !stencil.field<[-2,53]x[-2,103]x[-2,103]xf32>, !stencil.field<[-2,53]x[-2,103]x[-2,103]xf32>) {
      %u_t0_temp = stencil.load %u_t0 : !stencil.field<[-2,53]x[-2,103]x[-2,103]xf32> -> !stencil.temp<?x?x?xf32>
      %u_t2_temp = stencil.load %u_t2 : !stencil.field<[-2,53]x[-2,103]x[-2,103]xf32> -> !stencil.temp<?x?x?xf32>
      %u_t1_temp = stencil.apply(%u_t0_blk = %u_t0_temp : !stencil.temp<?x?x?xf32>, %u_t2_blk = %u_t2_temp : !stencil.temp<?x?x?xf32>) -> (!stencil.temp<?x?x?xf32>) {
        %dt = arith.constant 1.000000e-04 : f32
        %6 = arith.constant 2 : i64
        %7 = "math.fpowi"(%dt, %6) : (f32, i64) -> f32
        %8 = arith.constant -1 : i64
        %dt_1 = arith.constant 1.000000e-04 : f32
        %9 = arith.constant -2 : i64
        %10 = "math.fpowi"(%dt_1, %9) : (f32, i64) -> f32
        %11 = stencil.access %u_t2_blk[0, 0, 0] : !stencil.temp<?x?x?xf32>
        %12 = arith.mulf %10, %11 : f32
        %13 = arith.constant -2.000000e+00 : f32
        %dt_2 = arith.constant 1.000000e-04 : f32
        %14 = arith.constant -2 : i64
        %15 = "math.fpowi"(%dt_2, %14) : (f32, i64) -> f32
        %16 = stencil.access %u_t0_blk[0, 0, 0] : !stencil.temp<?x?x?xf32>
        %17 = arith.mulf %13, %15 : f32
        %18 = arith.mulf %17, %16 : f32
        %19 = arith.addf %12, %18 : f32
        %20 = arith.sitofp %8 : i64 to f32
        %21 = arith.mulf %20, %19 : f32
        %h_x = arith.constant 1.000000e-02 : f32
        %22 = arith.constant -2 : i64
        %23 = "math.fpowi"(%h_x, %22) : (f32, i64) -> f32
        %24 = stencil.access %u_t0_blk[-1, 0, 0] : !stencil.temp<?x?x?xf32>
        %25 = arith.mulf %23, %24 : f32
        %h_x_1 = arith.constant 1.000000e-02 : f32
        %26 = arith.constant -2 : i64
        %27 = "math.fpowi"(%h_x_1, %26) : (f32, i64) -> f32
        %28 = stencil.access %u_t0_blk[1, 0, 0] : !stencil.temp<?x?x?xf32>
        %29 = arith.mulf %27, %28 : f32
        %30 = arith.constant -2.000000e+00 : f32
        %h_x_2 = arith.constant 1.000000e-02 : f32
        %31 = arith.constant -2 : i64
        %32 = "math.fpowi"(%h_x_2, %31) : (f32, i64) -> f32
        %33 = stencil.access %u_t0_blk[0, 0, 0] : !stencil.temp<?x?x?xf32>
        %34 = arith.mulf %30, %32 : f32
        %35 = arith.mulf %34, %33 : f32
        %36 = arith.addf %25, %29 : f32
        %37 = arith.addf %36, %35 : f32
        %h_y = arith.constant 1.000000e-02 : f32
        %38 = arith.constant -2 : i64
        %39 = "math.fpowi"(%h_y, %38) : (f32, i64) -> f32
        %40 = stencil.access %u_t0_blk[0, -1, 0] : !stencil.temp<?x?x?xf32>
        %41 = arith.mulf %39, %40 : f32
        %h_y_1 = arith.constant 1.000000e-02 : f32
        %42 = arith.constant -2 : i64
        %43 = "math.fpowi"(%h_y_1, %42) : (f32, i64) -> f32
        %44 = stencil.access %u_t0_blk[0, 1, 0] : !stencil.temp<?x?x?xf32>
        %45 = arith.mulf %43, %44 : f32
        %46 = arith.constant -2.000000e+00 : f32
        %h_y_2 = arith.constant 1.000000e-02 : f32
        %47 = arith.constant -2 : i64
        %48 = "math.fpowi"(%h_y_2, %47) : (f32, i64) -> f32
        %49 = stencil.access %u_t0_blk[0, 0, 0] : !stencil.temp<?x?x?xf32>
        %50 = arith.mulf %46, %48 : f32
        %51 = arith.mulf %50, %49 : f32
        %52 = arith.addf %41, %45 : f32
        %53 = arith.addf %52, %51 : f32
        %h_z = arith.constant 1.000000e-02 : f32
        %54 = arith.constant -2 : i64
        %55 = "math.fpowi"(%h_z, %54) : (f32, i64) -> f32
        %56 = stencil.access %u_t0_blk[0, 0, -1] : !stencil.temp<?x?x?xf32>
        %57 = arith.mulf %55, %56 : f32
        %h_z_1 = arith.constant 1.000000e-02 : f32
        %58 = arith.constant -2 : i64
        %59 = "math.fpowi"(%h_z_1, %58) : (f32, i64) -> f32
        %60 = stencil.access %u_t0_blk[0, 0, 1] : !stencil.temp<?x?x?xf32>
        %61 = arith.mulf %59, %60 : f32
        %62 = arith.constant -2.000000e+00 : f32
        %h_z_2 = arith.constant 1.000000e-02 : f32
        %63 = arith.constant -2 : i64
        %64 = "math.fpowi"(%h_z_2, %63) : (f32, i64) -> f32
        %65 = stencil.access %u_t0_blk[0, 0, 0] : !stencil.temp<?x?x?xf32>
        %66 = arith.mulf %62, %64 : f32
        %67 = arith.mulf %66, %65 : f32
        %68 = arith.addf %57, %61 : f32
        %69 = arith.addf %68, %67 : f32
        %70 = arith.addf %21, %37 : f32
        %71 = arith.addf %70, %53 : f32
        %72 = arith.addf %71, %69 : f32
        %73 = arith.mulf %7, %72 : f32
        stencil.return %73 : f32
      }
      %u_t1_temp_1 = stencil.store %u_t1_temp to %u_t1 ([0, 0, 0] : [51, 101, 101]) : !stencil.temp<?x?x?xf32> to !stencil.field<[-2,53]x[-2,103]x[-2,103]xf32> with_halo : !stencil.temp<?x?x?xf32>
      scf.yield %u_t1, %u_t2, %u_t0 : !stencil.field<[-2,53]x[-2,103]x[-2,103]xf32>, !stencil.field<[-2,53]x[-2,103]x[-2,103]xf32>, !stencil.field<[-2,53]x[-2,103]x[-2,103]xf32>
    }
    %6 = func.call @timer_end(%0) : (f64) -> f64
    "llvm.store"(%6, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
    func.return
  }
  func.func private @timer_start() -> f64
  func.func private @timer_end(f64) -> f64
}


// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @Kernel(%u_vec0 : memref<55x105x105xf32>, %u_vec1 : memref<55x105x105xf32>, %u_vec2 : memref<55x105x105xf32>, %timers : !llvm.ptr) {
// CHECK-NEXT:      %0 = func.call @timer_start() : () -> f64
// CHECK-NEXT:      %time_m = arith.constant 1 : index
// CHECK-NEXT:      %time_M = arith.constant 20 : index
// CHECK-NEXT:      %1 = arith.constant 1 : index
// CHECK-NEXT:      %2 = arith.addi %time_M, %1 : index
// CHECK-NEXT:      %step = arith.constant 1 : index
// CHECK-NEXT:      %3, %4, %5 = scf.for %time = %time_m to %2 step %step iter_args(%u_t0 = %u_vec0, %u_t1 = %u_vec1, %u_t2 = %u_vec2) -> (memref<55x105x105xf32>, memref<55x105x105xf32>, memref<55x105x105xf32>) {
// CHECK-NEXT:        %u_t1_storeview = "memref.subview"(%u_t1) <{"static_offsets" = array<i64: 2, 2, 2>, "static_sizes" = array<i64: 51, 101, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<55x105x105xf32>) -> memref<51x101x101xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        %u_t0_loadview = "memref.subview"(%u_t0) <{"static_offsets" = array<i64: 2, 2, 2>, "static_sizes" = array<i64: 53, 103, 103>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<55x105x105xf32>) -> memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        "dmp.swap"(%u_t0_loadview) {"topo" = #dmp.topo<2x1x1>, "swaps" = [#dmp.exchange<at [51, 0, 0] size [1, 101, 101] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 101, 101] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 101, 0] size [51, 1, 101] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [51, 1, 101] source offset [0, 1, 0] to [0, -1, 0]>, #dmp.exchange<at [0, 0, 101] size [51, 101, 1] source offset [0, 0, -1] to [0, 0, 1]>, #dmp.exchange<at [0, 0, -1] size [51, 101, 1] source offset [0, 0, 1] to [0, 0, -1]>]} : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> ()
// CHECK-NEXT:        %u_t2_loadview = "memref.subview"(%u_t2) <{"static_offsets" = array<i64: 2, 2, 2>, "static_sizes" = array<i64: 51, 101, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<55x105x105xf32>) -> memref<51x101x101xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        %6 = arith.constant 0 : index
// CHECK-NEXT:        %7 = arith.constant 0 : index
// CHECK-NEXT:        %8 = arith.constant 0 : index
// CHECK-NEXT:        %9 = arith.constant 1 : index
// CHECK-NEXT:        %10 = arith.constant 1 : index
// CHECK-NEXT:        %11 = arith.constant 1 : index
// CHECK-NEXT:        %12 = arith.constant 51 : index
// CHECK-NEXT:        %13 = arith.constant 101 : index
// CHECK-NEXT:        %14 = arith.constant 101 : index
// CHECK-NEXT:        %15 = arith.constant 0 : index
// CHECK-NEXT:        %16 = arith.constant 64 : index
// CHECK-NEXT:        %17 = arith.constant 64 : index
// CHECK-NEXT:        %18 = arith.muli %9, %16 : index
// CHECK-NEXT:        %19 = arith.muli %10, %17 : index
// CHECK-NEXT:        "scf.parallel"(%6, %7, %12, %13, %18, %19) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:        ^0(%20 : index, %21 : index):
// CHECK-NEXT:          %22 = "affine.min"(%16, %12, %20) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:          %23 = "affine.min"(%17, %13, %21) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:          "scf.parallel"(%15, %15, %8, %22, %23, %14, %9, %10, %11) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:          ^1(%24 : index, %25 : index, %26 : index):
// CHECK-NEXT:            %27 = arith.addi %20, %24 : index
// CHECK-NEXT:            %28 = arith.addi %21, %25 : index
// CHECK-NEXT:            %dt = arith.constant 1.000000e-04 : f32
// CHECK-NEXT:            %29 = arith.constant 2 : i64
// CHECK-NEXT:            %30 = "math.fpowi"(%dt, %29) : (f32, i64) -> f32
// CHECK-NEXT:            %31 = arith.constant -1 : i64
// CHECK-NEXT:            %dt_1 = arith.constant 1.000000e-04 : f32
// CHECK-NEXT:            %32 = arith.constant -2 : i64
// CHECK-NEXT:            %33 = "math.fpowi"(%dt_1, %32) : (f32, i64) -> f32
// CHECK-NEXT:            %34 = memref.load %u_t2_loadview[%27, %28, %26] : memref<51x101x101xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %35 = arith.mulf %33, %34 : f32
// CHECK-NEXT:            %36 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:            %dt_2 = arith.constant 1.000000e-04 : f32
// CHECK-NEXT:            %37 = arith.constant -2 : i64
// CHECK-NEXT:            %38 = "math.fpowi"(%dt_2, %37) : (f32, i64) -> f32
// CHECK-NEXT:            %39 = memref.load %u_t0_loadview[%27, %28, %26] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %40 = arith.mulf %36, %38 : f32
// CHECK-NEXT:            %41 = arith.mulf %40, %39 : f32
// CHECK-NEXT:            %42 = arith.addf %35, %41 : f32
// CHECK-NEXT:            %43 = arith.sitofp %31 : i64 to f32
// CHECK-NEXT:            %44 = arith.mulf %43, %42 : f32
// CHECK-NEXT:            %h_x = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %45 = arith.constant -2 : i64
// CHECK-NEXT:            %46 = "math.fpowi"(%h_x, %45) : (f32, i64) -> f32
// CHECK-NEXT:            %47 = arith.constant -1 : index
// CHECK-NEXT:            %48 = arith.addi %27, %47 : index
// CHECK-NEXT:            %49 = memref.load %u_t0_loadview[%48, %28, %26] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %50 = arith.mulf %46, %49 : f32
// CHECK-NEXT:            %h_x_1 = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %51 = arith.constant -2 : i64
// CHECK-NEXT:            %52 = "math.fpowi"(%h_x_1, %51) : (f32, i64) -> f32
// CHECK-NEXT:            %53 = arith.constant 1 : index
// CHECK-NEXT:            %54 = arith.addi %27, %53 : index
// CHECK-NEXT:            %55 = memref.load %u_t0_loadview[%54, %28, %26] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %56 = arith.mulf %52, %55 : f32
// CHECK-NEXT:            %57 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:            %h_x_2 = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %58 = arith.constant -2 : i64
// CHECK-NEXT:            %59 = "math.fpowi"(%h_x_2, %58) : (f32, i64) -> f32
// CHECK-NEXT:            %60 = memref.load %u_t0_loadview[%27, %28, %26] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %61 = arith.mulf %57, %59 : f32
// CHECK-NEXT:            %62 = arith.mulf %61, %60 : f32
// CHECK-NEXT:            %63 = arith.addf %50, %56 : f32
// CHECK-NEXT:            %64 = arith.addf %63, %62 : f32
// CHECK-NEXT:            %h_y = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %65 = arith.constant -2 : i64
// CHECK-NEXT:            %66 = "math.fpowi"(%h_y, %65) : (f32, i64) -> f32
// CHECK-NEXT:            %67 = arith.constant -1 : index
// CHECK-NEXT:            %68 = arith.addi %28, %67 : index
// CHECK-NEXT:            %69 = memref.load %u_t0_loadview[%27, %68, %26] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %70 = arith.mulf %66, %69 : f32
// CHECK-NEXT:            %h_y_1 = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %71 = arith.constant -2 : i64
// CHECK-NEXT:            %72 = "math.fpowi"(%h_y_1, %71) : (f32, i64) -> f32
// CHECK-NEXT:            %73 = arith.constant 1 : index
// CHECK-NEXT:            %74 = arith.addi %28, %73 : index
// CHECK-NEXT:            %75 = memref.load %u_t0_loadview[%27, %74, %26] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %76 = arith.mulf %72, %75 : f32
// CHECK-NEXT:            %77 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:            %h_y_2 = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %78 = arith.constant -2 : i64
// CHECK-NEXT:            %79 = "math.fpowi"(%h_y_2, %78) : (f32, i64) -> f32
// CHECK-NEXT:            %80 = memref.load %u_t0_loadview[%27, %28, %26] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %81 = arith.mulf %77, %79 : f32
// CHECK-NEXT:            %82 = arith.mulf %81, %80 : f32
// CHECK-NEXT:            %83 = arith.addf %70, %76 : f32
// CHECK-NEXT:            %84 = arith.addf %83, %82 : f32
// CHECK-NEXT:            %h_z = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %85 = arith.constant -2 : i64
// CHECK-NEXT:            %86 = "math.fpowi"(%h_z, %85) : (f32, i64) -> f32
// CHECK-NEXT:            %87 = arith.constant -1 : index
// CHECK-NEXT:            %88 = arith.addi %26, %87 : index
// CHECK-NEXT:            %89 = memref.load %u_t0_loadview[%27, %28, %88] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %90 = arith.mulf %86, %89 : f32
// CHECK-NEXT:            %h_z_1 = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %91 = arith.constant -2 : i64
// CHECK-NEXT:            %92 = "math.fpowi"(%h_z_1, %91) : (f32, i64) -> f32
// CHECK-NEXT:            %93 = arith.constant 1 : index
// CHECK-NEXT:            %94 = arith.addi %26, %93 : index
// CHECK-NEXT:            %95 = memref.load %u_t0_loadview[%27, %28, %94] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %96 = arith.mulf %92, %95 : f32
// CHECK-NEXT:            %97 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:            %h_z_2 = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %98 = arith.constant -2 : i64
// CHECK-NEXT:            %99 = "math.fpowi"(%h_z_2, %98) : (f32, i64) -> f32
// CHECK-NEXT:            %100 = memref.load %u_t0_loadview[%27, %28, %26] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %101 = arith.mulf %97, %99 : f32
// CHECK-NEXT:            %102 = arith.mulf %101, %100 : f32
// CHECK-NEXT:            %103 = arith.addf %90, %96 : f32
// CHECK-NEXT:            %104 = arith.addf %103, %102 : f32
// CHECK-NEXT:            %105 = arith.addf %44, %64 : f32
// CHECK-NEXT:            %106 = arith.addf %105, %84 : f32
// CHECK-NEXT:            %107 = arith.addf %106, %104 : f32
// CHECK-NEXT:            %108 = arith.mulf %30, %107 : f32
// CHECK-NEXT:            memref.store %108, %u_t1_storeview[%27, %28, %26] : memref<51x101x101xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:        %u_t1_temp = "memref.subview"(%u_t1) <{"static_offsets" = array<i64: 2, 2, 2>, "static_sizes" = array<i64: 51, 101, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<55x105x105xf32>) -> memref<51x101x101xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        scf.yield %u_t1, %u_t2, %u_t0 : memref<55x105x105xf32>, memref<55x105x105xf32>, memref<55x105x105xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %109 = func.call @timer_end(%0) : (f64) -> f64
// CHECK-NEXT:      "llvm.store"(%109, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @timer_start() -> f64
// CHECK-NEXT:    func.func private @timer_end(f64) -> f64
// CHECK-NEXT:  }