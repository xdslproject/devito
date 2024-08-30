// RUN: xdsl-opt -p "canonicalize,cse,distribute-stencil{strategy=3d-grid slices=2,1,1 restrict_domain=false},shape-inference,canonicalize-dmp,stencil-bufferize,dmp-to-mpi{mpi_init=false},convert-stencil-to-ll-mlir,scf-parallel-loop-tiling{parallel-loop-tile-sizes=64,64,0},canonicalize,cse" %s | filecheck %s

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
      stencil.store %u_t1_temp to %u_t1(<[0, 0, 0], [51, 101, 101]>)  : !stencil.temp<?x?x?xf32> to !stencil.field<[-2,53]x[-2,103]x[-2,103]xf32>
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
// CHECK-NEXT:      %1 = arith.addi %time_M, %time_m : index
// CHECK-NEXT:      %2 = arith.constant 12 : i32
// CHECK-NEXT:      %3 = "mpi.allocate"(%2) {"dtype" = !mpi.request} : (i32) -> !mpi.vector<!mpi.request>
// CHECK-NEXT:      %4 = "mpi.comm.rank"() : () -> i32
// CHECK-NEXT:      %send_buff_ex0 = memref.alloc() {"alignment" = 64 : i64} : memref<101x101xf32>
// CHECK-NEXT:      %send_buff_ex0_ptr, %5, %6 = "mpi.unwrap_memref"(%send_buff_ex0) : (memref<101x101xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT:      %recv_buff_ex0 = memref.alloc() {"alignment" = 64 : i64} : memref<101x101xf32>
// CHECK-NEXT:      %recv_buff_ex0_ptr, %7, %8 = "mpi.unwrap_memref"(%recv_buff_ex0) : (memref<101x101xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT:      %send_buff_ex1 = memref.alloc() {"alignment" = 64 : i64} : memref<101x101xf32>
// CHECK-NEXT:      %send_buff_ex1_ptr, %9, %10 = "mpi.unwrap_memref"(%send_buff_ex1) : (memref<101x101xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT:      %recv_buff_ex1 = memref.alloc() {"alignment" = 64 : i64} : memref<101x101xf32>
// CHECK-NEXT:      %recv_buff_ex1_ptr, %11, %12 = "mpi.unwrap_memref"(%recv_buff_ex1) : (memref<101x101xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT:      %send_buff_ex2 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %send_buff_ex2_ptr, %13, %14 = "mpi.unwrap_memref"(%send_buff_ex2) : (memref<51x101xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT:      %recv_buff_ex2 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %recv_buff_ex2_ptr, %15, %16 = "mpi.unwrap_memref"(%recv_buff_ex2) : (memref<51x101xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT:      %send_buff_ex3 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %send_buff_ex3_ptr, %17, %18 = "mpi.unwrap_memref"(%send_buff_ex3) : (memref<51x101xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT:      %recv_buff_ex3 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %recv_buff_ex3_ptr, %19, %20 = "mpi.unwrap_memref"(%recv_buff_ex3) : (memref<51x101xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT:      %send_buff_ex4 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %send_buff_ex4_ptr, %21, %22 = "mpi.unwrap_memref"(%send_buff_ex4) : (memref<51x101xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT:      %recv_buff_ex4 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %recv_buff_ex4_ptr, %23, %24 = "mpi.unwrap_memref"(%recv_buff_ex4) : (memref<51x101xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT:      %send_buff_ex5 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %send_buff_ex5_ptr, %25, %26 = "mpi.unwrap_memref"(%send_buff_ex5) : (memref<51x101xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT:      %recv_buff_ex5 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %recv_buff_ex5_ptr, %27, %28 = "mpi.unwrap_memref"(%recv_buff_ex5) : (memref<51x101xf32>) -> (!llvm.ptr, i32, !mpi.datatype)
// CHECK-NEXT:      %29, %30, %31 = scf.for %time = %time_m to %1 step %time_m iter_args(%u_t0 = %u_vec0, %u_t1 = %u_vec1, %u_t2 = %u_vec2) -> (memref<55x105x105xf32>, memref<55x105x105xf32>, memref<55x105x105xf32>) {
// CHECK-NEXT:        %32 = arith.constant 0 : i32
// CHECK-NEXT:        %33 = arith.constant 1 : i32
// CHECK-NEXT:        %34 = arith.divui %4, %33 : i32
// CHECK-NEXT:        %35 = arith.remui %4, %33 : i32
// CHECK-NEXT:        %36 = arith.divui %35, %33 : i32
// CHECK-NEXT:        %37 = arith.remui %35, %33 : i32
// CHECK-NEXT:        %38 = arith.divui %37, %33 : i32
// CHECK-NEXT:        %39 = arith.remui %37, %33 : i32
// CHECK-NEXT:        %40 = arith.addi %34, %33 : i32
// CHECK-NEXT:        %41 = arith.constant 2 : i32
// CHECK-NEXT:        %42 = arith.cmpi slt, %40, %41 : i32
// CHECK-NEXT:        %43 = arith.constant true
// CHECK-NEXT:        %44 = arith.andi %42, %43 : i1
// CHECK-NEXT:        %45 = arith.andi %44, %43 : i1
// CHECK-NEXT:        %46 = arith.muli %33, %40 : i32
// CHECK-NEXT:        %47 = arith.addi %38, %46 : i32
// CHECK-NEXT:        %48 = arith.muli %33, %36 : i32
// CHECK-NEXT:        %49 = arith.addi %47, %48 : i32
// CHECK-NEXT:        %50 = arith.constant 6 : i32
// CHECK-NEXT:        %51 = "mpi.vector_get"(%3, %32) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
// CHECK-NEXT:        %52 = "mpi.vector_get"(%3, %50) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
// CHECK-NEXT:        "scf.if"(%45) ({
// CHECK-NEXT:          %53 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %54 = memref.subview %53[52, 2, 2] [1, 101, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<101x101xf32, strided<[105, 1], offset: 573512>>
// CHECK-NEXT:          "memref.copy"(%54, %send_buff_ex0) : (memref<101x101xf32, strided<[105, 1], offset: 573512>>, memref<101x101xf32>) -> ()
// CHECK-NEXT:          "mpi.isend"(%send_buff_ex0_ptr, %5, %6, %49, %32, %51) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
// CHECK-NEXT:          "mpi.irecv"(%recv_buff_ex0_ptr, %7, %8, %49, %32, %52) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          "mpi.request_null"(%51) : (!mpi.request) -> ()
// CHECK-NEXT:          "mpi.request_null"(%52) : (!mpi.request) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %55 = arith.remui %4, %33 : i32
// CHECK-NEXT:        %56 = arith.divui %55, %33 : i32
// CHECK-NEXT:        %57 = arith.remui %55, %33 : i32
// CHECK-NEXT:        %58 = arith.divui %57, %33 : i32
// CHECK-NEXT:        %59 = arith.remui %57, %33 : i32
// CHECK-NEXT:        %60 = arith.constant -1 : i32
// CHECK-NEXT:        %61 = arith.addi %34, %60 : i32
// CHECK-NEXT:        %62 = arith.cmpi sge, %61, %32 : i32
// CHECK-NEXT:        %63 = arith.andi %62, %43 : i1
// CHECK-NEXT:        %64 = arith.andi %63, %43 : i1
// CHECK-NEXT:        %65 = arith.muli %33, %61 : i32
// CHECK-NEXT:        %66 = arith.addi %58, %65 : i32
// CHECK-NEXT:        %67 = arith.muli %33, %56 : i32
// CHECK-NEXT:        %68 = arith.addi %66, %67 : i32
// CHECK-NEXT:        %69 = arith.constant 7 : i32
// CHECK-NEXT:        %70 = "mpi.vector_get"(%3, %33) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
// CHECK-NEXT:        %71 = "mpi.vector_get"(%3, %69) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
// CHECK-NEXT:        "scf.if"(%64) ({
// CHECK-NEXT:          %72 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %73 = memref.subview %72[2, 2, 2] [1, 101, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<101x101xf32, strided<[105, 1], offset: 22262>>
// CHECK-NEXT:          "memref.copy"(%73, %send_buff_ex1) : (memref<101x101xf32, strided<[105, 1], offset: 22262>>, memref<101x101xf32>) -> ()
// CHECK-NEXT:          "mpi.isend"(%send_buff_ex1_ptr, %9, %10, %68, %32, %70) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
// CHECK-NEXT:          "mpi.irecv"(%recv_buff_ex1_ptr, %11, %12, %68, %32, %71) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          "mpi.request_null"(%70) : (!mpi.request) -> ()
// CHECK-NEXT:          "mpi.request_null"(%71) : (!mpi.request) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %74 = arith.remui %4, %33 : i32
// CHECK-NEXT:        %75 = arith.divui %74, %33 : i32
// CHECK-NEXT:        %76 = arith.remui %74, %33 : i32
// CHECK-NEXT:        %77 = arith.divui %76, %33 : i32
// CHECK-NEXT:        %78 = arith.remui %76, %33 : i32
// CHECK-NEXT:        %79 = arith.addi %75, %33 : i32
// CHECK-NEXT:        %80 = arith.cmpi slt, %79, %33 : i32
// CHECK-NEXT:        %81 = arith.andi %43, %80 : i1
// CHECK-NEXT:        %82 = arith.andi %81, %43 : i1
// CHECK-NEXT:        %83 = arith.muli %33, %34 : i32
// CHECK-NEXT:        %84 = arith.addi %77, %83 : i32
// CHECK-NEXT:        %85 = arith.muli %33, %79 : i32
// CHECK-NEXT:        %86 = arith.addi %84, %85 : i32
// CHECK-NEXT:        %87 = arith.constant 8 : i32
// CHECK-NEXT:        %88 = "mpi.vector_get"(%3, %41) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
// CHECK-NEXT:        %89 = "mpi.vector_get"(%3, %87) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
// CHECK-NEXT:        "scf.if"(%82) ({
// CHECK-NEXT:          %90 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %91 = memref.subview %90[2, 102, 2] [51, 1, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 1], offset: 32762>>
// CHECK-NEXT:          "memref.copy"(%91, %send_buff_ex2) : (memref<51x101xf32, strided<[11025, 1], offset: 32762>>, memref<51x101xf32>) -> ()
// CHECK-NEXT:          "mpi.isend"(%send_buff_ex2_ptr, %13, %14, %86, %32, %88) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
// CHECK-NEXT:          "mpi.irecv"(%recv_buff_ex2_ptr, %15, %16, %86, %32, %89) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          "mpi.request_null"(%88) : (!mpi.request) -> ()
// CHECK-NEXT:          "mpi.request_null"(%89) : (!mpi.request) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %92 = arith.remui %4, %33 : i32
// CHECK-NEXT:        %93 = arith.divui %92, %33 : i32
// CHECK-NEXT:        %94 = arith.remui %92, %33 : i32
// CHECK-NEXT:        %95 = arith.divui %94, %33 : i32
// CHECK-NEXT:        %96 = arith.remui %94, %33 : i32
// CHECK-NEXT:        %97 = arith.addi %93, %60 : i32
// CHECK-NEXT:        %98 = arith.cmpi sge, %97, %32 : i32
// CHECK-NEXT:        %99 = arith.andi %43, %98 : i1
// CHECK-NEXT:        %100 = arith.andi %99, %43 : i1
// CHECK-NEXT:        %101 = arith.addi %95, %83 : i32
// CHECK-NEXT:        %102 = arith.muli %33, %97 : i32
// CHECK-NEXT:        %103 = arith.addi %101, %102 : i32
// CHECK-NEXT:        %104 = arith.constant 3 : i32
// CHECK-NEXT:        %105 = arith.constant 9 : i32
// CHECK-NEXT:        %106 = "mpi.vector_get"(%3, %104) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
// CHECK-NEXT:        %107 = "mpi.vector_get"(%3, %105) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
// CHECK-NEXT:        "scf.if"(%100) ({
// CHECK-NEXT:          %108 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %109 = memref.subview %108[2, 2, 2] [51, 1, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 1], offset: 22262>>
// CHECK-NEXT:          "memref.copy"(%109, %send_buff_ex3) : (memref<51x101xf32, strided<[11025, 1], offset: 22262>>, memref<51x101xf32>) -> ()
// CHECK-NEXT:          "mpi.isend"(%send_buff_ex3_ptr, %17, %18, %103, %32, %106) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
// CHECK-NEXT:          "mpi.irecv"(%recv_buff_ex3_ptr, %19, %20, %103, %32, %107) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          "mpi.request_null"(%106) : (!mpi.request) -> ()
// CHECK-NEXT:          "mpi.request_null"(%107) : (!mpi.request) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %110 = arith.remui %4, %33 : i32
// CHECK-NEXT:        %111 = arith.divui %110, %33 : i32
// CHECK-NEXT:        %112 = arith.remui %110, %33 : i32
// CHECK-NEXT:        %113 = arith.divui %112, %33 : i32
// CHECK-NEXT:        %114 = arith.remui %112, %33 : i32
// CHECK-NEXT:        %115 = arith.addi %113, %33 : i32
// CHECK-NEXT:        %116 = arith.cmpi slt, %115, %33 : i32
// CHECK-NEXT:        %117 = arith.andi %43, %43 : i1
// CHECK-NEXT:        %118 = arith.andi %117, %116 : i1
// CHECK-NEXT:        %119 = arith.addi %115, %83 : i32
// CHECK-NEXT:        %120 = arith.muli %33, %111 : i32
// CHECK-NEXT:        %121 = arith.addi %119, %120 : i32
// CHECK-NEXT:        %122 = arith.constant 4 : i32
// CHECK-NEXT:        %123 = arith.constant 10 : i32
// CHECK-NEXT:        %124 = "mpi.vector_get"(%3, %122) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
// CHECK-NEXT:        %125 = "mpi.vector_get"(%3, %123) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
// CHECK-NEXT:        "scf.if"(%118) ({
// CHECK-NEXT:          %126 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %127 = memref.subview %126[2, 2, 102] [51, 101, 1] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 105], offset: 22362>>
// CHECK-NEXT:          "memref.copy"(%127, %send_buff_ex4) : (memref<51x101xf32, strided<[11025, 105], offset: 22362>>, memref<51x101xf32>) -> ()
// CHECK-NEXT:          "mpi.isend"(%send_buff_ex4_ptr, %21, %22, %121, %32, %124) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
// CHECK-NEXT:          "mpi.irecv"(%recv_buff_ex4_ptr, %23, %24, %121, %32, %125) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          "mpi.request_null"(%124) : (!mpi.request) -> ()
// CHECK-NEXT:          "mpi.request_null"(%125) : (!mpi.request) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %128 = arith.remui %4, %33 : i32
// CHECK-NEXT:        %129 = arith.divui %128, %33 : i32
// CHECK-NEXT:        %130 = arith.remui %128, %33 : i32
// CHECK-NEXT:        %131 = arith.divui %130, %33 : i32
// CHECK-NEXT:        %132 = arith.remui %130, %33 : i32
// CHECK-NEXT:        %133 = arith.addi %131, %60 : i32
// CHECK-NEXT:        %134 = arith.cmpi sge, %133, %32 : i32
// CHECK-NEXT:        %135 = arith.andi %117, %134 : i1
// CHECK-NEXT:        %136 = arith.addi %133, %83 : i32
// CHECK-NEXT:        %137 = arith.muli %33, %129 : i32
// CHECK-NEXT:        %138 = arith.addi %136, %137 : i32
// CHECK-NEXT:        %139 = arith.constant 5 : i32
// CHECK-NEXT:        %140 = arith.constant 11 : i32
// CHECK-NEXT:        %141 = "mpi.vector_get"(%3, %139) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
// CHECK-NEXT:        %142 = "mpi.vector_get"(%3, %140) : (!mpi.vector<!mpi.request>, i32) -> !mpi.request
// CHECK-NEXT:        "scf.if"(%135) ({
// CHECK-NEXT:          %143 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %144 = memref.subview %143[2, 2, 2] [51, 101, 1] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 105], offset: 22262>>
// CHECK-NEXT:          "memref.copy"(%144, %send_buff_ex5) : (memref<51x101xf32, strided<[11025, 105], offset: 22262>>, memref<51x101xf32>) -> ()
// CHECK-NEXT:          "mpi.isend"(%send_buff_ex5_ptr, %25, %26, %138, %32, %141) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
// CHECK-NEXT:          "mpi.irecv"(%recv_buff_ex5_ptr, %27, %28, %138, %32, %142) : (!llvm.ptr, i32, !mpi.datatype, i32, i32, !mpi.request) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          "mpi.request_null"(%141) : (!mpi.request) -> ()
// CHECK-NEXT:          "mpi.request_null"(%142) : (!mpi.request) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "mpi.waitall"(%3, %2) : (!mpi.vector<!mpi.request>, i32) -> ()
// CHECK-NEXT:        "scf.if"(%45) ({
// CHECK-NEXT:          %145 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %146 = memref.subview %145[53, 2, 2] [1, 101, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<101x101xf32, strided<[105, 1], offset: 584537>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex0, %146) : (memref<101x101xf32>, memref<101x101xf32, strided<[105, 1], offset: 584537>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%64) ({
// CHECK-NEXT:          %147 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %148 = memref.subview %147[1, 2, 2] [1, 101, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<101x101xf32, strided<[105, 1], offset: 11237>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex1, %148) : (memref<101x101xf32>, memref<101x101xf32, strided<[105, 1], offset: 11237>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%82) ({
// CHECK-NEXT:          %149 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %150 = memref.subview %149[2, 103, 2] [51, 1, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 1], offset: 32867>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex2, %150) : (memref<51x101xf32>, memref<51x101xf32, strided<[11025, 1], offset: 32867>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%100) ({
// CHECK-NEXT:          %151 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %152 = memref.subview %151[2, 1, 2] [51, 1, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 1], offset: 22157>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex3, %152) : (memref<51x101xf32>, memref<51x101xf32, strided<[11025, 1], offset: 22157>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%118) ({
// CHECK-NEXT:          %153 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %154 = memref.subview %153[2, 2, 103] [51, 101, 1] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 105], offset: 22363>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex4, %154) : (memref<51x101xf32>, memref<51x101xf32, strided<[11025, 105], offset: 22363>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%135) ({
// CHECK-NEXT:          %155 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %156 = memref.subview %155[2, 2, 1] [51, 101, 1] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 105], offset: 22261>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex5, %156) : (memref<51x101xf32>, memref<51x101xf32, strided<[11025, 105], offset: 22261>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %157 = memref.subview %u_t1[2, 2, 2] [55, 105, 105] [1, 1, 1] : memref<55x105x105xf32> to memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        %u_t0_blk = memref.subview %u_t0[2, 2, 2] [55, 105, 105] [1, 1, 1] : memref<55x105x105xf32> to memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        %u_t2_blk = memref.subview %u_t2[2, 2, 2] [55, 105, 105] [1, 1, 1] : memref<55x105x105xf32> to memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        %158 = arith.constant 0 : index
// CHECK-NEXT:        %159 = arith.constant 51 : index
// CHECK-NEXT:        %160 = arith.constant 101 : index
// CHECK-NEXT:        %161 = arith.constant 64 : index
// CHECK-NEXT:        %162 = arith.muli %time_m, %161 : index
// CHECK-NEXT:        "scf.parallel"(%158, %158, %159, %160, %162, %162) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:        ^0(%163 : index, %164 : index):
// CHECK-NEXT:          %165 = "affine.min"(%161, %159, %163) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:          %166 = "affine.min"(%161, %160, %164) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:          "scf.parallel"(%158, %158, %158, %165, %166, %160, %time_m, %time_m, %time_m) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:          ^1(%167 : index, %168 : index, %169 : index):
// CHECK-NEXT:            %170 = arith.addi %163, %167 : index
// CHECK-NEXT:            %171 = arith.addi %164, %168 : index
// CHECK-NEXT:            %dt = arith.constant 1.000000e-04 : f32
// CHECK-NEXT:            %172 = arith.constant 2 : i64
// CHECK-NEXT:            %173 = "math.fpowi"(%dt, %172) : (f32, i64) -> f32
// CHECK-NEXT:            %174 = arith.constant -1 : i64
// CHECK-NEXT:            %175 = arith.constant -2 : i64
// CHECK-NEXT:            %176 = "math.fpowi"(%dt, %175) : (f32, i64) -> f32
// CHECK-NEXT:            %177 = memref.load %u_t2_blk[%170, %171, %169] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %178 = arith.mulf %176, %177 : f32
// CHECK-NEXT:            %179 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:            %180 = memref.load %u_t0_blk[%170, %171, %169] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %181 = arith.mulf %179, %176 : f32
// CHECK-NEXT:            %182 = arith.mulf %181, %180 : f32
// CHECK-NEXT:            %183 = arith.addf %178, %182 : f32
// CHECK-NEXT:            %184 = arith.sitofp %174 : i64 to f32
// CHECK-NEXT:            %185 = arith.mulf %184, %183 : f32
// CHECK-NEXT:            %h_x = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %186 = "math.fpowi"(%h_x, %175) : (f32, i64) -> f32
// CHECK-NEXT:            %187 = arith.constant -1 : index
// CHECK-NEXT:            %188 = arith.addi %170, %187 : index
// CHECK-NEXT:            %189 = memref.load %u_t0_blk[%188, %171, %169] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %190 = arith.mulf %186, %189 : f32
// CHECK-NEXT:            %191 = arith.addi %170, %time_m : index
// CHECK-NEXT:            %192 = memref.load %u_t0_blk[%191, %171, %169] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %193 = arith.mulf %186, %192 : f32
// CHECK-NEXT:            %194 = arith.mulf %179, %186 : f32
// CHECK-NEXT:            %195 = arith.mulf %194, %180 : f32
// CHECK-NEXT:            %196 = arith.addf %190, %193 : f32
// CHECK-NEXT:            %197 = arith.addf %196, %195 : f32
// CHECK-NEXT:            %198 = arith.addi %171, %187 : index
// CHECK-NEXT:            %199 = memref.load %u_t0_blk[%170, %198, %169] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %200 = arith.mulf %186, %199 : f32
// CHECK-NEXT:            %201 = arith.addi %171, %time_m : index
// CHECK-NEXT:            %202 = memref.load %u_t0_blk[%170, %201, %169] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %203 = arith.mulf %186, %202 : f32
// CHECK-NEXT:            %204 = arith.addf %200, %203 : f32
// CHECK-NEXT:            %205 = arith.addf %204, %195 : f32
// CHECK-NEXT:            %206 = arith.addi %169, %187 : index
// CHECK-NEXT:            %207 = memref.load %u_t0_blk[%170, %171, %206] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %208 = arith.mulf %186, %207 : f32
// CHECK-NEXT:            %209 = arith.addi %169, %time_m : index
// CHECK-NEXT:            %210 = memref.load %u_t0_blk[%170, %171, %209] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %211 = arith.mulf %186, %210 : f32
// CHECK-NEXT:            %212 = arith.addf %208, %211 : f32
// CHECK-NEXT:            %213 = arith.addf %212, %195 : f32
// CHECK-NEXT:            %214 = arith.addf %185, %197 : f32
// CHECK-NEXT:            %215 = arith.addf %214, %205 : f32
// CHECK-NEXT:            %216 = arith.addf %215, %213 : f32
// CHECK-NEXT:            %217 = arith.mulf %173, %216 : f32
// CHECK-NEXT:            memref.store %217, %157[%170, %171, %169] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:        scf.yield %u_t1, %u_t2, %u_t0 : memref<55x105x105xf32>, memref<55x105x105xf32>, memref<55x105x105xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %218 = func.call @timer_end(%0) : (f64) -> f64
// CHECK-NEXT:      "llvm.store"(%218, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @timer_start() -> f64
// CHECK-NEXT:    func.func private @timer_end(f64) -> f64
// CHECK-NEXT:  }
