// RUN: xdsl-opt -p "canonicalize,cse,distribute-stencil{strategy=3d-grid slices=2,1,1 restrict_domain=false},shape-inference,canonicalize-dmp,stencil-bufferize,dmp-to-mpi{mpi_init=false},convert-stencil-to-ll-mlir,scf-parallel-loop-tiling{parallel-loop-tile-sizes=64,64,0},dmp-to-mpi{mpi_init=false},lower-mpi,canonicalize,cse" %s | filecheck %s

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
// CHECK-NEXT:      %3 = "llvm.alloca"(%2) <{"alignment" = 32 : i64, "elem_type" = i32}> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %4 = arith.constant 1140850688 : i32
// CHECK-NEXT:      %5 = arith.constant 1 : i64
// CHECK-NEXT:      %6 = "llvm.alloca"(%5) <{"alignment" = 32 : i64, "elem_type" = i32}> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %7 = func.call @MPI_Comm_rank(%4, %6) : (i32, !llvm.ptr) -> i32
// CHECK-NEXT:      %8 = "llvm.load"(%6) : (!llvm.ptr) -> i32
// CHECK-NEXT:      %send_buff_ex0 = memref.alloc() {"alignment" = 64 : i64} : memref<101x101xf32>
// CHECK-NEXT:      %9 = "memref.extract_aligned_pointer_as_index"(%send_buff_ex0) : (memref<101x101xf32>) -> index
// CHECK-NEXT:      %10 = arith.index_cast %9 : index to i64
// CHECK-NEXT:      %send_buff_ex0_ptr = "llvm.inttoptr"(%10) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %11 = arith.constant 10201 : i32
// CHECK-NEXT:      %12 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %recv_buff_ex0 = memref.alloc() {"alignment" = 64 : i64} : memref<101x101xf32>
// CHECK-NEXT:      %13 = "memref.extract_aligned_pointer_as_index"(%recv_buff_ex0) : (memref<101x101xf32>) -> index
// CHECK-NEXT:      %14 = arith.index_cast %13 : index to i64
// CHECK-NEXT:      %recv_buff_ex0_ptr = "llvm.inttoptr"(%14) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %send_buff_ex1 = memref.alloc() {"alignment" = 64 : i64} : memref<101x101xf32>
// CHECK-NEXT:      %15 = "memref.extract_aligned_pointer_as_index"(%send_buff_ex1) : (memref<101x101xf32>) -> index
// CHECK-NEXT:      %16 = arith.index_cast %15 : index to i64
// CHECK-NEXT:      %send_buff_ex1_ptr = "llvm.inttoptr"(%16) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %recv_buff_ex1 = memref.alloc() {"alignment" = 64 : i64} : memref<101x101xf32>
// CHECK-NEXT:      %17 = "memref.extract_aligned_pointer_as_index"(%recv_buff_ex1) : (memref<101x101xf32>) -> index
// CHECK-NEXT:      %18 = arith.index_cast %17 : index to i64
// CHECK-NEXT:      %recv_buff_ex1_ptr = "llvm.inttoptr"(%18) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %send_buff_ex2 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %19 = "memref.extract_aligned_pointer_as_index"(%send_buff_ex2) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %20 = arith.index_cast %19 : index to i64
// CHECK-NEXT:      %send_buff_ex2_ptr = "llvm.inttoptr"(%20) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %21 = arith.constant 5151 : i32
// CHECK-NEXT:      %recv_buff_ex2 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %22 = "memref.extract_aligned_pointer_as_index"(%recv_buff_ex2) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %23 = arith.index_cast %22 : index to i64
// CHECK-NEXT:      %recv_buff_ex2_ptr = "llvm.inttoptr"(%23) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %send_buff_ex3 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %24 = "memref.extract_aligned_pointer_as_index"(%send_buff_ex3) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %25 = arith.index_cast %24 : index to i64
// CHECK-NEXT:      %send_buff_ex3_ptr = "llvm.inttoptr"(%25) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %recv_buff_ex3 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %26 = "memref.extract_aligned_pointer_as_index"(%recv_buff_ex3) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %27 = arith.index_cast %26 : index to i64
// CHECK-NEXT:      %recv_buff_ex3_ptr = "llvm.inttoptr"(%27) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %send_buff_ex4 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %28 = "memref.extract_aligned_pointer_as_index"(%send_buff_ex4) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %29 = arith.index_cast %28 : index to i64
// CHECK-NEXT:      %send_buff_ex4_ptr = "llvm.inttoptr"(%29) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %recv_buff_ex4 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %30 = "memref.extract_aligned_pointer_as_index"(%recv_buff_ex4) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %31 = arith.index_cast %30 : index to i64
// CHECK-NEXT:      %recv_buff_ex4_ptr = "llvm.inttoptr"(%31) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %send_buff_ex5 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %32 = "memref.extract_aligned_pointer_as_index"(%send_buff_ex5) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %33 = arith.index_cast %32 : index to i64
// CHECK-NEXT:      %send_buff_ex5_ptr = "llvm.inttoptr"(%33) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %recv_buff_ex5 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %34 = "memref.extract_aligned_pointer_as_index"(%recv_buff_ex5) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %35 = arith.index_cast %34 : index to i64
// CHECK-NEXT:      %recv_buff_ex5_ptr = "llvm.inttoptr"(%35) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %36, %37, %38 = scf.for %time = %time_m to %1 step %time_m iter_args(%u_t0 = %u_vec0, %u_t1 = %u_vec1, %u_t2 = %u_vec2) -> (memref<55x105x105xf32>, memref<55x105x105xf32>, memref<55x105x105xf32>) {
// CHECK-NEXT:        %39 = arith.constant 0 : i32
// CHECK-NEXT:        %40 = arith.constant 1 : i32
// CHECK-NEXT:        %41 = arith.divui %8, %40 : i32
// CHECK-NEXT:        %42 = arith.remui %8, %40 : i32
// CHECK-NEXT:        %43 = arith.divui %42, %40 : i32
// CHECK-NEXT:        %44 = arith.remui %42, %40 : i32
// CHECK-NEXT:        %45 = arith.divui %44, %40 : i32
// CHECK-NEXT:        %46 = arith.remui %44, %40 : i32
// CHECK-NEXT:        %47 = arith.addi %41, %40 : i32
// CHECK-NEXT:        %48 = arith.constant 2 : i32
// CHECK-NEXT:        %49 = arith.cmpi slt, %47, %48 : i32
// CHECK-NEXT:        %50 = arith.constant true
// CHECK-NEXT:        %51 = arith.andi %49, %50 : i1
// CHECK-NEXT:        %52 = arith.andi %51, %50 : i1
// CHECK-NEXT:        %53 = arith.muli %40, %47 : i32
// CHECK-NEXT:        %54 = arith.addi %45, %53 : i32
// CHECK-NEXT:        %55 = arith.muli %40, %43 : i32
// CHECK-NEXT:        %56 = arith.addi %54, %55 : i32
// CHECK-NEXT:        %57 = arith.constant 6 : i32
// CHECK-NEXT:        %58 = "llvm.ptrtoint"(%3) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %59 = arith.constant 4 : i64
// CHECK-NEXT:        %60 = arith.index_cast %39 : i32 to index
// CHECK-NEXT:        %61 = arith.index_cast %60 : index to i64
// CHECK-NEXT:        %62 = arith.muli %59, %61 : i64
// CHECK-NEXT:        %63 = arith.addi %62, %58 : i64
// CHECK-NEXT:        %64 = "llvm.inttoptr"(%63) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %65 = "llvm.ptrtoint"(%3) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %66 = arith.index_cast %57 : i32 to index
// CHECK-NEXT:        %67 = arith.index_cast %66 : index to i64
// CHECK-NEXT:        %68 = arith.muli %59, %67 : i64
// CHECK-NEXT:        %69 = arith.addi %68, %65 : i64
// CHECK-NEXT:        %70 = "llvm.inttoptr"(%69) : (i64) -> !llvm.ptr
// CHECK-NEXT:        "scf.if"(%52) ({
// CHECK-NEXT:          %71 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %72 = memref.subview %71[52, 2, 2] [1, 101, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<101x101xf32, strided<[105, 1], offset: 573512>>
// CHECK-NEXT:          "memref.copy"(%72, %send_buff_ex0) : (memref<101x101xf32, strided<[105, 1], offset: 573512>>, memref<101x101xf32>) -> ()
// CHECK-NEXT:          %73 = func.call @MPI_Isend(%send_buff_ex0_ptr, %11, %12, %56, %39, %4, %64) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          %74 = func.call @MPI_Irecv(%recv_buff_ex0_ptr, %11, %12, %56, %39, %4, %70) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %75 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%75, %64) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          "llvm.store"(%75, %70) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %76 = arith.remui %8, %40 : i32
// CHECK-NEXT:        %77 = arith.divui %76, %40 : i32
// CHECK-NEXT:        %78 = arith.remui %76, %40 : i32
// CHECK-NEXT:        %79 = arith.divui %78, %40 : i32
// CHECK-NEXT:        %80 = arith.remui %78, %40 : i32
// CHECK-NEXT:        %81 = arith.constant -1 : i32
// CHECK-NEXT:        %82 = arith.addi %41, %81 : i32
// CHECK-NEXT:        %83 = arith.cmpi sge, %82, %39 : i32
// CHECK-NEXT:        %84 = arith.andi %83, %50 : i1
// CHECK-NEXT:        %85 = arith.andi %84, %50 : i1
// CHECK-NEXT:        %86 = arith.muli %40, %82 : i32
// CHECK-NEXT:        %87 = arith.addi %79, %86 : i32
// CHECK-NEXT:        %88 = arith.muli %40, %77 : i32
// CHECK-NEXT:        %89 = arith.addi %87, %88 : i32
// CHECK-NEXT:        %90 = arith.constant 7 : i32
// CHECK-NEXT:        %91 = "llvm.ptrtoint"(%3) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %92 = arith.index_cast %40 : i32 to index
// CHECK-NEXT:        %93 = arith.index_cast %92 : index to i64
// CHECK-NEXT:        %94 = arith.muli %59, %93 : i64
// CHECK-NEXT:        %95 = arith.addi %94, %91 : i64
// CHECK-NEXT:        %96 = "llvm.inttoptr"(%95) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %97 = "llvm.ptrtoint"(%3) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %98 = arith.index_cast %90 : i32 to index
// CHECK-NEXT:        %99 = arith.index_cast %98 : index to i64
// CHECK-NEXT:        %100 = arith.muli %59, %99 : i64
// CHECK-NEXT:        %101 = arith.addi %100, %97 : i64
// CHECK-NEXT:        %102 = "llvm.inttoptr"(%101) : (i64) -> !llvm.ptr
// CHECK-NEXT:        "scf.if"(%85) ({
// CHECK-NEXT:          %103 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %104 = memref.subview %103[2, 2, 2] [1, 101, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<101x101xf32, strided<[105, 1], offset: 22262>>
// CHECK-NEXT:          "memref.copy"(%104, %send_buff_ex1) : (memref<101x101xf32, strided<[105, 1], offset: 22262>>, memref<101x101xf32>) -> ()
// CHECK-NEXT:          %105 = func.call @MPI_Isend(%send_buff_ex1_ptr, %11, %12, %89, %39, %4, %96) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          %106 = func.call @MPI_Irecv(%recv_buff_ex1_ptr, %11, %12, %89, %39, %4, %102) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %107 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%107, %96) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          "llvm.store"(%107, %102) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %108 = arith.remui %8, %40 : i32
// CHECK-NEXT:        %109 = arith.divui %108, %40 : i32
// CHECK-NEXT:        %110 = arith.remui %108, %40 : i32
// CHECK-NEXT:        %111 = arith.divui %110, %40 : i32
// CHECK-NEXT:        %112 = arith.remui %110, %40 : i32
// CHECK-NEXT:        %113 = arith.addi %109, %40 : i32
// CHECK-NEXT:        %114 = arith.cmpi slt, %113, %40 : i32
// CHECK-NEXT:        %115 = arith.andi %50, %114 : i1
// CHECK-NEXT:        %116 = arith.andi %115, %50 : i1
// CHECK-NEXT:        %117 = arith.muli %40, %41 : i32
// CHECK-NEXT:        %118 = arith.addi %111, %117 : i32
// CHECK-NEXT:        %119 = arith.muli %40, %113 : i32
// CHECK-NEXT:        %120 = arith.addi %118, %119 : i32
// CHECK-NEXT:        %121 = arith.constant 8 : i32
// CHECK-NEXT:        %122 = "llvm.ptrtoint"(%3) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %123 = arith.index_cast %48 : i32 to index
// CHECK-NEXT:        %124 = arith.index_cast %123 : index to i64
// CHECK-NEXT:        %125 = arith.muli %59, %124 : i64
// CHECK-NEXT:        %126 = arith.addi %125, %122 : i64
// CHECK-NEXT:        %127 = "llvm.inttoptr"(%126) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %128 = "llvm.ptrtoint"(%3) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %129 = arith.index_cast %121 : i32 to index
// CHECK-NEXT:        %130 = arith.index_cast %129 : index to i64
// CHECK-NEXT:        %131 = arith.muli %59, %130 : i64
// CHECK-NEXT:        %132 = arith.addi %131, %128 : i64
// CHECK-NEXT:        %133 = "llvm.inttoptr"(%132) : (i64) -> !llvm.ptr
// CHECK-NEXT:        "scf.if"(%116) ({
// CHECK-NEXT:          %134 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %135 = memref.subview %134[2, 102, 2] [51, 1, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 1], offset: 32762>>
// CHECK-NEXT:          "memref.copy"(%135, %send_buff_ex2) : (memref<51x101xf32, strided<[11025, 1], offset: 32762>>, memref<51x101xf32>) -> ()
// CHECK-NEXT:          %136 = func.call @MPI_Isend(%send_buff_ex2_ptr, %21, %12, %120, %39, %4, %127) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          %137 = func.call @MPI_Irecv(%recv_buff_ex2_ptr, %21, %12, %120, %39, %4, %133) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %138 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%138, %127) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          "llvm.store"(%138, %133) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %139 = arith.remui %8, %40 : i32
// CHECK-NEXT:        %140 = arith.divui %139, %40 : i32
// CHECK-NEXT:        %141 = arith.remui %139, %40 : i32
// CHECK-NEXT:        %142 = arith.divui %141, %40 : i32
// CHECK-NEXT:        %143 = arith.remui %141, %40 : i32
// CHECK-NEXT:        %144 = arith.addi %140, %81 : i32
// CHECK-NEXT:        %145 = arith.cmpi sge, %144, %39 : i32
// CHECK-NEXT:        %146 = arith.andi %50, %145 : i1
// CHECK-NEXT:        %147 = arith.andi %146, %50 : i1
// CHECK-NEXT:        %148 = arith.addi %142, %117 : i32
// CHECK-NEXT:        %149 = arith.muli %40, %144 : i32
// CHECK-NEXT:        %150 = arith.addi %148, %149 : i32
// CHECK-NEXT:        %151 = arith.constant 3 : i32
// CHECK-NEXT:        %152 = arith.constant 9 : i32
// CHECK-NEXT:        %153 = "llvm.ptrtoint"(%3) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %154 = arith.index_cast %151 : i32 to index
// CHECK-NEXT:        %155 = arith.index_cast %154 : index to i64
// CHECK-NEXT:        %156 = arith.muli %59, %155 : i64
// CHECK-NEXT:        %157 = arith.addi %156, %153 : i64
// CHECK-NEXT:        %158 = "llvm.inttoptr"(%157) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %159 = "llvm.ptrtoint"(%3) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %160 = arith.index_cast %152 : i32 to index
// CHECK-NEXT:        %161 = arith.index_cast %160 : index to i64
// CHECK-NEXT:        %162 = arith.muli %59, %161 : i64
// CHECK-NEXT:        %163 = arith.addi %162, %159 : i64
// CHECK-NEXT:        %164 = "llvm.inttoptr"(%163) : (i64) -> !llvm.ptr
// CHECK-NEXT:        "scf.if"(%147) ({
// CHECK-NEXT:          %165 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %166 = memref.subview %165[2, 2, 2] [51, 1, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 1], offset: 22262>>
// CHECK-NEXT:          "memref.copy"(%166, %send_buff_ex3) : (memref<51x101xf32, strided<[11025, 1], offset: 22262>>, memref<51x101xf32>) -> ()
// CHECK-NEXT:          %167 = func.call @MPI_Isend(%send_buff_ex3_ptr, %21, %12, %150, %39, %4, %158) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          %168 = func.call @MPI_Irecv(%recv_buff_ex3_ptr, %21, %12, %150, %39, %4, %164) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %169 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%169, %158) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          "llvm.store"(%169, %164) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %170 = arith.remui %8, %40 : i32
// CHECK-NEXT:        %171 = arith.divui %170, %40 : i32
// CHECK-NEXT:        %172 = arith.remui %170, %40 : i32
// CHECK-NEXT:        %173 = arith.divui %172, %40 : i32
// CHECK-NEXT:        %174 = arith.remui %172, %40 : i32
// CHECK-NEXT:        %175 = arith.addi %173, %40 : i32
// CHECK-NEXT:        %176 = arith.cmpi slt, %175, %40 : i32
// CHECK-NEXT:        %177 = arith.andi %50, %50 : i1
// CHECK-NEXT:        %178 = arith.andi %177, %176 : i1
// CHECK-NEXT:        %179 = arith.addi %175, %117 : i32
// CHECK-NEXT:        %180 = arith.muli %40, %171 : i32
// CHECK-NEXT:        %181 = arith.addi %179, %180 : i32
// CHECK-NEXT:        %182 = arith.constant 4 : i32
// CHECK-NEXT:        %183 = arith.constant 10 : i32
// CHECK-NEXT:        %184 = "llvm.ptrtoint"(%3) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %185 = arith.index_cast %182 : i32 to index
// CHECK-NEXT:        %186 = arith.index_cast %185 : index to i64
// CHECK-NEXT:        %187 = arith.muli %59, %186 : i64
// CHECK-NEXT:        %188 = arith.addi %187, %184 : i64
// CHECK-NEXT:        %189 = "llvm.inttoptr"(%188) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %190 = "llvm.ptrtoint"(%3) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %191 = arith.index_cast %183 : i32 to index
// CHECK-NEXT:        %192 = arith.index_cast %191 : index to i64
// CHECK-NEXT:        %193 = arith.muli %59, %192 : i64
// CHECK-NEXT:        %194 = arith.addi %193, %190 : i64
// CHECK-NEXT:        %195 = "llvm.inttoptr"(%194) : (i64) -> !llvm.ptr
// CHECK-NEXT:        "scf.if"(%178) ({
// CHECK-NEXT:          %196 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %197 = memref.subview %196[2, 2, 102] [51, 101, 1] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 105], offset: 22362>>
// CHECK-NEXT:          "memref.copy"(%197, %send_buff_ex4) : (memref<51x101xf32, strided<[11025, 105], offset: 22362>>, memref<51x101xf32>) -> ()
// CHECK-NEXT:          %198 = func.call @MPI_Isend(%send_buff_ex4_ptr, %21, %12, %181, %39, %4, %189) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          %199 = func.call @MPI_Irecv(%recv_buff_ex4_ptr, %21, %12, %181, %39, %4, %195) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %200 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%200, %189) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          "llvm.store"(%200, %195) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %201 = arith.remui %8, %40 : i32
// CHECK-NEXT:        %202 = arith.divui %201, %40 : i32
// CHECK-NEXT:        %203 = arith.remui %201, %40 : i32
// CHECK-NEXT:        %204 = arith.divui %203, %40 : i32
// CHECK-NEXT:        %205 = arith.remui %203, %40 : i32
// CHECK-NEXT:        %206 = arith.addi %204, %81 : i32
// CHECK-NEXT:        %207 = arith.cmpi sge, %206, %39 : i32
// CHECK-NEXT:        %208 = arith.andi %177, %207 : i1
// CHECK-NEXT:        %209 = arith.addi %206, %117 : i32
// CHECK-NEXT:        %210 = arith.muli %40, %202 : i32
// CHECK-NEXT:        %211 = arith.addi %209, %210 : i32
// CHECK-NEXT:        %212 = arith.constant 5 : i32
// CHECK-NEXT:        %213 = arith.constant 11 : i32
// CHECK-NEXT:        %214 = "llvm.ptrtoint"(%3) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %215 = arith.index_cast %212 : i32 to index
// CHECK-NEXT:        %216 = arith.index_cast %215 : index to i64
// CHECK-NEXT:        %217 = arith.muli %59, %216 : i64
// CHECK-NEXT:        %218 = arith.addi %217, %214 : i64
// CHECK-NEXT:        %219 = "llvm.inttoptr"(%218) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %220 = "llvm.ptrtoint"(%3) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %221 = arith.index_cast %213 : i32 to index
// CHECK-NEXT:        %222 = arith.index_cast %221 : index to i64
// CHECK-NEXT:        %223 = arith.muli %59, %222 : i64
// CHECK-NEXT:        %224 = arith.addi %223, %220 : i64
// CHECK-NEXT:        %225 = "llvm.inttoptr"(%224) : (i64) -> !llvm.ptr
// CHECK-NEXT:        "scf.if"(%208) ({
// CHECK-NEXT:          %226 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %227 = memref.subview %226[2, 2, 2] [51, 101, 1] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 105], offset: 22262>>
// CHECK-NEXT:          "memref.copy"(%227, %send_buff_ex5) : (memref<51x101xf32, strided<[11025, 105], offset: 22262>>, memref<51x101xf32>) -> ()
// CHECK-NEXT:          %228 = func.call @MPI_Isend(%send_buff_ex5_ptr, %21, %12, %211, %39, %4, %219) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          %229 = func.call @MPI_Irecv(%recv_buff_ex5_ptr, %21, %12, %211, %39, %4, %225) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %230 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%230, %219) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          "llvm.store"(%230, %225) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %231 = "llvm.inttoptr"(%5) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %232 = func.call @MPI_Waitall(%2, %3, %231) : (i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:        "scf.if"(%52) ({
// CHECK-NEXT:          %233 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %234 = memref.subview %233[53, 2, 2] [1, 101, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<101x101xf32, strided<[105, 1], offset: 584537>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex0, %234) : (memref<101x101xf32>, memref<101x101xf32, strided<[105, 1], offset: 584537>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%85) ({
// CHECK-NEXT:          %235 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %236 = memref.subview %235[1, 2, 2] [1, 101, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<101x101xf32, strided<[105, 1], offset: 11237>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex1, %236) : (memref<101x101xf32>, memref<101x101xf32, strided<[105, 1], offset: 11237>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%116) ({
// CHECK-NEXT:          %237 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %238 = memref.subview %237[2, 103, 2] [51, 1, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 1], offset: 32867>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex2, %238) : (memref<51x101xf32>, memref<51x101xf32, strided<[11025, 1], offset: 32867>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%147) ({
// CHECK-NEXT:          %239 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %240 = memref.subview %239[2, 1, 2] [51, 1, 101] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 1], offset: 22157>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex3, %240) : (memref<51x101xf32>, memref<51x101xf32, strided<[11025, 1], offset: 22157>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%178) ({
// CHECK-NEXT:          %241 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %242 = memref.subview %241[2, 2, 103] [51, 101, 1] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 105], offset: 22363>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex4, %242) : (memref<51x101xf32>, memref<51x101xf32, strided<[11025, 105], offset: 22363>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%208) ({
// CHECK-NEXT:          %243 = builtin.unrealized_conversion_cast %u_t0 : memref<55x105x105xf32> to memref<55x105x105xf32>
// CHECK-NEXT:          %244 = memref.subview %243[2, 2, 1] [51, 101, 1] [1, 1, 1] : memref<55x105x105xf32> to memref<51x101xf32, strided<[11025, 105], offset: 22261>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex5, %244) : (memref<51x101xf32>, memref<51x101xf32, strided<[11025, 105], offset: 22261>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %245 = memref.subview %u_t1[2, 2, 2] [55, 105, 105] [1, 1, 1] : memref<55x105x105xf32> to memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        %u_t0_blk = memref.subview %u_t0[2, 2, 2] [55, 105, 105] [1, 1, 1] : memref<55x105x105xf32> to memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        %u_t2_blk = memref.subview %u_t2[2, 2, 2] [55, 105, 105] [1, 1, 1] : memref<55x105x105xf32> to memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        %246 = arith.constant 0 : index
// CHECK-NEXT:        %247 = arith.constant 51 : index
// CHECK-NEXT:        %248 = arith.constant 101 : index
// CHECK-NEXT:        %249 = arith.constant 64 : index
// CHECK-NEXT:        %250 = arith.muli %time_m, %249 : index
// CHECK-NEXT:        "scf.parallel"(%246, %246, %247, %248, %250, %250) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:        ^0(%251 : index, %252 : index):
// CHECK-NEXT:          %253 = "affine.min"(%249, %247, %251) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:          %254 = "affine.min"(%249, %248, %252) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:          "scf.parallel"(%246, %246, %246, %253, %254, %248, %time_m, %time_m, %time_m) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:          ^1(%255 : index, %256 : index, %257 : index):
// CHECK-NEXT:            %258 = arith.addi %251, %255 : index
// CHECK-NEXT:            %259 = arith.addi %252, %256 : index
// CHECK-NEXT:            %dt = arith.constant 1.000000e-04 : f32
// CHECK-NEXT:            %260 = arith.constant 2 : i64
// CHECK-NEXT:            %261 = "math.fpowi"(%dt, %260) : (f32, i64) -> f32
// CHECK-NEXT:            %262 = arith.constant -1 : i64
// CHECK-NEXT:            %263 = arith.constant -2 : i64
// CHECK-NEXT:            %264 = "math.fpowi"(%dt, %263) : (f32, i64) -> f32
// CHECK-NEXT:            %265 = memref.load %u_t2_blk[%258, %259, %257] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %266 = arith.mulf %264, %265 : f32
// CHECK-NEXT:            %267 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:            %268 = memref.load %u_t0_blk[%258, %259, %257] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %269 = arith.mulf %267, %264 : f32
// CHECK-NEXT:            %270 = arith.mulf %269, %268 : f32
// CHECK-NEXT:            %271 = arith.addf %266, %270 : f32
// CHECK-NEXT:            %272 = arith.sitofp %262 : i64 to f32
// CHECK-NEXT:            %273 = arith.mulf %272, %271 : f32
// CHECK-NEXT:            %h_x = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %274 = "math.fpowi"(%h_x, %263) : (f32, i64) -> f32
// CHECK-NEXT:            %275 = arith.constant -1 : index
// CHECK-NEXT:            %276 = arith.addi %258, %275 : index
// CHECK-NEXT:            %277 = memref.load %u_t0_blk[%276, %259, %257] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %278 = arith.mulf %274, %277 : f32
// CHECK-NEXT:            %279 = arith.addi %258, %time_m : index
// CHECK-NEXT:            %280 = memref.load %u_t0_blk[%279, %259, %257] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %281 = arith.mulf %274, %280 : f32
// CHECK-NEXT:            %282 = arith.mulf %267, %274 : f32
// CHECK-NEXT:            %283 = arith.mulf %282, %268 : f32
// CHECK-NEXT:            %284 = arith.addf %278, %281 : f32
// CHECK-NEXT:            %285 = arith.addf %284, %283 : f32
// CHECK-NEXT:            %286 = arith.addi %259, %275 : index
// CHECK-NEXT:            %287 = memref.load %u_t0_blk[%258, %286, %257] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %288 = arith.mulf %274, %287 : f32
// CHECK-NEXT:            %289 = arith.addi %259, %time_m : index
// CHECK-NEXT:            %290 = memref.load %u_t0_blk[%258, %289, %257] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %291 = arith.mulf %274, %290 : f32
// CHECK-NEXT:            %292 = arith.addf %288, %291 : f32
// CHECK-NEXT:            %293 = arith.addf %292, %283 : f32
// CHECK-NEXT:            %294 = arith.addi %257, %275 : index
// CHECK-NEXT:            %295 = memref.load %u_t0_blk[%258, %259, %294] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %296 = arith.mulf %274, %295 : f32
// CHECK-NEXT:            %297 = arith.addi %257, %time_m : index
// CHECK-NEXT:            %298 = memref.load %u_t0_blk[%258, %259, %297] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %299 = arith.mulf %274, %298 : f32
// CHECK-NEXT:            %300 = arith.addf %296, %299 : f32
// CHECK-NEXT:            %301 = arith.addf %300, %283 : f32
// CHECK-NEXT:            %302 = arith.addf %273, %285 : f32
// CHECK-NEXT:            %303 = arith.addf %302, %293 : f32
// CHECK-NEXT:            %304 = arith.addf %303, %301 : f32
// CHECK-NEXT:            %305 = arith.mulf %261, %304 : f32
// CHECK-NEXT:            memref.store %305, %245[%258, %259, %257] : memref<55x105x105xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:        scf.yield %u_t1, %u_t2, %u_t0 : memref<55x105x105xf32>, memref<55x105x105xf32>, memref<55x105x105xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %306 = func.call @timer_end(%0) : (f64) -> f64
// CHECK-NEXT:      "llvm.store"(%306, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @timer_start() -> f64
// CHECK-NEXT:    func.func private @timer_end(f64) -> f64
// CHECK-NEXT:    func.func private @MPI_Comm_rank(i32, !llvm.ptr) -> i32
// CHECK-NEXT:    func.func private @MPI_Isend(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    func.func private @MPI_Irecv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    func.func private @MPI_Waitall(i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:  }
