// RUN: xdsl-opt -p "distribute-stencil{strategy=3d-grid slices=2,1,1 restrict_domain=false},canonicalize-dmp,convert-stencil-to-ll-mlir,scf-parallel-loop-tiling{parallel-loop-tile-sizes=64,64,0},dmp-to-mpi{mpi_init=false},lower-mpi" %s | filecheck %s

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
// CHECK-NEXT:      %3 = arith.constant 12 : i32
// CHECK-NEXT:      %4 = "llvm.alloca"(%3) <{"alignment" = 32 : i64, "elem_type" = i32}> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %5 = arith.constant 1140850688 : i32
// CHECK-NEXT:      %6 = arith.constant 1 : i64
// CHECK-NEXT:      %7 = "llvm.alloca"(%6) <{"alignment" = 32 : i64, "elem_type" = i32}> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %8 = func.call @MPI_Comm_rank(%5, %7) : (i32, !llvm.ptr) -> i32
// CHECK-NEXT:      %9 = "llvm.load"(%7) : (!llvm.ptr) -> i32
// CHECK-NEXT:      %send_buff_ex0 = memref.alloc() {"alignment" = 64 : i64} : memref<101x101xf32>
// CHECK-NEXT:      %10 = "memref.extract_aligned_pointer_as_index"(%send_buff_ex0) : (memref<101x101xf32>) -> index
// CHECK-NEXT:      %11 = arith.index_cast %10 : index to i64
// CHECK-NEXT:      %send_buff_ex0_ptr = "llvm.inttoptr"(%11) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %12 = arith.constant 10201 : i32
// CHECK-NEXT:      %13 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %recv_buff_ex0 = memref.alloc() {"alignment" = 64 : i64} : memref<101x101xf32>
// CHECK-NEXT:      %14 = "memref.extract_aligned_pointer_as_index"(%recv_buff_ex0) : (memref<101x101xf32>) -> index
// CHECK-NEXT:      %15 = arith.index_cast %14 : index to i64
// CHECK-NEXT:      %recv_buff_ex0_ptr = "llvm.inttoptr"(%15) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %16 = arith.constant 10201 : i32
// CHECK-NEXT:      %17 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %send_buff_ex1 = memref.alloc() {"alignment" = 64 : i64} : memref<101x101xf32>
// CHECK-NEXT:      %18 = "memref.extract_aligned_pointer_as_index"(%send_buff_ex1) : (memref<101x101xf32>) -> index
// CHECK-NEXT:      %19 = arith.index_cast %18 : index to i64
// CHECK-NEXT:      %send_buff_ex1_ptr = "llvm.inttoptr"(%19) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %20 = arith.constant 10201 : i32
// CHECK-NEXT:      %21 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %recv_buff_ex1 = memref.alloc() {"alignment" = 64 : i64} : memref<101x101xf32>
// CHECK-NEXT:      %22 = "memref.extract_aligned_pointer_as_index"(%recv_buff_ex1) : (memref<101x101xf32>) -> index
// CHECK-NEXT:      %23 = arith.index_cast %22 : index to i64
// CHECK-NEXT:      %recv_buff_ex1_ptr = "llvm.inttoptr"(%23) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %24 = arith.constant 10201 : i32
// CHECK-NEXT:      %25 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %send_buff_ex2 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %26 = "memref.extract_aligned_pointer_as_index"(%send_buff_ex2) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %27 = arith.index_cast %26 : index to i64
// CHECK-NEXT:      %send_buff_ex2_ptr = "llvm.inttoptr"(%27) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %28 = arith.constant 5151 : i32
// CHECK-NEXT:      %29 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %recv_buff_ex2 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %30 = "memref.extract_aligned_pointer_as_index"(%recv_buff_ex2) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %31 = arith.index_cast %30 : index to i64
// CHECK-NEXT:      %recv_buff_ex2_ptr = "llvm.inttoptr"(%31) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %32 = arith.constant 5151 : i32
// CHECK-NEXT:      %33 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %send_buff_ex3 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %34 = "memref.extract_aligned_pointer_as_index"(%send_buff_ex3) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %35 = arith.index_cast %34 : index to i64
// CHECK-NEXT:      %send_buff_ex3_ptr = "llvm.inttoptr"(%35) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %36 = arith.constant 5151 : i32
// CHECK-NEXT:      %37 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %recv_buff_ex3 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %38 = "memref.extract_aligned_pointer_as_index"(%recv_buff_ex3) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %39 = arith.index_cast %38 : index to i64
// CHECK-NEXT:      %recv_buff_ex3_ptr = "llvm.inttoptr"(%39) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %40 = arith.constant 5151 : i32
// CHECK-NEXT:      %41 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %send_buff_ex4 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %42 = "memref.extract_aligned_pointer_as_index"(%send_buff_ex4) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %43 = arith.index_cast %42 : index to i64
// CHECK-NEXT:      %send_buff_ex4_ptr = "llvm.inttoptr"(%43) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %44 = arith.constant 5151 : i32
// CHECK-NEXT:      %45 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %recv_buff_ex4 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %46 = "memref.extract_aligned_pointer_as_index"(%recv_buff_ex4) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %47 = arith.index_cast %46 : index to i64
// CHECK-NEXT:      %recv_buff_ex4_ptr = "llvm.inttoptr"(%47) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %48 = arith.constant 5151 : i32
// CHECK-NEXT:      %49 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %send_buff_ex5 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %50 = "memref.extract_aligned_pointer_as_index"(%send_buff_ex5) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %51 = arith.index_cast %50 : index to i64
// CHECK-NEXT:      %send_buff_ex5_ptr = "llvm.inttoptr"(%51) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %52 = arith.constant 5151 : i32
// CHECK-NEXT:      %53 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %recv_buff_ex5 = memref.alloc() {"alignment" = 64 : i64} : memref<51x101xf32>
// CHECK-NEXT:      %54 = "memref.extract_aligned_pointer_as_index"(%recv_buff_ex5) : (memref<51x101xf32>) -> index
// CHECK-NEXT:      %55 = arith.index_cast %54 : index to i64
// CHECK-NEXT:      %recv_buff_ex5_ptr = "llvm.inttoptr"(%55) : (i64) -> !llvm.ptr
// CHECK-NEXT:      %56 = arith.constant 5151 : i32
// CHECK-NEXT:      %57 = arith.constant 1275069450 : i32
// CHECK-NEXT:      %58, %59, %60 = scf.for %time = %time_m to %2 step %step iter_args(%u_t0 = %u_vec0, %u_t1 = %u_vec1, %u_t2 = %u_vec2) -> (memref<55x105x105xf32>, memref<55x105x105xf32>, memref<55x105x105xf32>) {
// CHECK-NEXT:        %u_t1_storeview = "memref.subview"(%u_t1) <{"static_offsets" = array<i64: 2, 2, 2>, "static_sizes" = array<i64: 51, 101, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<55x105x105xf32>) -> memref<51x101x101xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        %u_t0_loadview = "memref.subview"(%u_t0) <{"static_offsets" = array<i64: 2, 2, 2>, "static_sizes" = array<i64: 53, 103, 103>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<55x105x105xf32>) -> memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        %61 = arith.constant 0 : i32
// CHECK-NEXT:        %62 = arith.constant 1 : i32
// CHECK-NEXT:        %63 = arith.divui %9, %62 : i32
// CHECK-NEXT:        %64 = arith.remui %9, %62 : i32
// CHECK-NEXT:        %65 = arith.constant 1 : i32
// CHECK-NEXT:        %66 = arith.divui %64, %65 : i32
// CHECK-NEXT:        %67 = arith.remui %64, %65 : i32
// CHECK-NEXT:        %68 = arith.constant 1 : i32
// CHECK-NEXT:        %69 = arith.divui %67, %68 : i32
// CHECK-NEXT:        %70 = arith.remui %67, %68 : i32
// CHECK-NEXT:        %71 = arith.constant 1 : i32
// CHECK-NEXT:        %72 = arith.addi %63, %71 : i32
// CHECK-NEXT:        %73 = arith.constant 2 : i32
// CHECK-NEXT:        %74 = arith.cmpi slt, %72, %73 : i32
// CHECK-NEXT:        %75 = arith.constant true
// CHECK-NEXT:        %76 = arith.constant true
// CHECK-NEXT:        %77 = arith.andi %74, %75 : i1
// CHECK-NEXT:        %78 = arith.andi %77, %76 : i1
// CHECK-NEXT:        %79 = arith.constant 1 : i32
// CHECK-NEXT:        %80 = arith.muli %79, %72 : i32
// CHECK-NEXT:        %81 = arith.addi %69, %80 : i32
// CHECK-NEXT:        %82 = arith.constant 1 : i32
// CHECK-NEXT:        %83 = arith.muli %82, %66 : i32
// CHECK-NEXT:        %84 = arith.addi %81, %83 : i32
// CHECK-NEXT:        %85 = arith.constant 0 : i32
// CHECK-NEXT:        %86 = arith.constant 6 : i32
// CHECK-NEXT:        %87 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %88 = arith.constant 4 : i64
// CHECK-NEXT:        %89 = arith.index_cast %85 : i32 to index
// CHECK-NEXT:        %90 = arith.index_cast %89 : index to i64
// CHECK-NEXT:        %91 = arith.muli %88, %90 : i64
// CHECK-NEXT:        %92 = arith.addi %91, %87 : i64
// CHECK-NEXT:        %93 = "llvm.inttoptr"(%92) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %94 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %95 = arith.constant 4 : i64
// CHECK-NEXT:        %96 = arith.index_cast %86 : i32 to index
// CHECK-NEXT:        %97 = arith.index_cast %96 : index to i64
// CHECK-NEXT:        %98 = arith.muli %95, %97 : i64
// CHECK-NEXT:        %99 = arith.addi %98, %94 : i64
// CHECK-NEXT:        %100 = "llvm.inttoptr"(%99) : (i64) -> !llvm.ptr
// CHECK-NEXT:        "scf.if"(%78) ({
// CHECK-NEXT:          %101 = "memref.subview"(%u_t0_loadview) <{"static_offsets" = array<i64: 50, 0, 0>, "static_sizes" = array<i64: 1, 101, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> memref<101x101xf32, strided<[105, 1], offset: 573512>>
// CHECK-NEXT:          "memref.copy"(%101, %send_buff_ex0) : (memref<101x101xf32, strided<[105, 1], offset: 573512>>, memref<101x101xf32>) -> ()
// CHECK-NEXT:          %102 = arith.constant 1140850688 : i32
// CHECK-NEXT:          %103 = func.call @MPI_Isend(%send_buff_ex0_ptr, %12, %13, %84, %61, %102, %93) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          %104 = arith.constant 1140850688 : i32
// CHECK-NEXT:          %105 = func.call @MPI_Irecv(%recv_buff_ex0_ptr, %16, %17, %84, %61, %104, %100) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %106 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%106, %93) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          %107 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%107, %100) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %108 = arith.constant 1 : i32
// CHECK-NEXT:        %109 = arith.divui %9, %108 : i32
// CHECK-NEXT:        %110 = arith.remui %9, %108 : i32
// CHECK-NEXT:        %111 = arith.constant 1 : i32
// CHECK-NEXT:        %112 = arith.divui %110, %111 : i32
// CHECK-NEXT:        %113 = arith.remui %110, %111 : i32
// CHECK-NEXT:        %114 = arith.constant 1 : i32
// CHECK-NEXT:        %115 = arith.divui %113, %114 : i32
// CHECK-NEXT:        %116 = arith.remui %113, %114 : i32
// CHECK-NEXT:        %117 = arith.constant -1 : i32
// CHECK-NEXT:        %118 = arith.addi %109, %117 : i32
// CHECK-NEXT:        %119 = arith.constant 0 : i32
// CHECK-NEXT:        %120 = arith.cmpi sge, %118, %119 : i32
// CHECK-NEXT:        %121 = arith.constant true
// CHECK-NEXT:        %122 = arith.constant true
// CHECK-NEXT:        %123 = arith.andi %120, %121 : i1
// CHECK-NEXT:        %124 = arith.andi %123, %122 : i1
// CHECK-NEXT:        %125 = arith.constant 1 : i32
// CHECK-NEXT:        %126 = arith.muli %125, %118 : i32
// CHECK-NEXT:        %127 = arith.addi %115, %126 : i32
// CHECK-NEXT:        %128 = arith.constant 1 : i32
// CHECK-NEXT:        %129 = arith.muli %128, %112 : i32
// CHECK-NEXT:        %130 = arith.addi %127, %129 : i32
// CHECK-NEXT:        %131 = arith.constant 1 : i32
// CHECK-NEXT:        %132 = arith.constant 7 : i32
// CHECK-NEXT:        %133 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %134 = arith.constant 4 : i64
// CHECK-NEXT:        %135 = arith.index_cast %131 : i32 to index
// CHECK-NEXT:        %136 = arith.index_cast %135 : index to i64
// CHECK-NEXT:        %137 = arith.muli %134, %136 : i64
// CHECK-NEXT:        %138 = arith.addi %137, %133 : i64
// CHECK-NEXT:        %139 = "llvm.inttoptr"(%138) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %140 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %141 = arith.constant 4 : i64
// CHECK-NEXT:        %142 = arith.index_cast %132 : i32 to index
// CHECK-NEXT:        %143 = arith.index_cast %142 : index to i64
// CHECK-NEXT:        %144 = arith.muli %141, %143 : i64
// CHECK-NEXT:        %145 = arith.addi %144, %140 : i64
// CHECK-NEXT:        %146 = "llvm.inttoptr"(%145) : (i64) -> !llvm.ptr
// CHECK-NEXT:        "scf.if"(%124) ({
// CHECK-NEXT:          %147 = "memref.subview"(%u_t0_loadview) <{"static_offsets" = array<i64: 0, 0, 0>, "static_sizes" = array<i64: 1, 101, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> memref<101x101xf32, strided<[105, 1], offset: 22262>>
// CHECK-NEXT:          "memref.copy"(%147, %send_buff_ex1) : (memref<101x101xf32, strided<[105, 1], offset: 22262>>, memref<101x101xf32>) -> ()
// CHECK-NEXT:          %148 = arith.constant 1140850688 : i32
// CHECK-NEXT:          %149 = func.call @MPI_Isend(%send_buff_ex1_ptr, %20, %21, %130, %61, %148, %139) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          %150 = arith.constant 1140850688 : i32
// CHECK-NEXT:          %151 = func.call @MPI_Irecv(%recv_buff_ex1_ptr, %24, %25, %130, %61, %150, %146) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %152 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%152, %139) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          %153 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%153, %146) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %154 = arith.constant 1 : i32
// CHECK-NEXT:        %155 = arith.divui %9, %154 : i32
// CHECK-NEXT:        %156 = arith.remui %9, %154 : i32
// CHECK-NEXT:        %157 = arith.constant 1 : i32
// CHECK-NEXT:        %158 = arith.divui %156, %157 : i32
// CHECK-NEXT:        %159 = arith.remui %156, %157 : i32
// CHECK-NEXT:        %160 = arith.constant 1 : i32
// CHECK-NEXT:        %161 = arith.divui %159, %160 : i32
// CHECK-NEXT:        %162 = arith.remui %159, %160 : i32
// CHECK-NEXT:        %163 = arith.constant true
// CHECK-NEXT:        %164 = arith.constant 1 : i32
// CHECK-NEXT:        %165 = arith.addi %158, %164 : i32
// CHECK-NEXT:        %166 = arith.constant 1 : i32
// CHECK-NEXT:        %167 = arith.cmpi slt, %165, %166 : i32
// CHECK-NEXT:        %168 = arith.constant true
// CHECK-NEXT:        %169 = arith.andi %163, %167 : i1
// CHECK-NEXT:        %170 = arith.andi %169, %168 : i1
// CHECK-NEXT:        %171 = arith.constant 1 : i32
// CHECK-NEXT:        %172 = arith.muli %171, %155 : i32
// CHECK-NEXT:        %173 = arith.addi %161, %172 : i32
// CHECK-NEXT:        %174 = arith.constant 1 : i32
// CHECK-NEXT:        %175 = arith.muli %174, %165 : i32
// CHECK-NEXT:        %176 = arith.addi %173, %175 : i32
// CHECK-NEXT:        %177 = arith.constant 2 : i32
// CHECK-NEXT:        %178 = arith.constant 8 : i32
// CHECK-NEXT:        %179 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %180 = arith.constant 4 : i64
// CHECK-NEXT:        %181 = arith.index_cast %177 : i32 to index
// CHECK-NEXT:        %182 = arith.index_cast %181 : index to i64
// CHECK-NEXT:        %183 = arith.muli %180, %182 : i64
// CHECK-NEXT:        %184 = arith.addi %183, %179 : i64
// CHECK-NEXT:        %185 = "llvm.inttoptr"(%184) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %186 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %187 = arith.constant 4 : i64
// CHECK-NEXT:        %188 = arith.index_cast %178 : i32 to index
// CHECK-NEXT:        %189 = arith.index_cast %188 : index to i64
// CHECK-NEXT:        %190 = arith.muli %187, %189 : i64
// CHECK-NEXT:        %191 = arith.addi %190, %186 : i64
// CHECK-NEXT:        %192 = "llvm.inttoptr"(%191) : (i64) -> !llvm.ptr
// CHECK-NEXT:        "scf.if"(%170) ({
// CHECK-NEXT:          %193 = "memref.subview"(%u_t0_loadview) <{"static_offsets" = array<i64: 0, 100, 0>, "static_sizes" = array<i64: 51, 1, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> memref<51x101xf32, strided<[11025, 1], offset: 32762>>
// CHECK-NEXT:          "memref.copy"(%193, %send_buff_ex2) : (memref<51x101xf32, strided<[11025, 1], offset: 32762>>, memref<51x101xf32>) -> ()
// CHECK-NEXT:          %194 = arith.constant 1140850688 : i32
// CHECK-NEXT:          %195 = func.call @MPI_Isend(%send_buff_ex2_ptr, %28, %29, %176, %61, %194, %185) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          %196 = arith.constant 1140850688 : i32
// CHECK-NEXT:          %197 = func.call @MPI_Irecv(%recv_buff_ex2_ptr, %32, %33, %176, %61, %196, %192) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %198 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%198, %185) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          %199 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%199, %192) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %200 = arith.constant 1 : i32
// CHECK-NEXT:        %201 = arith.divui %9, %200 : i32
// CHECK-NEXT:        %202 = arith.remui %9, %200 : i32
// CHECK-NEXT:        %203 = arith.constant 1 : i32
// CHECK-NEXT:        %204 = arith.divui %202, %203 : i32
// CHECK-NEXT:        %205 = arith.remui %202, %203 : i32
// CHECK-NEXT:        %206 = arith.constant 1 : i32
// CHECK-NEXT:        %207 = arith.divui %205, %206 : i32
// CHECK-NEXT:        %208 = arith.remui %205, %206 : i32
// CHECK-NEXT:        %209 = arith.constant true
// CHECK-NEXT:        %210 = arith.constant -1 : i32
// CHECK-NEXT:        %211 = arith.addi %204, %210 : i32
// CHECK-NEXT:        %212 = arith.constant 0 : i32
// CHECK-NEXT:        %213 = arith.cmpi sge, %211, %212 : i32
// CHECK-NEXT:        %214 = arith.constant true
// CHECK-NEXT:        %215 = arith.andi %209, %213 : i1
// CHECK-NEXT:        %216 = arith.andi %215, %214 : i1
// CHECK-NEXT:        %217 = arith.constant 1 : i32
// CHECK-NEXT:        %218 = arith.muli %217, %201 : i32
// CHECK-NEXT:        %219 = arith.addi %207, %218 : i32
// CHECK-NEXT:        %220 = arith.constant 1 : i32
// CHECK-NEXT:        %221 = arith.muli %220, %211 : i32
// CHECK-NEXT:        %222 = arith.addi %219, %221 : i32
// CHECK-NEXT:        %223 = arith.constant 3 : i32
// CHECK-NEXT:        %224 = arith.constant 9 : i32
// CHECK-NEXT:        %225 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %226 = arith.constant 4 : i64
// CHECK-NEXT:        %227 = arith.index_cast %223 : i32 to index
// CHECK-NEXT:        %228 = arith.index_cast %227 : index to i64
// CHECK-NEXT:        %229 = arith.muli %226, %228 : i64
// CHECK-NEXT:        %230 = arith.addi %229, %225 : i64
// CHECK-NEXT:        %231 = "llvm.inttoptr"(%230) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %232 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %233 = arith.constant 4 : i64
// CHECK-NEXT:        %234 = arith.index_cast %224 : i32 to index
// CHECK-NEXT:        %235 = arith.index_cast %234 : index to i64
// CHECK-NEXT:        %236 = arith.muli %233, %235 : i64
// CHECK-NEXT:        %237 = arith.addi %236, %232 : i64
// CHECK-NEXT:        %238 = "llvm.inttoptr"(%237) : (i64) -> !llvm.ptr
// CHECK-NEXT:        "scf.if"(%216) ({
// CHECK-NEXT:          %239 = "memref.subview"(%u_t0_loadview) <{"static_offsets" = array<i64: 0, 0, 0>, "static_sizes" = array<i64: 51, 1, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> memref<51x101xf32, strided<[11025, 1], offset: 22262>>
// CHECK-NEXT:          "memref.copy"(%239, %send_buff_ex3) : (memref<51x101xf32, strided<[11025, 1], offset: 22262>>, memref<51x101xf32>) -> ()
// CHECK-NEXT:          %240 = arith.constant 1140850688 : i32
// CHECK-NEXT:          %241 = func.call @MPI_Isend(%send_buff_ex3_ptr, %36, %37, %222, %61, %240, %231) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          %242 = arith.constant 1140850688 : i32
// CHECK-NEXT:          %243 = func.call @MPI_Irecv(%recv_buff_ex3_ptr, %40, %41, %222, %61, %242, %238) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %244 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%244, %231) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          %245 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%245, %238) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %246 = arith.constant 1 : i32
// CHECK-NEXT:        %247 = arith.divui %9, %246 : i32
// CHECK-NEXT:        %248 = arith.remui %9, %246 : i32
// CHECK-NEXT:        %249 = arith.constant 1 : i32
// CHECK-NEXT:        %250 = arith.divui %248, %249 : i32
// CHECK-NEXT:        %251 = arith.remui %248, %249 : i32
// CHECK-NEXT:        %252 = arith.constant 1 : i32
// CHECK-NEXT:        %253 = arith.divui %251, %252 : i32
// CHECK-NEXT:        %254 = arith.remui %251, %252 : i32
// CHECK-NEXT:        %255 = arith.constant true
// CHECK-NEXT:        %256 = arith.constant true
// CHECK-NEXT:        %257 = arith.constant 1 : i32
// CHECK-NEXT:        %258 = arith.addi %253, %257 : i32
// CHECK-NEXT:        %259 = arith.constant 1 : i32
// CHECK-NEXT:        %260 = arith.cmpi slt, %258, %259 : i32
// CHECK-NEXT:        %261 = arith.andi %255, %256 : i1
// CHECK-NEXT:        %262 = arith.andi %261, %260 : i1
// CHECK-NEXT:        %263 = arith.constant 1 : i32
// CHECK-NEXT:        %264 = arith.muli %263, %247 : i32
// CHECK-NEXT:        %265 = arith.addi %258, %264 : i32
// CHECK-NEXT:        %266 = arith.constant 1 : i32
// CHECK-NEXT:        %267 = arith.muli %266, %250 : i32
// CHECK-NEXT:        %268 = arith.addi %265, %267 : i32
// CHECK-NEXT:        %269 = arith.constant 4 : i32
// CHECK-NEXT:        %270 = arith.constant 10 : i32
// CHECK-NEXT:        %271 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %272 = arith.constant 4 : i64
// CHECK-NEXT:        %273 = arith.index_cast %269 : i32 to index
// CHECK-NEXT:        %274 = arith.index_cast %273 : index to i64
// CHECK-NEXT:        %275 = arith.muli %272, %274 : i64
// CHECK-NEXT:        %276 = arith.addi %275, %271 : i64
// CHECK-NEXT:        %277 = "llvm.inttoptr"(%276) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %278 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %279 = arith.constant 4 : i64
// CHECK-NEXT:        %280 = arith.index_cast %270 : i32 to index
// CHECK-NEXT:        %281 = arith.index_cast %280 : index to i64
// CHECK-NEXT:        %282 = arith.muli %279, %281 : i64
// CHECK-NEXT:        %283 = arith.addi %282, %278 : i64
// CHECK-NEXT:        %284 = "llvm.inttoptr"(%283) : (i64) -> !llvm.ptr
// CHECK-NEXT:        "scf.if"(%262) ({
// CHECK-NEXT:          %285 = "memref.subview"(%u_t0_loadview) <{"static_offsets" = array<i64: 0, 0, 100>, "static_sizes" = array<i64: 51, 101, 1>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> memref<51x101xf32, strided<[11025, 105], offset: 22362>>
// CHECK-NEXT:          "memref.copy"(%285, %send_buff_ex4) : (memref<51x101xf32, strided<[11025, 105], offset: 22362>>, memref<51x101xf32>) -> ()
// CHECK-NEXT:          %286 = arith.constant 1140850688 : i32
// CHECK-NEXT:          %287 = func.call @MPI_Isend(%send_buff_ex4_ptr, %44, %45, %268, %61, %286, %277) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          %288 = arith.constant 1140850688 : i32
// CHECK-NEXT:          %289 = func.call @MPI_Irecv(%recv_buff_ex4_ptr, %48, %49, %268, %61, %288, %284) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %290 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%290, %277) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          %291 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%291, %284) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %292 = arith.constant 1 : i32
// CHECK-NEXT:        %293 = arith.divui %9, %292 : i32
// CHECK-NEXT:        %294 = arith.remui %9, %292 : i32
// CHECK-NEXT:        %295 = arith.constant 1 : i32
// CHECK-NEXT:        %296 = arith.divui %294, %295 : i32
// CHECK-NEXT:        %297 = arith.remui %294, %295 : i32
// CHECK-NEXT:        %298 = arith.constant 1 : i32
// CHECK-NEXT:        %299 = arith.divui %297, %298 : i32
// CHECK-NEXT:        %300 = arith.remui %297, %298 : i32
// CHECK-NEXT:        %301 = arith.constant true
// CHECK-NEXT:        %302 = arith.constant true
// CHECK-NEXT:        %303 = arith.constant -1 : i32
// CHECK-NEXT:        %304 = arith.addi %299, %303 : i32
// CHECK-NEXT:        %305 = arith.constant 0 : i32
// CHECK-NEXT:        %306 = arith.cmpi sge, %304, %305 : i32
// CHECK-NEXT:        %307 = arith.andi %301, %302 : i1
// CHECK-NEXT:        %308 = arith.andi %307, %306 : i1
// CHECK-NEXT:        %309 = arith.constant 1 : i32
// CHECK-NEXT:        %310 = arith.muli %309, %293 : i32
// CHECK-NEXT:        %311 = arith.addi %304, %310 : i32
// CHECK-NEXT:        %312 = arith.constant 1 : i32
// CHECK-NEXT:        %313 = arith.muli %312, %296 : i32
// CHECK-NEXT:        %314 = arith.addi %311, %313 : i32
// CHECK-NEXT:        %315 = arith.constant 5 : i32
// CHECK-NEXT:        %316 = arith.constant 11 : i32
// CHECK-NEXT:        %317 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %318 = arith.constant 4 : i64
// CHECK-NEXT:        %319 = arith.index_cast %315 : i32 to index
// CHECK-NEXT:        %320 = arith.index_cast %319 : index to i64
// CHECK-NEXT:        %321 = arith.muli %318, %320 : i64
// CHECK-NEXT:        %322 = arith.addi %321, %317 : i64
// CHECK-NEXT:        %323 = "llvm.inttoptr"(%322) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %324 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
// CHECK-NEXT:        %325 = arith.constant 4 : i64
// CHECK-NEXT:        %326 = arith.index_cast %316 : i32 to index
// CHECK-NEXT:        %327 = arith.index_cast %326 : index to i64
// CHECK-NEXT:        %328 = arith.muli %325, %327 : i64
// CHECK-NEXT:        %329 = arith.addi %328, %324 : i64
// CHECK-NEXT:        %330 = "llvm.inttoptr"(%329) : (i64) -> !llvm.ptr
// CHECK-NEXT:        "scf.if"(%308) ({
// CHECK-NEXT:          %331 = "memref.subview"(%u_t0_loadview) <{"static_offsets" = array<i64: 0, 0, 0>, "static_sizes" = array<i64: 51, 101, 1>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> memref<51x101xf32, strided<[11025, 105], offset: 22262>>
// CHECK-NEXT:          "memref.copy"(%331, %send_buff_ex5) : (memref<51x101xf32, strided<[11025, 105], offset: 22262>>, memref<51x101xf32>) -> ()
// CHECK-NEXT:          %332 = arith.constant 1140850688 : i32
// CHECK-NEXT:          %333 = func.call @MPI_Isend(%send_buff_ex5_ptr, %52, %53, %314, %61, %332, %323) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          %334 = arith.constant 1140850688 : i32
// CHECK-NEXT:          %335 = func.call @MPI_Irecv(%recv_buff_ex5_ptr, %56, %57, %314, %61, %334, %330) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          %336 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%336, %323) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          %337 = arith.constant 738197504 : i32
// CHECK-NEXT:          "llvm.store"(%337, %330) <{"ordering" = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %338 = arith.constant 1 : i64
// CHECK-NEXT:        %339 = "llvm.inttoptr"(%338) : (i64) -> !llvm.ptr
// CHECK-NEXT:        %340 = func.call @MPI_Waitall(%3, %4, %339) : (i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:        "scf.if"(%78) ({
// CHECK-NEXT:          %341 = "memref.subview"(%u_t0_loadview) <{"static_offsets" = array<i64: 51, 0, 0>, "static_sizes" = array<i64: 1, 101, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> memref<101x101xf32, strided<[105, 1], offset: 584537>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex0, %341) : (memref<101x101xf32>, memref<101x101xf32, strided<[105, 1], offset: 584537>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%124) ({
// CHECK-NEXT:          %342 = "memref.subview"(%u_t0_loadview) <{"static_offsets" = array<i64: -1, 0, 0>, "static_sizes" = array<i64: 1, 101, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> memref<101x101xf32, strided<[105, 1], offset: 11237>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex1, %342) : (memref<101x101xf32>, memref<101x101xf32, strided<[105, 1], offset: 11237>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%170) ({
// CHECK-NEXT:          %343 = "memref.subview"(%u_t0_loadview) <{"static_offsets" = array<i64: 0, 101, 0>, "static_sizes" = array<i64: 51, 1, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> memref<51x101xf32, strided<[11025, 1], offset: 32867>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex2, %343) : (memref<51x101xf32>, memref<51x101xf32, strided<[11025, 1], offset: 32867>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%216) ({
// CHECK-NEXT:          %344 = "memref.subview"(%u_t0_loadview) <{"static_offsets" = array<i64: 0, -1, 0>, "static_sizes" = array<i64: 51, 1, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> memref<51x101xf32, strided<[11025, 1], offset: 22157>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex3, %344) : (memref<51x101xf32>, memref<51x101xf32, strided<[11025, 1], offset: 22157>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%262) ({
// CHECK-NEXT:          %345 = "memref.subview"(%u_t0_loadview) <{"static_offsets" = array<i64: 0, 0, 101>, "static_sizes" = array<i64: 51, 101, 1>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> memref<51x101xf32, strided<[11025, 105], offset: 22363>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex4, %345) : (memref<51x101xf32>, memref<51x101xf32, strided<[11025, 105], offset: 22363>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        "scf.if"(%308) ({
// CHECK-NEXT:          %346 = "memref.subview"(%u_t0_loadview) <{"static_offsets" = array<i64: 0, 0, -1>, "static_sizes" = array<i64: 51, 101, 1>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>) -> memref<51x101xf32, strided<[11025, 105], offset: 22261>>
// CHECK-NEXT:          "memref.copy"(%recv_buff_ex5, %346) : (memref<51x101xf32>, memref<51x101xf32, strided<[11025, 105], offset: 22261>>) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }, {
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (i1) -> ()
// CHECK-NEXT:        %u_t2_loadview = "memref.subview"(%u_t2) <{"static_offsets" = array<i64: 2, 2, 2>, "static_sizes" = array<i64: 51, 101, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<55x105x105xf32>) -> memref<51x101x101xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        %347 = arith.constant 0 : index
// CHECK-NEXT:        %348 = arith.constant 0 : index
// CHECK-NEXT:        %349 = arith.constant 0 : index
// CHECK-NEXT:        %350 = arith.constant 1 : index
// CHECK-NEXT:        %351 = arith.constant 1 : index
// CHECK-NEXT:        %352 = arith.constant 1 : index
// CHECK-NEXT:        %353 = arith.constant 51 : index
// CHECK-NEXT:        %354 = arith.constant 101 : index
// CHECK-NEXT:        %355 = arith.constant 101 : index
// CHECK-NEXT:        %356 = arith.constant 0 : index
// CHECK-NEXT:        %357 = arith.constant 64 : index
// CHECK-NEXT:        %358 = arith.constant 64 : index
// CHECK-NEXT:        %359 = arith.muli %350, %357 : index
// CHECK-NEXT:        %360 = arith.muli %351, %358 : index
// CHECK-NEXT:        "scf.parallel"(%347, %348, %353, %354, %359, %360) <{"operandSegmentSizes" = array<i32: 2, 2, 2, 0>}> ({
// CHECK-NEXT:        ^0(%361 : index, %362 : index):
// CHECK-NEXT:          %363 = "affine.min"(%357, %353, %361) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:          %364 = "affine.min"(%358, %354, %362) <{"map" = affine_map<(d0, d1, d2) -> (d0, (d1 + (d2 * -1)))>}> : (index, index, index) -> index
// CHECK-NEXT:          "scf.parallel"(%356, %356, %349, %363, %364, %355, %350, %351, %352) <{"operandSegmentSizes" = array<i32: 3, 3, 3, 0>}> ({
// CHECK-NEXT:          ^1(%365 : index, %366 : index, %367 : index):
// CHECK-NEXT:            %368 = arith.addi %361, %365 : index
// CHECK-NEXT:            %369 = arith.addi %362, %366 : index
// CHECK-NEXT:            %dt = arith.constant 1.000000e-04 : f32
// CHECK-NEXT:            %370 = arith.constant 2 : i64
// CHECK-NEXT:            %371 = "math.fpowi"(%dt, %370) : (f32, i64) -> f32
// CHECK-NEXT:            %372 = arith.constant -1 : i64
// CHECK-NEXT:            %dt_1 = arith.constant 1.000000e-04 : f32
// CHECK-NEXT:            %373 = arith.constant -2 : i64
// CHECK-NEXT:            %374 = "math.fpowi"(%dt_1, %373) : (f32, i64) -> f32
// CHECK-NEXT:            %375 = memref.load %u_t2_loadview[%368, %369, %367] : memref<51x101x101xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %376 = arith.mulf %374, %375 : f32
// CHECK-NEXT:            %377 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:            %dt_2 = arith.constant 1.000000e-04 : f32
// CHECK-NEXT:            %378 = arith.constant -2 : i64
// CHECK-NEXT:            %379 = "math.fpowi"(%dt_2, %378) : (f32, i64) -> f32
// CHECK-NEXT:            %380 = memref.load %u_t0_loadview[%368, %369, %367] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %381 = arith.mulf %377, %379 : f32
// CHECK-NEXT:            %382 = arith.mulf %381, %380 : f32
// CHECK-NEXT:            %383 = arith.addf %376, %382 : f32
// CHECK-NEXT:            %384 = arith.sitofp %372 : i64 to f32
// CHECK-NEXT:            %385 = arith.mulf %384, %383 : f32
// CHECK-NEXT:            %h_x = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %386 = arith.constant -2 : i64
// CHECK-NEXT:            %387 = "math.fpowi"(%h_x, %386) : (f32, i64) -> f32
// CHECK-NEXT:            %388 = arith.constant -1 : index
// CHECK-NEXT:            %389 = arith.addi %368, %388 : index
// CHECK-NEXT:            %390 = memref.load %u_t0_loadview[%389, %369, %367] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %391 = arith.mulf %387, %390 : f32
// CHECK-NEXT:            %h_x_1 = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %392 = arith.constant -2 : i64
// CHECK-NEXT:            %393 = "math.fpowi"(%h_x_1, %392) : (f32, i64) -> f32
// CHECK-NEXT:            %394 = arith.constant 1 : index
// CHECK-NEXT:            %395 = arith.addi %368, %394 : index
// CHECK-NEXT:            %396 = memref.load %u_t0_loadview[%395, %369, %367] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %397 = arith.mulf %393, %396 : f32
// CHECK-NEXT:            %398 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:            %h_x_2 = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %399 = arith.constant -2 : i64
// CHECK-NEXT:            %400 = "math.fpowi"(%h_x_2, %399) : (f32, i64) -> f32
// CHECK-NEXT:            %401 = memref.load %u_t0_loadview[%368, %369, %367] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %402 = arith.mulf %398, %400 : f32
// CHECK-NEXT:            %403 = arith.mulf %402, %401 : f32
// CHECK-NEXT:            %404 = arith.addf %391, %397 : f32
// CHECK-NEXT:            %405 = arith.addf %404, %403 : f32
// CHECK-NEXT:            %h_y = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %406 = arith.constant -2 : i64
// CHECK-NEXT:            %407 = "math.fpowi"(%h_y, %406) : (f32, i64) -> f32
// CHECK-NEXT:            %408 = arith.constant -1 : index
// CHECK-NEXT:            %409 = arith.addi %369, %408 : index
// CHECK-NEXT:            %410 = memref.load %u_t0_loadview[%368, %409, %367] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %411 = arith.mulf %407, %410 : f32
// CHECK-NEXT:            %h_y_1 = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %412 = arith.constant -2 : i64
// CHECK-NEXT:            %413 = "math.fpowi"(%h_y_1, %412) : (f32, i64) -> f32
// CHECK-NEXT:            %414 = arith.constant 1 : index
// CHECK-NEXT:            %415 = arith.addi %369, %414 : index
// CHECK-NEXT:            %416 = memref.load %u_t0_loadview[%368, %415, %367] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %417 = arith.mulf %413, %416 : f32
// CHECK-NEXT:            %418 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:            %h_y_2 = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %419 = arith.constant -2 : i64
// CHECK-NEXT:            %420 = "math.fpowi"(%h_y_2, %419) : (f32, i64) -> f32
// CHECK-NEXT:            %421 = memref.load %u_t0_loadview[%368, %369, %367] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %422 = arith.mulf %418, %420 : f32
// CHECK-NEXT:            %423 = arith.mulf %422, %421 : f32
// CHECK-NEXT:            %424 = arith.addf %411, %417 : f32
// CHECK-NEXT:            %425 = arith.addf %424, %423 : f32
// CHECK-NEXT:            %h_z = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %426 = arith.constant -2 : i64
// CHECK-NEXT:            %427 = "math.fpowi"(%h_z, %426) : (f32, i64) -> f32
// CHECK-NEXT:            %428 = arith.constant -1 : index
// CHECK-NEXT:            %429 = arith.addi %367, %428 : index
// CHECK-NEXT:            %430 = memref.load %u_t0_loadview[%368, %369, %429] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %431 = arith.mulf %427, %430 : f32
// CHECK-NEXT:            %h_z_1 = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %432 = arith.constant -2 : i64
// CHECK-NEXT:            %433 = "math.fpowi"(%h_z_1, %432) : (f32, i64) -> f32
// CHECK-NEXT:            %434 = arith.constant 1 : index
// CHECK-NEXT:            %435 = arith.addi %367, %434 : index
// CHECK-NEXT:            %436 = memref.load %u_t0_loadview[%368, %369, %435] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %437 = arith.mulf %433, %436 : f32
// CHECK-NEXT:            %438 = arith.constant -2.000000e+00 : f32
// CHECK-NEXT:            %h_z_2 = arith.constant 1.000000e-02 : f32
// CHECK-NEXT:            %439 = arith.constant -2 : i64
// CHECK-NEXT:            %440 = "math.fpowi"(%h_z_2, %439) : (f32, i64) -> f32
// CHECK-NEXT:            %441 = memref.load %u_t0_loadview[%368, %369, %367] : memref<53x103x103xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            %442 = arith.mulf %438, %440 : f32
// CHECK-NEXT:            %443 = arith.mulf %442, %441 : f32
// CHECK-NEXT:            %444 = arith.addf %431, %437 : f32
// CHECK-NEXT:            %445 = arith.addf %444, %443 : f32
// CHECK-NEXT:            %446 = arith.addf %385, %405 : f32
// CHECK-NEXT:            %447 = arith.addf %446, %425 : f32
// CHECK-NEXT:            %448 = arith.addf %447, %445 : f32
// CHECK-NEXT:            %449 = arith.mulf %371, %448 : f32
// CHECK-NEXT:            memref.store %449, %u_t1_storeview[%368, %369, %367] : memref<51x101x101xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          }) : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        }) : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:        %u_t1_temp = "memref.subview"(%u_t1) <{"static_offsets" = array<i64: 2, 2, 2>, "static_sizes" = array<i64: 51, 101, 101>, "static_strides" = array<i64: 1, 1, 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (memref<55x105x105xf32>) -> memref<51x101x101xf32, strided<[11025, 105, 1], offset: 22262>>
// CHECK-NEXT:        scf.yield %u_t1, %u_t2, %u_t0 : memref<55x105x105xf32>, memref<55x105x105xf32>, memref<55x105x105xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %450 = func.call @timer_end(%0) : (f64) -> f64
// CHECK-NEXT:      "llvm.store"(%450, %timers) <{"ordering" = 0 : i64}> : (f64, !llvm.ptr) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @timer_start() -> f64
// CHECK-NEXT:    func.func private @timer_end(f64) -> f64
// CHECK-NEXT:    func.func private @MPI_Comm_rank(i32, !llvm.ptr) -> i32
// CHECK-NEXT:    func.func private @MPI_Isend(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    func.func private @MPI_Irecv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    func.func private @MPI_Waitall(i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:  }