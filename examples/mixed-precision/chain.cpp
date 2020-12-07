#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "openacc.h"
#include <assert.h> 
#include <cuda_fp16.h>


struct dataobj
{
  void *restrict data;
  int * size;
  int * npsize;
  int * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
} ;

struct profiler
{
  double section0;
} ;

typedef unsigned uint;

union FP32
{
    uint u;
    float f;
    struct
    {
        uint Mantissa : 23;
        uint Exponent : 8;
        uint Sign : 1;
    };
};

union FP16
{
    unsigned short u;
    struct
    {
        uint Mantissa : 10;
        uint Exponent : 5;
        uint Sign : 1;
    };
};

// Approximate solution. This is faster but converts some sNaNs to
// infinity and doesn't round correctly. Handle with care.
// Approximate solution. This is faster but converts some sNaNs to
// infinity and doesn't round correctly. Handle with care.
static half approx_float_to_half(float fl)
{
    FP32 f32infty = { 255 << 23 };
    FP32 f16max = { (127 + 16) << 23 };
    FP32 magic = { 15 << 23 };
    FP32 expinf = { (255 ^ 31) << 23 };
    uint sign_mask = 0x80000000u;
    FP16 o = { 0 };

    FP32 f = *((FP32*)&fl);

    uint sign = f.u & sign_mask;
    f.u ^= sign;

    if (!(f.f < f32infty.u)) // Inf or NaN
        o.u = f.u ^ expinf.u;
    else
    {
        if (f.f > f16max.f) f.f = f16max.f;
        f.f *= magic.f;
    }

    o.u = f.u >> 13; // Take the mantissa bits
    o.u |= sign >> 16;
    return *((half*)&o);
}

// from half->float code - just for verification.
static float half_to_float(half hf)
{
    FP16 h = *((FP16*)&hf);

    static const FP32 magic = { 113 << 23 };
    static const uint shifted_exp = 0x7c00 << 13; // exponent mask after shift
    FP32 o;

    o.u = (h.u & 0x7fff) << 13;     // exponent/mantissa bits
    uint exp = shifted_exp & o.u;   // just the exponent
    o.u += (127 - 15) << 23;        // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) // Inf/NaN?
        o.u += (128 - 16) << 23;    // extra exp adjust
    else if (exp == 0) // Zero/Denormal?
    {
        o.u += 1 << 23;             // extra exp adjust
        o.f -= magic.f;             // renormalize
    }

    o.u |= (h.u & 0x8000) << 16;    // sign bit
    return o.f;
}


extern "C" int Kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, struct dataobj *restrict D_vec, struct dataobj *restrict E_vec, struct dataobj *restrict F_vec, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, const int l_M, const int l_m, struct profiler * timers);

// dynamically allocate 2D array 
half** Make2DhalfArray(int arraySizeX, int arraySizeY) {
    half** theArray;
    theArray = (half**) malloc(arraySizeX*sizeof(half*));
    for (int i = 0; i < arraySizeX; i++)
        theArray[i] = (half*) malloc(arraySizeY*sizeof(half));
    return theArray;
} 

int Kernel(struct dataobj *restrict A_vec, struct dataobj *restrict B_vec, struct dataobj *restrict C_vec, struct dataobj *restrict D_vec, struct dataobj *restrict E_vec, struct dataobj *restrict F_vec, const int i_M, const int i_m, const int j_M, const int j_m, const int k_M, const int k_m, const int l_M, const int l_m, struct profiler * timers)
{
  float (*restrict A)[A_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[A_vec->size[1]]) A_vec->data;
  float (*restrict B)[B_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[B_vec->size[1]]) B_vec->data;
  float (*restrict C)[C_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[C_vec->size[1]]) C_vec->data;
  float (*restrict D)[D_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[D_vec->size[1]]) D_vec->data;
  float (*restrict E)[E_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[E_vec->size[1]]) E_vec->data;
  float (*restrict F)[F_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[F_vec->size[1]]) F_vec->data;

  #pragma acc enter data copyin(D[0:D_vec->size[0]][0:D_vec->size[1]])
  #pragma acc enter data copyin(F[0:F_vec->size[0]][0:F_vec->size[1]])
  #pragma acc enter data copyin(A[0:A_vec->size[0]][0:A_vec->size[1]])
  #pragma acc enter data copyin(B[0:B_vec->size[0]][0:B_vec->size[1]])
  #pragma acc enter data copyin(C[0:C_vec->size[0]][0:C_vec->size[1]])
  #pragma acc enter data copyin(E[0:E_vec->size[0]][0:E_vec->size[1]])

  half **a16 = Make2DhalfArray(A_vec->size[1], A_vec->size[1]);
  half **b16 = Make2DhalfArray(A_vec->size[1], A_vec->size[1]);
  half **c16 = Make2DhalfArray(A_vec->size[1], A_vec->size[1]);
  half **d16 = Make2DhalfArray(A_vec->size[1], A_vec->size[1]);  
  half **e16 = Make2DhalfArray(A_vec->size[1], A_vec->size[1]);
  half **f16 = Make2DhalfArray(A_vec->size[1], A_vec->size[1]);
  
  #pragma acc enter data create(d16[0:D_vec->size[0]][0:D_vec->size[1]])
  #pragma acc enter data create(f16[0:F_vec->size[0]][0:F_vec->size[1]])
  #pragma acc enter data create(a16[0:A_vec->size[0]][0:A_vec->size[1]])
  #pragma acc enter data create(b16[0:B_vec->size[0]][0:B_vec->size[1]])
  #pragma acc enter data create(c16[0:C_vec->size[0]][0:C_vec->size[1]])
  #pragma acc enter data create(e16[0:E_vec->size[0]][0:E_vec->size[1]])


  //#pragma acc enter data copyin(d16[0:D_vec->size[0]][0:D_vec->size[1]])
  //#pragma acc enter data copyin(f16[0:F_vec->size[0]][0:F_vec->size[1]])
  //#pragma acc enter data copyin(a16[0:A_vec->size[0]][0:A_vec->size[1]])
  //#pragma acc enter data copyin(b16[0:B_vec->size[0]][0:B_vec->size[1]])
  //#pragma acc enter data copyin(c16[0:C_vec->size[0]][0:C_vec->size[1]])
  //#pragma acc enter data copyin(e16[0:E_vec->size[0]][0:E_vec->size[1]])
 
  //#pragma acc parallel loop collapse(2)
  for (int i = i_m; i <= i_M; i += 1)
  {
    for (int j = j_m; j <= j_M; j += 1)
    {
      a16[i][j] = approx_float_to_half( A[i][j]);
      b16[i][j] = approx_float_to_half( B[i][j]);
      c16[i][j] = approx_float_to_half( C[i][j]);
      d16[i][j] = approx_float_to_half( D[i][j]);
      e16[i][j] = approx_float_to_half( E[i][j]); 
      f16[i][j] = approx_float_to_half( F[i][j]);
    }
  }
 
  struct timeval start_section0, end_section0;
  gettimeofday(&start_section0, NULL);
  /* Begin section0 */
  #pragma acc parallel loop collapse(1)
  for (int i = i_m; i <= i_M; i += 1)
  {
    for (int j = j_m; j <= j_M; j += 1)
    {
      for (int k = k_m; k <= k_M; k += 1)
      {
        //D[i][k] += A[i][j]*B[j][k] + A[i][j]*C[j][k];
        d16[i][k] += approx_float_to_half(a16[i][j]*b16[j][k] + a16[i][j]*c16[j][k]);
       }
    }
    for (int k = k_m; k <= k_M; k += 1)
    {
      for (int l = l_m; l <= l_M; l += 1)
      {
       // F[i][l] += D[i][k]*E[k][l];
       f16[i][l] += d16[i][k]*e16[k][l];
      }
    }
  }
  /* End section0 */
  gettimeofday(&end_section0, NULL);
  timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;

  #pragma acc exit data copyout(D[0:D_vec->size[0]][0:D_vec->size[1]])
  #pragma acc exit data delete(D[0:D_vec->size[0]][0:D_vec->size[1]])
  #pragma acc exit data copyout(F[0:F_vec->size[0]][0:F_vec->size[1]])
  #pragma acc exit data delete(F[0:F_vec->size[0]][0:F_vec->size[1]])
  #pragma acc exit data delete(A[0:A_vec->size[0]][0:A_vec->size[1]])
  #pragma acc exit data delete(B[0:B_vec->size[0]][0:B_vec->size[1]])
  #pragma acc exit data delete(C[0:C_vec->size[0]][0:C_vec->size[1]])
  #pragma acc exit data delete(E[0:E_vec->size[0]][0:E_vec->size[1]])
  return 0;
}







/* Backdoor edit at Mon Dec  7 15:59:27 2020*/ 
/* Backdoor edit at Mon Dec  7 16:00:01 2020*/ 
/* Backdoor edit at Mon Dec  7 16:04:08 2020*/ 
/* Backdoor edit at Mon Dec  7 16:04:31 2020*/ 
/* Backdoor edit at Mon Dec  7 16:05:34 2020*/ 
/* Backdoor edit at Mon Dec  7 16:07:35 2020*/ 
/* Backdoor edit at Mon Dec  7 16:09:38 2020*/ 
/* Backdoor edit at Mon Dec  7 16:26:50 2020*/ 
/* Backdoor edit at Mon Dec  7 16:29:09 2020*/ 
/* Backdoor edit at Mon Dec  7 16:32:28 2020*/ 
/* Backdoor edit at Mon Dec  7 16:33:04 2020*/ 
/* Backdoor edit at Mon Dec  7 16:33:33 2020*/ 
/* Backdoor edit at Mon Dec  7 16:34:13 2020*/ 
/* Backdoor edit at Mon Dec  7 16:35:41 2020*/ 
/* Backdoor edit at Mon Dec  7 16:52:31 2020*/ 
/* Backdoor edit at Mon Dec  7 17:00:16 2020*/ 
/* Backdoor edit at Mon Dec  7 17:03:37 2020*/ 
/* Backdoor edit at Mon Dec  7 17:03:55 2020*/ 
/* Backdoor edit at Mon Dec  7 17:04:12 2020*/ 
/* Backdoor edit at Mon Dec  7 17:09:33 2020*/ 
/* Backdoor edit at Mon Dec  7 17:14:22 2020*/ 
/* Backdoor edit at Mon Dec  7 17:14:54 2020*/ 
/* Backdoor edit at Mon Dec  7 17:15:04 2020*/ 
/* Backdoor edit at Mon Dec  7 17:15:33 2020*/ 
/* Backdoor edit at Mon Dec  7 17:16:09 2020*/ 
/* Backdoor edit at Mon Dec  7 17:17:22 2020*/ 
/* Backdoor edit at Mon Dec  7 17:18:41 2020*/ 
/* Backdoor edit at Mon Dec  7 17:20:07 2020*/ 
/* Backdoor edit at Mon Dec  7 17:22:21 2020*/ 
