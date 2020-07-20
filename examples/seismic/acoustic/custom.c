#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

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


int Kernel(const float h_x, const float h_y, const float h_z, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict save_src_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict u2_vec, const int sp_zi_m, const int time_M, const int time_m, struct profiler * timers, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m)
{
  int (*restrict nnz_sp_source_mask)[nnz_sp_source_mask_vec->size[1]] __attribute__ ((aligned (64))) = (int (*)[nnz_sp_source_mask_vec->size[1]]) nnz_sp_source_mask_vec->data;
  float (*restrict save_src)[save_src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[save_src_vec->size[1]]) save_src_vec->data;
  int (*restrict source_id)[source_id_vec->size[1]][source_id_vec->size[2]] __attribute__ ((aligned (64))) = (int (*)[source_id_vec->size[1]][source_id_vec->size[2]]) source_id_vec->data;
  int (*restrict source_mask)[source_mask_vec->size[1]][source_mask_vec->size[2]] __attribute__ ((aligned (64))) = (int (*)[source_mask_vec->size[1]][source_mask_vec->size[2]]) source_mask_vec->data;
  int (*restrict sp_source_mask)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]] __attribute__ ((aligned (64))) = (int (*)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]]) sp_source_mask_vec->data;
  float (*restrict u2)[u2_vec->size[1]][u2_vec->size[2]][u2_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u2_vec->size[1]][u2_vec->size[2]][u2_vec->size[3]]) u2_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
#pragma omp parallel num_threads(6)
    {
#pragma omp for collapse(2) schedule(dynamic, 1)
      for (int x = x_m; x <= x_M; x += 1)
      {
        for (int y = y_m; y <= y_M; y += 1)
        {
#pragma omp simd aligned(u2 : 32)
          for (int z = z_m; z <= z_M; z += 1)
          {
            float r1 = -2.0F * u2[t0][x + 2][y + 2][z + 2];
            u2[t1][x + 2][y + 2][z + 2] = 1.0e-1F + (r1 + u2[t0][x + 2][y + 2][z + 1] + u2[t0][x + 2][y + 2][z + 3]) / ((h_z * h_z)) + (r1 + u2[t0][x + 2][y + 1][z + 2] + u2[t0][x + 2][y + 3][z + 2]) / ((h_y * h_y)) + (r1 + u2[t0][x + 1][y + 2][z + 2] + u2[t0][x + 3][y + 2][z + 2]) / ((h_x * h_x));
          }
#pragma omp simd aligned(u2 : 8)
          for (int sp_zi = sp_zi_m; sp_zi <= nnz_sp_source_mask[x + 1][y + 1]; sp_zi += 1)
          {
            float r0 = save_src[time][source_id[x + 1][y + 1][sp_source_mask[x + 1][y + 1][sp_zi] + 1]] * source_mask[x + 1][y + 1][sp_source_mask[x + 1][y + 1][sp_zi] + 1];
            u2[t1][x + 2][y + 2][sp_source_mask[x + 1][y + 1][sp_zi] + 2] += r0;
          }
        }
      }
    }
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
  }
  return 0;
}
/* Backdoor edit at Thu Jun 25 20:33:18 2020*/ 
/* Backdoor edit at Thu Jun 25 20:33:37 2020*/ 
/* Backdoor edit at Thu Jun 25 20:34:13 2020*/ 
/* Backdoor edit at Thu Jun 25 20:35:13 2020*/ 
/* Backdoor edit at Thu Jun 25 20:35:43 2020*/ 
/* Backdoor edit at Thu Jun 25 20:36:12 2020*/ 
/* Backdoor edit at Thu Jun 25 20:37:27 2020*/ 
/* Backdoor edit at Thu Jun 25 20:43:46 2020*/ 
/* Backdoor edit at Thu Jun 25 20:44:41 2020*/ 
/* Backdoor edit at Thu Jun 25 20:44:57 2020*/ 
/* Backdoor edit at Thu Jun 25 20:45:43 2020*/ 
/* Backdoor edit at Thu Jun 25 20:47:26 2020*/ 
/* Backdoor edit at Thu Jun 25 20:48:55 2020*/ 
