#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include <stdio.h>
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

struct dataobj
{
  void *restrict data;
  int *size;
  int *npsize;
  int *dsize;
  int *hsize;
  int *hofs;
  int *oofs;
};

struct profiler
{
  double section0;
};

int Kernel(const float h_x, const float h_y, const float h_z, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict save_src_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict u2_vec, const int sp_zi_m, const int time_M, const int time_m, struct profiler *timers, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m)
{
  int(*restrict nnz_sp_source_mask)[nnz_sp_source_mask_vec->size[1]] __attribute__((aligned(64))) = (int(*)[nnz_sp_source_mask_vec->size[1]])nnz_sp_source_mask_vec->data;
  float(*restrict save_src)[save_src_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_vec->size[1]])save_src_vec->data;
  int(*restrict source_id)[source_id_vec->size[1]][source_id_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_id_vec->size[1]][source_id_vec->size[2]])source_id_vec->data;
  int(*restrict source_mask)[source_mask_vec->size[1]][source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_mask_vec->size[1]][source_mask_vec->size[2]])source_mask_vec->data;
  int(*restrict sp_source_mask)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]])sp_source_mask_vec->data;
  float(*restrict u2)[u2_vec->size[1]][u2_vec->size[2]][u2_vec->size[3]] __attribute__((aligned(64))) = (float(*)[u2_vec->size[1]][u2_vec->size[2]][u2_vec->size[3]])u2_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  struct timeval start_section0, end_section0;
  gettimeofday(&start_section0, NULL);
  int xb_size = 64;
  int yb_size = 64; // to fix as 8/16 etc
  int sf = 16;
  //int t_blk_size = time_M - time_m ;
  int t_blk_size = 10 * (time_M - time_m);
  //printf("Global time loop to timesteps = %d \n", time_M - time_m +1 );
  for (int t_blk = time_m; t_blk < sf * (time_M - time_m); t_blk += sf * t_blk_size) // for each t block
  //int t_blk = time_m;
  {
    for (int xb = x_m; xb <= (x_M + sf * (time_M - time_m)); xb += xb_size + 1)
    {
      //printf(" Change of xblock %d \n", xb);
      for (int yb = y_m; yb <= (y_M + sf * (time_M - time_m)); yb += yb_size + 1)
      {
        //printf("----y0_blk0 loop from y0_blk0 = %d to %d \n", y_m, (y_M + sf * (time_M - time_m)));
        for (int time = t_blk, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3); time <= 1 + min(t_blk + t_blk_size, sf * (time_M - time_m)); time += sf, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3))
        {

          int x0_blk0_size = 32;
          int y0_blk0_size = 32;
          /* Begin section0 */
#pragma omp parallel num_threads(6)
          {
#pragma omp for collapse(2) schedule(dynamic, 1)
            for (int x0_blk0 = max((x_m + time), xb); x0_blk0 <= min((x_M + time), (xb + xb_size)); x0_blk0 += xb_size)
            {
              for (int y0_blk0 = max((y_m + time), yb); y0_blk0 <= min((y_M + time), (yb + yb_size)); y0_blk0 += yb_size)
              {
                for (int x = x0_blk0; x <= min(min((x_M + time), (xb + xb_size)), (x0_blk0 + xb_size - 1)); x++)
                {
                  for (int y = y0_blk0; y <= min(min((y_M + time), (yb + yb_size)), (y0_blk0 + yb_size - 1)); y++)
                  {
#pragma omp simd aligned(u2 : 32)
                    for (int z = z_m; z <= z_M; z += 1)
                    {
                      float r1 = -3.0548441F * u2[t0][x - time + 16][y - time + 16][z + 16];
                      u2[t1][x - time + 16][y - time + 16][z + 16] = 1.0e-1F + (r1 - 2.42812743e-6F * (u2[t0][x - time + 16][y - time + 16][z + 8] + u2[t0][x - time + 16][y - time + 16][z + 24]) + 5.07429079e-5F * (u2[t0][x - time + 16][y - time + 16][z + 9] + u2[t0][x - time + 16][y - time + 16][z + 23]) - 5.18000518e-4F * (u2[t0][x - time + 16][y - time + 16][z + 10] + u2[t0][x - time + 16][y - time + 16][z + 22]) + 3.48096348e-3F * (u2[t0][x - time + 16][y - time + 16][z + 11] + u2[t0][x - time + 16][y - time + 16][z + 21]) - 1.76767677e-2F * (u2[t0][x - time + 16][y - time + 16][z + 12] + u2[t0][x - time + 16][y - time + 16][z + 20]) + 7.54208754e-2F * (u2[t0][x - time + 16][y - time + 16][z + 13] + u2[t0][x - time + 16][y - time + 16][z + 19]) - 3.11111111e-1F * (u2[t0][x - time + 16][y - time + 16][z + 14] + u2[t0][x - time + 16][y - time + 16][z + 18]) + 1.77777778F * (u2[t0][x - time + 16][y - time + 16][z + 15] + u2[t0][x - time + 16][y - time + 16][z + 17])) / ((h_z * h_z)) + (r1 - 2.42812743e-6F * (u2[t0][x - time + 16][y - time + 8][z + 16] + u2[t0][x - time + 16][y - time + 24][z + 16]) + 5.07429079e-5F * (u2[t0][x - time + 16][y - time + 9][z + 16] + u2[t0][x - time + 16][y - time + 23][z + 16]) - 5.18000518e-4F * (u2[t0][x - time + 16][y - time + 10][z + 16] + u2[t0][x - time + 16][y - time + 22][z + 16]) + 3.48096348e-3F * (u2[t0][x - time + 16][y - time + 11][z + 16] + u2[t0][x - time + 16][y - time + 21][z + 16]) - 1.76767677e-2F * (u2[t0][x - time + 16][y - time + 12][z + 16] + u2[t0][x - time + 16][y - time + 20][z + 16]) + 7.54208754e-2F * (u2[t0][x - time + 16][y - time + 13][z + 16] + u2[t0][x - time + 16][y - time + 19][z + 16]) - 3.11111111e-1F * (u2[t0][x - time + 16][y - time + 14][z + 16] + u2[t0][x - time + 16][y - time + 18][z + 16]) + 1.77777778F * (u2[t0][x - time + 16][y - time + 15][z + 16] + u2[t0][x - time + 16][y - time + 17][z + 16])) / ((h_y * h_y)) + (r1 - 2.42812743e-6F * (u2[t0][x - time + 8][y - time + 16][z + 16] + u2[t0][x - time + 24][y - time + 16][z + 16]) + 5.07429079e-5F * (u2[t0][x - time + 9][y - time + 16][z + 16] + u2[t0][x - time + 23][y - time + 16][z + 16]) - 5.18000518e-4F * (u2[t0][x - time + 10][y - time + 16][z + 16] + u2[t0][x - time + 22][y - time + 16][z + 16]) + 3.48096348e-3F * (u2[t0][x - time + 11][y - time + 16][z + 16] + u2[t0][x - time + 21][y - time + 16][z + 16]) - 1.76767677e-2F * (u2[t0][x - time + 12][y - time + 16][z + 16] + u2[t0][x - time + 20][y - time + 16][z + 16]) + 7.54208754e-2F * (u2[t0][x - time + 13][y - time + 16][z + 16] + u2[t0][x - time + 19][y - time + 16][z + 16]) - 3.11111111e-1F * (u2[t0][x - time + 14][y - time + 16][z + 16] + u2[t0][x - time + 18][y - time + 16][z + 16]) + 1.77777778F * (u2[t0][x - time + 15][y - time + 16][z + 16] + u2[t0][x - time + 17][y - time + 16][z + 16])) / ((h_x * h_x));
                    }
                    
                    if (nnz_sp_source_mask[x - time + 16][y - time + 16])
                      {printf(" x-time+16: %d , y-time+16: %d , sp_zi_M: %d \n", x-time+16 , y-time+16, nnz_sp_source_mask[x - time + 16][y - time + 16]);
                      }
                    #pragma omp simd aligned(u2 : 32)
                    for (int sp_zi = sp_zi_m; sp_zi < nnz_sp_source_mask[x - time + 16][y - time + 16]; sp_zi += 1)
                    {
                      // printf(" sp_zi: %f \n", sp_zi);
                      float r0 = save_src[time][source_id[x - time + 16][y - time + 16][sp_source_mask[x - time + 16][y - time + 16][sp_zi]]] * source_mask[x - time + 16][y - time + 16][sp_source_mask[x - time + 16][y - time + 16][sp_zi]];
                      // printf(" r0: %f \n", r0);

                      u2[t1][x - time + 16][y - time + 16][sp_source_mask[x - time + 16][y - time + 16][sp_zi] + 16] += r0;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec - start_section0.tv_sec) + (double)(end_section0.tv_usec - start_section0.tv_usec) / 1000000;
  }
  return 0;
}
/* Backdoor edit at Fri Jun 26 15:27:54 2020*/ 
/* Backdoor edit at Fri Jun 26 15:30:01 2020*/ 
/* Backdoor edit at Fri Jun 26 15:31:07 2020*/ 
/* Backdoor edit at Fri Jun 26 15:33:30 2020*/ 
/* Backdoor edit at Fri Jun 26 15:38:32 2020*/ 
/* Backdoor edit at Fri Jun 26 15:45:25 2020*/ 
/* Backdoor edit at Fri Jun 26 15:47:32 2020*/ 
/* Backdoor edit at Fri Jun 26 15:57:20 2020*/ 
/* Backdoor edit at Fri Jun 26 16:03:31 2020*/ 
/* Backdoor edit at Fri Jun 26 16:08:17 2020*/ 
/* Backdoor edit at Fri Jun 26 16:10:50 2020*/ 
/* Backdoor edit at Fri Jun 26 16:13:45 2020*/ 
/* Backdoor edit at Fri Jun 26 16:16:40 2020*/ 
/* Backdoor edit at Fri Jun 26 16:17:59 2020*/ 
/* Backdoor edit at Fri Jun 26 16:26:13 2020*/ 
/* Backdoor edit at Fri Jun 26 16:28:02 2020*/ 
