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
  int xb_size = 32;
  int yb_size = 32; // to fix as 8/16 etc

  int x0_blk0_size = 8;
  int y0_blk0_size = 8;

  int sf = 2;
  //int t_blk_size = time_M - time_m ;
  int t_blk_size = 20 * (time_M - time_m);

  //printf("Global time loop to timesteps = %d \n", time_M - time_m +1 );
  for (int t_blk = time_m; t_blk < sf * (time_M - time_m); t_blk += sf * t_blk_size) // for each t block

  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    //printf("From x: %d to %d \n", x_m, (x_M + sf * (time_M - time_m)));
    //printf("From y: %d to %d \n", y_m, (y_M + sf * (time_M - time_m)));
    for (int xb = x_m; xb <= (x_M + sf * (time_M - time_m)); xb += xb_size + 1)
    {
      //printf(" Change of xblock %d \n", xb);
      for (int yb = y_m; yb <= (y_M + sf * (time_M - time_m)); yb += yb_size + 1)
      {
        //printf(" Change of yblock %d \n", yb);
        //printf("----y0_blk0 loop from y0_blk0 = %d to %d \n", y_m, (y_M + sf * (time_M - time_m)));
        for (int time = t_blk, t0 = (time) % (2), t1 = (time + 1) % (2); time <= 1 + min(t_blk + t_blk_size, sf * (time_M - time_m)); time += 1, t0 = (time) % (2), t1 = (time + 1) % (2))
        {
          int tw = ((time / sf) % (time_M - time_m + 1));
          /* Begin section0 */
#pragma omp parallel num_threads(6)
          {
#pragma omp for collapse(2) schedule(dynamic, 1)
            for (int x0_blk0 = max((x_m + time), xb); x0_blk0 <= min((x_M + time), (xb + xb_size)); x0_blk0 += x0_blk0_size)
            {
              for (int y0_blk0 = max((y_m + time), yb); y0_blk0 <= min((y_M + time), (yb + yb_size)); y0_blk0 += y0_blk0_size)
              {
                for (int x = x0_blk0; x <= min(min((x_M + time), (xb + xb_size)), (x0_blk0 + x0_blk0_size - 1)); x++)
                {
                  for (int y = y0_blk0; y <= min(min((y_M + time), (yb + yb_size)), (y0_blk0 + y0_blk0_size - 1)); y++)
                  {
                    //printf(" t1: %d , x-time+2: %d , y-time+2: %d \n", t1, x -time + 2, y -time + 2);
#pragma omp simd aligned(u2 : 32)
                    for (int z = z_m; z <= z_M; z += 1)
                    {
                      float r1 = -2.0F * u2[t0][x - time + 2][y - time + 2][z + 2];
                      u2[t1][x - time + 2][y - time + 2][z + 2] = 1.0e-1F + (r1 + u2[t0][x - time + 2][y - time + 2][z + 1] + u2[t0][x - time + 2][y - time + 2][z + 3]) / ((h_z * h_z)) + (r1 + u2[t0][x - time + 2][y - time + 1][z + 2] + u2[t0][x - time + 2][y - time + 3][z + 2]) / ((h_y * h_y)) + (r1 + u2[t0][x - time + 1][y - time + 2][z + 2] + u2[t0][x - time + 3][y - time + 2][z + 2]) / ((h_x * h_x));
                      //if (source_mask[x - time + 2][y - time + 2][z + 2])
                      //{
                      //  printf(" x-time+2: %d , y-time+2: %d, z: %d \n", x - time + 2, y - time + 2, z + 2);
                      //}
                    }
                    //if (nnz_sp_source_mask[x - time + 2][y - time + 2])
                    //{
                    //  printf(" x-time+2: %d , y-time+2: %d , sp_zi_M: %d \n", x - time + 2, y - time + 2, nnz_sp_source_mask[x - time + 2][y - time + 2]);
                    //}
                    //int sp_zi_M = nnz_sp_source_mask[x - time + 2][y - time + 2];
                    for (int sp_zi = sp_zi_m; sp_zi < sp_zi_m + nnz_sp_source_mask[x - time + 2][y - time + 2]; sp_zi += 1)
                    {
                      //printf(" sp_source_mask = %d \n", sp_source_mask[x - time + 2][y - time + 2][sp_zi] + 1);
                      int zind = sp_source_mask[x - time + 2][y - time + 2][sp_zi] + 1;
                      //printf(" source_mask = %d \n", source_mask[x - time + 2][y - time + 2][zind]);
                      //printf(" source_id = %d \n", source_id[x - time + 2][y - time + 2][zind]);
                      float r0 = save_src[tw][source_id[x - time + 2][y - time + 2][zind] + 1] * source_mask[x - time + 2][y - time + 2][zind];
                      //printf(" Input %f \n", r0);
                      //printf(" time is : %d \n", tw);
                      u2[t1][x - time + 2][y - time + 2][zind + 2] += r0;
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
/* Backdoor edit at Fri Jun 26 16:41:34 2020*/
/* Backdoor edit at Fri Jun 26 16:42:14 2020*/
/* Backdoor edit at Fri Jun 26 16:42:57 2020*/
/* Backdoor edit at Fri Jun 26 16:43:47 2020*/
/* Backdoor edit at Fri Jun 26 16:44:46 2020*/
/* Backdoor edit at Fri Jun 26 16:47:48 2020*/
/* Backdoor edit at Fri Jun 26 16:48:51 2020*/
/* Backdoor edit at Fri Jun 26 16:49:30 2020*/
/* Backdoor edit at Fri Jun 26 16:51:46 2020*/
/* Backdoor edit at Fri Jun 26 16:52:21 2020*/
/* Backdoor edit at Fri Jun 26 16:54:22 2020*/
/* Backdoor edit at Fri Jun 26 16:57:35 2020*/
/* Backdoor edit at Fri Jun 26 17:01:19 2020*/
/* Backdoor edit at Fri Jun 26 17:03:10 2020*/
/* Backdoor edit at Fri Jun 26 17:04:25 2020*/
/* Backdoor edit at Fri Jun 26 17:04:50 2020*/
/* Backdoor edit at Fri Jun 26 17:08:06 2020*/
/* Backdoor edit at Fri Jun 26 17:08:57 2020*/
/* Backdoor edit at Fri Jun 26 17:09:58 2020*/
/* Backdoor edit at Fri Jun 26 17:11:39 2020*/
/* Backdoor edit at Fri Jun 26 17:12:13 2020*/
/* Backdoor edit at Fri Jun 26 17:12:33 2020*/
/* Backdoor edit at Fri Jun 26 17:13:01 2020*/
/* Backdoor edit at Fri Jun 26 17:14:30 2020*/
/* Backdoor edit at Fri Jun 26 17:14:49 2020*/
/* Backdoor edit at Fri Jun 26 17:15:03 2020*/
/* Backdoor edit at Fri Jun 26 17:15:49 2020*/
/* Backdoor edit at Fri Jun 26 17:16:52 2020*/
/* Backdoor edit at Fri Jun 26 17:17:40 2020*/
/* Backdoor edit at Fri Jun 26 17:21:29 2020*/
/* Backdoor edit at Fri Jun 26 17:22:03 2020*/
/* Backdoor edit at Fri Jun 26 17:23:08 2020*/
/* Backdoor edit at Fri Jun 26 17:23:48 2020*/
/* Backdoor edit at Fri Jun 26 17:25:34 2020*/
/* Backdoor edit at Fri Jun 26 17:26:09 2020*/
/* Backdoor edit at Fri Jun 26 17:27:45 2020*/
/* Backdoor edit at Fri Jun 26 17:28:18 2020*/
/* Backdoor edit at Fri Jun 26 17:29:45 2020*/
/* Backdoor edit at Fri Jun 26 17:31:21 2020*/
/* Backdoor edit at Fri Jun 26 17:34:18 2020*/
/* Backdoor edit at Fri Jun 26 17:34:49 2020*/
/* Backdoor edit at Fri Jun 26 17:37:12 2020*/
/* Backdoor edit at Fri Jun 26 17:39:44 2020*/
/* Backdoor edit at Fri Jun 26 17:40:21 2020*/
/* Backdoor edit at Fri Jun 26 17:40:44 2020*/
/* Backdoor edit at Fri Jun 26 17:41:52 2020*/
/* Backdoor edit at Fri Jun 26 17:45:01 2020*/
/* Backdoor edit at Fri Jun 26 17:46:33 2020*/
/* Backdoor edit at Fri Jun 26 17:48:52 2020*/
/* Backdoor edit at Fri Jun 26 17:49:20 2020*/
/* Backdoor edit at Fri Jun 26 17:50:03 2020*/
/* Backdoor edit at Fri Jun 26 17:50:37 2020*/
/* Backdoor edit at Fri Jun 26 17:53:02 2020*/
/* Backdoor edit at Fri Jun 26 17:53:31 2020*/
/* Backdoor edit at Fri Jun 26 17:54:14 2020*/
/* Backdoor edit at Fri Jun 26 17:54:34 2020*/
/* Backdoor edit at Fri Jun 26 17:55:09 2020*/
/* Backdoor edit at Fri Jun 26 17:57:29 2020*/
/* Backdoor edit at Fri Jun 26 18:04:04 2020*/
/* Backdoor edit at Fri Jun 26 18:05:16 2020*/
/* Backdoor edit at Fri Jun 26 18:05:51 2020*/
/* Backdoor edit at Fri Jun 26 18:06:21 2020*/
/* Backdoor edit at Fri Jun 26 18:06:42 2020*/
/* Backdoor edit at Fri Jun 26 18:07:04 2020*/
/* Backdoor edit at Fri Jun 26 18:07:46 2020*/
/* Backdoor edit at Fri Jun 26 18:08:53 2020*/
/* Backdoor edit at Fri Jun 26 18:10:27 2020*/
/* Backdoor edit at Fri Jun 26 18:11:56 2020*/
/* Backdoor edit at Fri Jun 26 18:15:00 2020*/
/* Backdoor edit at Fri Jun 26 18:15:27 2020*/
/* Backdoor edit at Fri Jun 26 18:16:28 2020*/
/* Backdoor edit at Fri Jun 26 18:17:33 2020*/
/* Backdoor edit at Fri Jun 26 18:21:05 2020*/
/* Backdoor edit at Fri Jun 26 18:22:10 2020*/
/* Backdoor edit at Fri Jun 26 18:22:58 2020*/
/* Backdoor edit at Fri Jun 26 18:23:23 2020*/
/* Backdoor edit at Fri Jun 26 18:24:22 2020*/
/* Backdoor edit at Fri Jun 26 18:24:57 2020*/
/* Backdoor edit at Fri Jun 26 18:25:57 2020*/
/* Backdoor edit at Fri Jun 26 18:26:21 2020*/
/* Backdoor edit at Fri Jun 26 18:27:13 2020*/
/* Backdoor edit at Fri Jun 26 18:27:45 2020*/
/* Backdoor edit at Fri Jun 26 18:28:26 2020*/
/* Backdoor edit at Fri Jun 26 18:29:29 2020*/
/* Backdoor edit at Fri Jun 26 18:29:52 2020*/
/* Backdoor edit at Fri Jun 26 18:30:45 2020*/
/* Backdoor edit at Fri Jun 26 18:31:42 2020*/
/* Backdoor edit at Fri Jun 26 18:32:08 2020*/
/* Backdoor edit at Fri Jun 26 18:32:52 2020*/
/* Backdoor edit at Fri Jun 26 18:35:13 2020*/
/* Backdoor edit at Fri Jun 26 18:37:09 2020*/
/* Backdoor edit at Fri Jun 26 18:37:40 2020*/
/* Backdoor edit at Fri Jun 26 18:42:24 2020*/
/* Backdoor edit at Fri Jun 26 18:46:02 2020*/
/* Backdoor edit at Fri Jun 26 18:47:20 2020*/
/* Backdoor edit at Fri Jun 26 18:47:50 2020*/
/* Backdoor edit at Fri Jun 26 18:48:10 2020*/
/* Backdoor edit at Fri Jun 26 18:49:01 2020*/
/* Backdoor edit at Fri Jun 26 18:56:19 2020*/
/* Backdoor edit at Fri Jun 26 18:57:50 2020*/
/* Backdoor edit at Fri Jun 26 18:58:17 2020*/
/* Backdoor edit at Fri Jun 26 18:58:55 2020*/
/* Backdoor edit at Fri Jun 26 18:59:21 2020*/
/* Backdoor edit at Fri Jun 26 19:00:13 2020*/
/* Backdoor edit at Fri Jun 26 19:00:48 2020*/
/* Backdoor edit at Fri Jun 26 19:01:11 2020*/
/* Backdoor edit at Fri Jun 26 19:01:42 2020*/
/* Backdoor edit at Fri Jun 26 19:02:16 2020*/
/* Backdoor edit at Fri Jun 26 19:02:56 2020*/
/* Backdoor edit at Fri Jun 26 19:03:29 2020*/
/* Backdoor edit at Fri Jun 26 19:04:12 2020*/
/* Backdoor edit at Fri Jun 26 19:05:25 2020*/
/* Backdoor edit at Fri Jun 26 19:05:53 2020*/
/* Backdoor edit at Fri Jun 26 19:06:32 2020*/
/* Backdoor edit at Fri Jun 26 19:07:29 2020*/
/* Backdoor edit at Fri Jun 26 19:07:52 2020*/
/* Backdoor edit at Fri Jun 26 19:08:27 2020*/
/* Backdoor edit at Fri Jun 26 19:08:56 2020*/
/* Backdoor edit at Fri Jun 26 19:09:17 2020*/
/* Backdoor edit at Fri Jun 26 19:10:16 2020*/
/* Backdoor edit at Fri Jun 26 19:11:39 2020*/
/* Backdoor edit at Fri Jun 26 19:12:13 2020*/
/* Backdoor edit at Fri Jun 26 19:12:58 2020*/
/* Backdoor edit at Fri Jun 26 19:13:52 2020*/
/* Backdoor edit at Fri Jun 26 19:29:33 2020*/ 
/* Backdoor edit at Fri Jun 26 19:33:47 2020*/ 
/* Backdoor edit at Fri Jun 26 19:39:21 2020*/ 
/* Backdoor edit at Fri Jun 26 19:46:11 2020*/ 
/* Backdoor edit at Fri Jun 26 19:50:02 2020*/ 
/* Backdoor edit at Fri Jun 26 19:54:31 2020*/ 
/* Backdoor edit at Fri Jun 26 19:59:07 2020*/ 
/* Backdoor edit at Fri Jun 26 20:03:31 2020*/ 
/* Backdoor edit at Fri Jun 26 20:06:36 2020*/ 
