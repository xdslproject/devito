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

int Kernel(struct dataobj *restrict damp_vec, const float dt, const float h_x, const float h_y, const float h_z, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict save_src_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict u2_vec, struct dataobj *restrict vp_vec, const int sp_zi_m, const int time_M, const int time_m, struct profiler *timers, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m)
{
  float(*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[damp_vec->size[1]][damp_vec->size[2]])damp_vec->data;
  int(*restrict nnz_sp_source_mask)[nnz_sp_source_mask_vec->size[1]] __attribute__((aligned(64))) = (int(*)[nnz_sp_source_mask_vec->size[1]])nnz_sp_source_mask_vec->data;
  float(*restrict save_src)[save_src_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_vec->size[1]])save_src_vec->data;
  int(*restrict source_id)[source_id_vec->size[1]][source_id_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_id_vec->size[1]][source_id_vec->size[2]])source_id_vec->data;
  int(*restrict source_mask)[source_mask_vec->size[1]][source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_mask_vec->size[1]][source_mask_vec->size[2]])source_mask_vec->data;
  int(*restrict sp_source_mask)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]])sp_source_mask_vec->data;
  float(*restrict u2)[u2_vec->size[1]][u2_vec->size[2]][u2_vec->size[3]] __attribute__((aligned(64))) = (float(*)[u2_vec->size[1]][u2_vec->size[2]][u2_vec->size[3]])u2_vec->data;
  float(*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[vp_vec->size[1]][vp_vec->size[2]])vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  int xb_size = 128;
  int yb_size = 128; // to fix as 8/16 etc

  int x0_blk0_size = 32;
  int y0_blk0_size = 32;

  int sf = 8;
  //int t_blk_size = time_M - time_m ;
  int t_blk_size = 2 * sf * (time_M - time_m);

  struct timeval start_section0, end_section0;
  gettimeofday(&start_section0, NULL);
  for (int t_blk = time_m; t_blk < sf * (time_M - time_m); t_blk += sf * t_blk_size) // for each t block
  {
    //printf(" Change of tblock %d \n", t_blk);
    for (int xb = x_m; xb <= (x_M + sf * (time_M - time_m)); xb += xb_size + 1)
    {
      //printf(" Change of outer xblock %d \n", xb);
      for (int yb = y_m; yb <= (y_M + sf * (time_M - time_m)); yb += yb_size + 1)
      {
        //printf(" Change of yblock %d \n", yb);
        for (int time = t_blk, t0 = (time + 1) % (3), t1 = (time) % (3), t2 = (time + 2) % (3); time <= 1 + min(t_blk + t_blk_size - 1, sf * (time_M - time_m)); time += sf, t0 = (((time / sf) % (time_M - time_m + 1)) + 1) % (3), t1 = (((time / sf) % (time_M - time_m + 1))) % (3), t2 = (((time / sf) % (time_M - time_m + 1)) + 2) % (3))
        {
          int tw = ((time / sf) % (time_M - time_m + 1));
          //printf(" Change of time %d t0: %d t1: %d t2: %d \n", tw, t0, t1, t2);
          /* Begin section0 */
#pragma omp parallel num_threads(1)
          {
#pragma omp for collapse(2) schedule(dynamic, 1)
            for (int x0_blk0 = max((x_m + time), xb); x0_blk0 <= min((x_M + time), (xb + xb_size)); x0_blk0 += x0_blk0_size)
            {
              // printf(" Change of inner xblock %d \n", x0_blk0);
              for (int y0_blk0 = max((y_m + time), yb); y0_blk0 <= min((y_M + time), (yb + yb_size)); y0_blk0 += y0_blk0_size)
              {
                for (int x = x0_blk0; x <= min(min((x_M + time), (xb + xb_size)), (x0_blk0 + x0_blk0_size - 1)); x++)
                {
                  //printf(" time: %d , x: %d \n", time, x - time);
                  for (int y = y0_blk0; y <= min(min((y_M + time), (yb + yb_size)), (y0_blk0 + y0_blk0_size - 1)); y++)
                  {
                    //printf(" time: %d , x : %d , y : %d \n", tw, x - time , y - time );
//#pragma omp simd aligned(damp, u2, vp : 32)
                    for (int z = z_m; z <= z_M; z += 1)
                    {
                      float r14 = -2.84722222F * u2[t1][x - time + 8][y - time + 8][z + 8];
                      float r13 = 1.0 / dt;
                      float r12 = 1.0 / (dt * dt);
                      float r11 = 1.0 / (vp[x - time + 8][y - time + 8][z + 8] * vp[x - time + 8][y - time + 8][z + 8]);
                      u2[t0][x - time + 8][y - time + 8][z + 8] += 0.0F; //  (r11 * (-r12 * (-2.0F * u2[t1][x - time + 8][y - time + 8][z + 8] + u2[t2][x - time + 8][y - time + 8][z + 8])) + r13 * (damp[x - time + 1][y - time + 1][z + 1] * u2[t1][x - time + 8][y - time + 8][z + 8]) + (r14 - 1.78571429e-3F * (u2[t1][x - time + 8][y - time + 8][z + 4] + u2[t1][x - time + 8][y - time + 8][z + 12]) + 2.53968254e-2F * (u2[t1][x - time + 8][y - time + 8][z + 5] + u2[t1][x - time + 8][y - time + 8][z + 11]) - 2.0e-1F * (u2[t1][x - time + 8][y - time + 8][z + 6] + u2[t1][x - time + 8][y - time + 8][z + 10]) + 1.6F * (u2[t1][x - time + 8][y - time + 8][z + 7] + u2[t1][x - time + 8][y - time + 8][z + 9])) / ((h_z * h_z)) + (r14 - 1.78571429e-3F * (u2[t1][x - time + 8][y - time + 4][z + 8] + u2[t1][x - time + 8][y - time + 12][z + 8]) + 2.53968254e-2F * (u2[t1][x - time + 8][y - time + 5][z + 8] + u2[t1][x - time + 8][y - time + 11][z + 8]) - 2.0e-1F * (u2[t1][x - time + 8][y - time + 6][z + 8] + u2[t1][x - time + 8][y - time + 10][z + 8]) + 1.6F * (u2[t1][x - time + 8][y - time + 7][z + 8] + u2[t1][x - time + 8][y - time + 9][z + 8])) / ((h_y * h_y)) + (r14 - 1.78571429e-3F * (u2[t1][x - time + 4][y - time + 8][z + 8] + u2[t1][x - time + 12][y - time + 8][z + 8]) + 2.53968254e-2F * (u2[t1][x - time + 5][y - time + 8][z + 8] + u2[t1][x - time + 11][y - time + 8][z + 8]) - 2.0e-1F * (u2[t1][x - time + 6][y - time + 8][z + 8] + u2[t1][x - time + 10][y - time + 8][z + 8]) + 1.6F * (u2[t1][x - time + 7][y - time + 8][z + 8] + u2[t1][x - time + 9][y - time + 8][z + 8])) / ((h_x * h_x))) / (r11 * r12 + r13 * damp[x - time + 1][y - time + 1][z + 1]);
                      
                      if (source_mask[x - time + 1][y - time + 1][z + 1])
                      {
                      //printf(" x-time+1: %d , y-time+1: %d, z: %d \n", x - time + 1, y - time + 1, z + 1);
                      }
                    }
//if (nnz_sp_source_mask[x - time + 2][y - time + 2])
//{
//  printf(" x-time+2: %d , y-time+2: %d , sp_zi_M: %d \n", x - time + 2, y - time + 2, nnz_sp_source_mask[x - time + 2][y - time + 2]);
//}
//int sp_zi_M = nnz_sp_source_mask[x - time + 2][y - time + 2];

//#pragma omp simd aligned(u2 : 32)

                    for (int sp_zi = sp_zi_m; sp_zi < sp_zi_m + nnz_sp_source_mask[x - time + 1][y - time + 1]; sp_zi += 1)
                    {
                      //printf(" time is : %d \n", ((time / sf) % (time_M - time_m + 1)));
                      //printf(" sp_source_mask = %d \n", sp_source_mask[x - time + 1][y - time + 1][sp_zi] + 1);
                      int zind = sp_source_mask[x - time + 1][y - time + 1][sp_zi] + 1;
                      //printf(" source_mask = %d \n", source_mask[x - time + 1][y - time + 1][zind]);
                      //printf(" source_id = %d \n", source_id[x - time + 2][y - time + 2][zind]);
                      float r0 = save_src[((time / sf) % (time_M - time_m + 1))][source_id[x - time + 1][y - time + 1][zind + 1] + 1] * source_mask[x - time + 1][y - time + 1][sp_source_mask[x - time + 1][y - time + 1][sp_zi] + 1];
                      //printf(" Input %f in %d, %d, %d valued %f \n", r0, x - time + 8, y - time + 8,zind + 8, u2[t0][x - time + 8][y - time + 8][zind + 8]);
                      
                      u2[t0][x - time + 8][y - time + 8][zind + 8] += r0;
                    }
   
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
  return 0;
}
/* Backdoor edit at Wed Jul  1 18:59:09 2020*/
/* Backdoor edit at Wed Jul  1 19:03:51 2020*/
/* Backdoor edit at Wed Jul  1 19:15:48 2020*/
/* Backdoor edit at Wed Jul  1 19:18:10 2020*/
/* Backdoor edit at Wed Jul  1 19:20:16 2020*/
/* Backdoor edit at Wed Jul  1 19:22:01 2020*/
/* Backdoor edit at Wed Jul  1 19:23:01 2020*/
/* Backdoor edit at Wed Jul  1 19:25:36 2020*/
/* Backdoor edit at Wed Jul  1 19:29:06 2020*/
/* Backdoor edit at Wed Jul  1 19:29:35 2020*/
/* Backdoor edit at Wed Jul  1 19:32:26 2020*/
/* Backdoor edit at Wed Jul  1 19:33:16 2020*/
/* Backdoor edit at Wed Jul  1 19:34:18 2020*/
/* Backdoor edit at Wed Jul  1 19:37:19 2020*/
/* Backdoor edit at Wed Jul  1 19:39:37 2020*/
/* Backdoor edit at Wed Jul  1 19:40:26 2020*/
/* Backdoor edit at Wed Jul  1 19:41:26 2020*/
/* Backdoor edit at Wed Jul  1 19:42:19 2020*/
/* Backdoor edit at Wed Jul  1 19:47:06 2020*/
/* Backdoor edit at Wed Jul  1 19:47:42 2020*/
/* Backdoor edit at Wed Jul  1 19:59:45 2020*/
/* Backdoor edit at Wed Jul  1 20:03:22 2020*/
/* Backdoor edit at Wed Jul  1 20:10:21 2020*/ 
/* Backdoor edit at Wed Jul  1 20:12:08 2020*/ 
/* Backdoor edit at Wed Jul  1 20:12:49 2020*/ 
/* Backdoor edit at Wed Jul  1 20:18:54 2020*/ 
/* Backdoor edit at Wed Jul  1 20:23:14 2020*/ 
/* Backdoor edit at Wed Jul  1 20:27:23 2020*/ 
/* Backdoor edit at Wed Jul  1 20:28:09 2020*/ 
/* Backdoor edit at Wed Jul  1 20:35:11 2020*/ 
/* Backdoor edit at Wed Jul  1 20:36:27 2020*/ 
/* Backdoor edit at Wed Jul  1 20:38:43 2020*/ 
/* Backdoor edit at Wed Jul  1 20:40:13 2020*/ 
/* Backdoor edit at Wed Jul  1 20:41:58 2020*/ 
/* Backdoor edit at Wed Jul  1 20:43:31 2020*/ 
/* Backdoor edit at Wed Jul  1 20:44:28 2020*/ 
/* Backdoor edit at Wed Jul  1 20:45:50 2020*/ 
/* Backdoor edit at Wed Jul  1 20:47:51 2020*/ 
/* Backdoor edit at Wed Jul  1 20:49:32 2020*/ 
/* Backdoor edit at Wed Jul  1 20:50:49 2020*/ 
/* Backdoor edit at Wed Jul  1 20:56:02 2020*/ 
/* Backdoor edit at Wed Jul  1 20:57:43 2020*/ 
/* Backdoor edit at Wed Jul  1 21:05:04 2020*/ 
/* Backdoor edit at Wed Jul  1 21:06:25 2020*/ 
/* Backdoor edit at Wed Jul  1 21:10:04 2020*/ 
/* Backdoor edit at Wed Jul  1 21:11:35 2020*/ 
/* Backdoor edit at Wed Jul  1 21:12:50 2020*/ 
/* Backdoor edit at Wed Jul  1 21:13:30 2020*/ 
/* Backdoor edit at Wed Jul  1 21:14:34 2020*/ 
/* Backdoor edit at Wed Jul  1 21:17:30 2020*/ 
/* Backdoor edit at Wed Jul  1 21:18:33 2020*/ 
/* Backdoor edit at Wed Jul  1 21:22:30 2020*/ 
/* Backdoor edit at Wed Jul  1 21:25:07 2020*/ 
/* Backdoor edit at Wed Jul  1 21:28:36 2020*/ 
/* Backdoor edit at Wed Jul  1 21:29:28 2020*/ 
/* Backdoor edit at Wed Jul  1 21:32:59 2020*/ 
/* Backdoor edit at Wed Jul  1 21:33:38 2020*/ 
/* Backdoor edit at Wed Jul  1 21:34:49 2020*/ 
/* Backdoor edit at Wed Jul  1 21:37:36 2020*/ 
/* Backdoor edit at Wed Jul  1 21:38:23 2020*/ 
/* Backdoor edit at Wed Jul  1 21:39:44 2020*/ 
/* Backdoor edit at Wed Jul  1 21:40:22 2020*/ 
/* Backdoor edit at Wed Jul  1 21:41:34 2020*/ 
/* Backdoor edit at Wed Jul  1 21:43:33 2020*/ 
/* Backdoor edit at Wed Jul  1 21:44:38 2020*/ 
/* Backdoor edit at Wed Jul  1 21:47:11 2020*/ 
/* Backdoor edit at Wed Jul  1 21:47:35 2020*/ 
/* Backdoor edit at Wed Jul  1 21:49:49 2020*/ 
/* Backdoor edit at Wed Jul  1 21:55:09 2020*/ 
/* Backdoor edit at Wed Jul  1 21:55:51 2020*/ 
/* Backdoor edit at Wed Jul  1 21:58:49 2020*/ 
/* Backdoor edit at Wed Jul  1 21:59:20 2020*/ 
/* Backdoor edit at Wed Jul  1 22:00:27 2020*/ 
/* Backdoor edit at Wed Jul  1 22:00:51 2020*/ 
/* Backdoor edit at Wed Jul  1 22:02:20 2020*/ 
/* Backdoor edit at Wed Jul  1 22:05:03 2020*/ 
