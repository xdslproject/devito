#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "omp.h"
#include <stdio.h>

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
  double section1;
  double section2;
};

int ForwardTTI(struct dataobj *restrict block_sizes_vec, struct dataobj *restrict damp_vec, struct dataobj *restrict delta_vec, const float dt, struct dataobj *restrict epsilon_vec, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict phi_vec, struct dataobj *restrict save_src_u_vec, struct dataobj *restrict save_src_v_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict theta_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x_size, const int y_size, const int z_size, const int sp_zi_m, const int time_M, const int time_m, struct profiler *timers, const int x1_blk0_size, const int x_M, const int x_m, const int y1_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, const int nthreads_nonaffine)
{
  int(*restrict block_sizes) __attribute__((aligned(64))) = (int(*))block_sizes_vec->data;

  float(*restrict delta)[delta_vec->size[1]][delta_vec->size[2]] __attribute__((aligned(64))) = (float(*)[delta_vec->size[1]][delta_vec->size[2]])delta_vec->data;
  int(*restrict nnz_sp_source_mask)[nnz_sp_source_mask_vec->size[1]] __attribute__((aligned(64))) = (int(*)[nnz_sp_source_mask_vec->size[1]])nnz_sp_source_mask_vec->data;
  float(*restrict phi)[phi_vec->size[1]][phi_vec->size[2]] __attribute__((aligned(64))) = (float(*)[phi_vec->size[1]][phi_vec->size[2]])phi_vec->data;
  float(*restrict save_src_u)[save_src_u_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_u_vec->size[1]])save_src_u_vec->data;
  float(*restrict save_src_v)[save_src_v_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_v_vec->size[1]])save_src_v_vec->data;
  int(*restrict source_id)[source_id_vec->size[1]][source_id_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_id_vec->size[1]][source_id_vec->size[2]])source_id_vec->data;
  int(*restrict source_mask)[source_mask_vec->size[1]][source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_mask_vec->size[1]][source_mask_vec->size[2]])source_mask_vec->data;
  int(*restrict sp_source_mask)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]])sp_source_mask_vec->data;
  float(*restrict theta)[theta_vec->size[1]][theta_vec->size[2]] __attribute__((aligned(64))) = (float(*)[theta_vec->size[1]][theta_vec->size[2]])theta_vec->data;

  float(*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__((aligned(64))) = (float(*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]])u_vec->data;

  float(*restrict v)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]] __attribute__((aligned(64))) = (float(*)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]])v_vec->data;

  float(*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[damp_vec->size[1]][damp_vec->size[2]])damp_vec->data;
  float(*restrict epsilon)[epsilon_vec->size[1]][epsilon_vec->size[2]] __attribute__((aligned(64))) = (float(*)[epsilon_vec->size[1]][epsilon_vec->size[2]])epsilon_vec->data;
  float(*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[vp_vec->size[1]][vp_vec->size[2]])vp_vec->data;

  printf("Allocate \n");

  float(*r21)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r21, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float(*r20)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r20, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float(*r19)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r19, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float(*r18)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r18, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float(*r17)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r17, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float(*r34)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r34, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float(*r35)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r35, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  printf("Allocate is over \n");

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  int y0_blk0_size = block_sizes[3];
  int x0_blk0_size = block_sizes[2];
  //int yb_size = block_sizes[1];
  //int xb_size = block_sizes[0];

  struct timeval start_section0, end_section0;
  gettimeofday(&start_section0, NULL);
/* Begin section0 */
// trigonometric
#pragma omp parallel num_threads(nthreads)
  {
#pragma omp for collapse(1) schedule(static, 1)
    for (int x = x_m - 1; x <= x_M; x += 1)
    {
      for (int y = y_m - 1; y <= y_M; y += 1)
      {
#pragma omp simd aligned(delta, phi, theta : 32)
        for (int z = z_m - 1; z <= z_M; z += 1)
        {
          r21[x + 1][y + 1][z + 1] = cos(phi[x + 4][y + 4][z + 4]);
          r20[x + 1][y + 1][z + 1] = sin(theta[x + 4][y + 4][z + 4]);
          r19[x + 1][y + 1][z + 1] = sin(phi[x + 4][y + 4][z + 4]);
          r18[x + 1][y + 1][z + 1] = cos(theta[x + 4][y + 4][z + 4]);
          r17[x + 1][y + 1][z + 1] = sqrt(2 * delta[x + 4][y + 4][z + 4] + 1);
        }
      }
    }
  }
  printf("Trigonometric is over \n");

  /* End section0 */
  gettimeofday(&end_section0, NULL);
  timers->section0 += (double)(end_section0.tv_sec - start_section0.tv_sec) + (double)(end_section0.tv_usec - start_section0.tv_usec) / 1000000;
  for (int time = time_m, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3); time <= time_M; time += 1, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3))
  {
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    /* Begin section1 */

    printf("Alignemnt is over \n");


#pragma omp parallel num_threads(nthreads)
    {
#pragma omp for collapse(1) schedule(dynamic, 1)
      for (int x0_blk0 = x_m-1; x0_blk0 <=  x_M - (x_M - x_m + 2)%(x0_blk0_size); x0_blk0 += x0_blk0_size)
      {
        for (int y0_blk0 = y_m-1; y0_blk0 <= (y_M - y_m + 2)%(y0_blk0_size); y0_blk0 += y0_blk0_size)
        {
          for (int x = x0_blk0; x <= x0_blk0 + x0_blk0_size - 1; x += 1)
          {
            for (int y = y0_blk0; y <= y0_blk0 + y0_blk0_size - 1; y += 1)
            {
#pragma omp simd aligned(u, v : 32)
              for (int z = z_m; z <= z_M; z += 1)
              {
                printf(" Updating x %d z: %d \n", x, z );
                float r39 = -v[t0][x + 4][y + 4][z + 4];
                r35[x + 1][y + 1][z + 1] = 1.0e-1F * (-(r39 + v[t0][x + 4][y + 4][z + 5]) * r18[x + 1][y + 1][z + 1] - (r39 + v[t0][x + 4][y + 5][z + 4]) * r19[x + 1][y + 1][z + 1] * r20[x + 1][y + 1][z + 1] - (r39 + v[t0][x + 5][y + 4][z + 4]) * r20[x + 1][y + 1][z + 1] * r21[x + 1][y + 1][z + 1]);
                float r40 = -u[t0][x + 4][y + 4][z + 4];
                r34[x + 1][y + 1][z + 1] = 1.0e-1F * (-(r40 + u[t0][x + 4][y + 4][z + 5]) * r18[x + 1][y + 1][z + 1] - (r40 + u[t0][x + 4][y + 5][z + 4]) * r19[x + 1][y + 1][z + 1] * r20[x + 1][y + 1][z + 1] - (r40 + u[t0][x + 5][y + 4][z + 4]) * r20[x + 1][y + 1][z + 1] * r21[x + 1][y + 1][z + 1]);
              }
            }
          }
        }
      }
    }

    printf("First loop is over \n");

#pragma omp parallel num_threads(nthreads)
    {
#pragma omp for collapse(1) schedule(dynamic, 1)
      for (int x1_blk0 = x_m; x1_blk0 <= x_M; x1_blk0 += x1_blk0_size)
      {
        for (int y1_blk0 = y_m; y1_blk0 <= y_M; y1_blk0 += y1_blk0_size)
        {
          for (int x = x1_blk0; x <= x1_blk0 + x1_blk0_size - 1; x += 1)
          {
            for (int y = y1_blk0; y <= y1_blk0 + y1_blk0_size - 1; y += 1)
            {
#pragma omp simd aligned(damp, epsilon, u, v, vp : 32)
              for (int z = z_m; z <= z_M; z += 1)
              {
                float r46 = 1.0 / dt;
                float r45 = 1.0 / (dt * dt);
                float r44 = r18[x + 1][y + 1][z] * r35[x + 1][y + 1][z] - r18[x + 1][y + 1][z + 1] * r35[x + 1][y + 1][z + 1] + r19[x + 1][y][z + 1] * r20[x + 1][y][z + 1] * r35[x + 1][y][z + 1] - r19[x + 1][y + 1][z + 1] * r20[x + 1][y + 1][z + 1] * r35[x + 1][y + 1][z + 1] + r20[x][y + 1][z + 1] * r21[x][y + 1][z + 1] * r35[x][y + 1][z + 1] - r20[x + 1][y + 1][z + 1] * r21[x + 1][y + 1][z + 1] * r35[x + 1][y + 1][z + 1];
                float r43 = pow(vp[x + 4][y + 4][z + 4], -2);
                float r42 = 1.0e-1F * (-r18[x + 1][y + 1][z] * r34[x + 1][y + 1][z] + r18[x + 1][y + 1][z + 1] * r34[x + 1][y + 1][z + 1] - r19[x + 1][y][z + 1] * r20[x + 1][y][z + 1] * r34[x + 1][y][z + 1] + r19[x + 1][y + 1][z + 1] * r20[x + 1][y + 1][z + 1] * r34[x + 1][y + 1][z + 1] - r20[x][y + 1][z + 1] * r21[x][y + 1][z + 1] * r34[x][y + 1][z + 1] + r20[x + 1][y + 1][z + 1] * r21[x + 1][y + 1][z + 1] * r34[x + 1][y + 1][z + 1]) - 8.33333315e-4F * (u[t0][x + 2][y + 4][z + 4] + u[t0][x + 4][y + 2][z + 4] + u[t0][x + 4][y + 4][z + 2] + u[t0][x + 4][y + 4][z + 6] + u[t0][x + 4][y + 6][z + 4] + u[t0][x + 6][y + 4][z + 4]) + 1.3333333e-2F * (u[t0][x + 3][y + 4][z + 4] + u[t0][x + 4][y + 3][z + 4] + u[t0][x + 4][y + 4][z + 3] + u[t0][x + 4][y + 4][z + 5] + u[t0][x + 4][y + 5][z + 4] + u[t0][x + 5][y + 4][z + 4]) - 7.49999983e-2F * u[t0][x + 4][y + 4][z + 4];
                float r41 = 1.0 / (r43 * r45 + r46 * damp[x + 1][y + 1][z + 1]);
                float r32 = r45 * (-2.0F * u[t0][x + 4][y + 4][z + 4] + u[t2][x + 4][y + 4][z + 4]);
                float r33 = r45 * (-2.0F * v[t0][x + 4][y + 4][z + 4] + v[t2][x + 4][y + 4][z + 4]);
                u[t1][x + 4][y + 4][z + 4] = r41 * ((-r32) * r43 + r42 * (2 * epsilon[x + 4][y + 4][z + 4] + 1) + 1.0e-1F * r44 * r17[x + 1][y + 1][z + 1] + r46 * (damp[x + 1][y + 1][z + 1] * u[t0][x + 4][y + 4][z + 4]));
                v[t1][x + 4][y + 4][z + 4] = r41 * ((-r33) * r43 + r42 * r17[x + 1][y + 1][z + 1] + 1.0e-1F * r44 + r46 * (damp[x + 1][y + 1][z + 1] * v[t0][x + 4][y + 4][z + 4]));
              }
              /*
              for (int sp_zi = sp_zi_m; sp_zi <= nnz_sp_source_mask[x][y] - 1; sp_zi += 1)
              {
                int zind = sp_source_mask[x][y][sp_zi];
                float r22 = save_src_u[time][source_id[x][y][zind]] * source_mask[x][y][zind];
#pragma omp atomic update
                u[t1][x + 4][y + 4][zind + 4] += r22;
                float r23 = save_src_v[time][source_id[x][y][zind]] * source_mask[x][y][zind];
#pragma omp atomic update
                v[t1][x + 4][y + 4][zind + 4] += r23;
              }
              */
            }
          }
        }
      }
    }
    printf("Second loop is over \n");

    /* End section1 */
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec - start_section1.tv_sec) + (double)(end_section1.tv_usec - start_section1.tv_usec) / 1000000;
  }
  printf(" free \n");
  free(r21);
  free(r20);
  free(r19);
  free(r18);
  free(r17);
  free(r34);
  free(r35);
  printf(" freeing has finished \n");
  return 0;
}
/* Backdoor edit at Mon Sep  7 17:48:55 2020*/
/* Backdoor edit at Mon Sep  7 17:58:16 2020*/
/* Backdoor edit at Mon Sep  7 18:01:48 2020*/
/* Backdoor edit at Mon Sep  7 18:11:14 2020*/
/* Backdoor edit at Mon Sep  7 18:15:56 2020*/
/* Backdoor edit at Mon Sep  7 18:19:30 2020*/
/* Backdoor edit at Mon Sep  7 18:21:32 2020*/
/* Backdoor edit at Mon Sep  7 18:23:52 2020*/
/* Backdoor edit at Mon Sep  7 18:28:43 2020*/
/* Backdoor edit at Mon Sep  7 18:31:14 2020*/
/* Backdoor edit at Mon Sep  7 18:37:28 2020*/ 
/* Backdoor edit at Mon Sep  7 18:39:59 2020*/ 
/* Backdoor edit at Mon Sep  7 18:42:54 2020*/ 
/* Backdoor edit at Mon Sep  7 19:07:47 2020*/ 
/* Backdoor edit at Mon Sep  7 19:55:53 2020*/ 
