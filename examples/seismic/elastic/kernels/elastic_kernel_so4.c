#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include <stdio.h>
#include "omp.h"
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

int Kernel(const float h_x, const float h_z, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict save_src_fxx_vec, struct dataobj *restrict save_src_fzz_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict tau_sol_xx_vec, struct dataobj *restrict tau_sol_xz_vec, struct dataobj *restrict tau_sol_zz_vec, struct dataobj *restrict v_sol_x_vec, struct dataobj *restrict v_sol_z_vec, const int sp_zi_m, const int time_M, const int time_m, struct profiler *timers, const int x_M, const int x_m, const int z_M, const int z_m, const int nthreads)
{
  int(*restrict nnz_sp_source_mask) __attribute__((aligned(64))) = (int(*))nnz_sp_source_mask_vec->data;
  float(*restrict save_src_fxx)[save_src_fxx_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_fxx_vec->size[1]])save_src_fxx_vec->data;
  float(*restrict save_src_fzz)[save_src_fzz_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_fzz_vec->size[1]])save_src_fzz_vec->data;
  int(*restrict source_id)[source_id_vec->size[1]] __attribute__((aligned(64))) = (int(*)[source_id_vec->size[1]])source_id_vec->data;
  int(*restrict source_mask)[source_mask_vec->size[1]] __attribute__((aligned(64))) = (int(*)[source_mask_vec->size[1]])source_mask_vec->data;
  int(*restrict sp_source_mask)[sp_source_mask_vec->size[1]] __attribute__((aligned(64))) = (int(*)[sp_source_mask_vec->size[1]])sp_source_mask_vec->data;
  float(*restrict tau_sol_xx)[tau_sol_xx_vec->size[1]][tau_sol_xx_vec->size[2]] __attribute__((aligned(64))) = (float(*)[tau_sol_xx_vec->size[1]][tau_sol_xx_vec->size[2]])tau_sol_xx_vec->data;
  float(*restrict tau_sol_xz)[tau_sol_xz_vec->size[1]][tau_sol_xz_vec->size[2]] __attribute__((aligned(64))) = (float(*)[tau_sol_xz_vec->size[1]][tau_sol_xz_vec->size[2]])tau_sol_xz_vec->data;
  float(*restrict tau_sol_zz)[tau_sol_zz_vec->size[1]][tau_sol_zz_vec->size[2]] __attribute__((aligned(64))) = (float(*)[tau_sol_zz_vec->size[1]][tau_sol_zz_vec->size[2]])tau_sol_zz_vec->data;
  float(*restrict v_sol_x)[v_sol_x_vec->size[1]][v_sol_x_vec->size[2]] __attribute__((aligned(64))) = (float(*)[v_sol_x_vec->size[1]][v_sol_x_vec->size[2]])v_sol_x_vec->data;
  float(*restrict v_sol_z)[v_sol_z_vec->size[1]][v_sol_z_vec->size[2]] __attribute__((aligned(64))) = (float(*)[v_sol_z_vec->size[1]][v_sol_z_vec->size[2]])v_sol_z_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  int sf = 4;
  int t_blk_size = 2 * sf * (time_M - time_m);
  int xb_size = 8;
  int x0_blk0_size = 4;

  //printf(" Tiles: %d, ::: Blocks %d \n", x0_blk0_size, xb_size);
  struct timeval start_section0, end_section0;
  gettimeofday(&start_section0, NULL);
  for (int t_blk = time_m; t_blk < sf * (time_M - time_m); t_blk += sf * t_blk_size) // for each t block
  {
    for (int xb = x_m; xb <= (x_M + sf * (time_M - time_m)); xb += xb_size)
    {
      //printf(" Change of outer xblock %d \n", xb);
      for (int time = t_blk, t0 = (time) % (2), t1 = (time + 1) % (2); time <= 1 + min(t_blk + t_blk_size - 1, sf * (time_M - time_m)); time += sf, t0 = (((time / sf) % (time_M - time_m + 1)) + 1) % (2), t1 = (((time / sf) % (time_M - time_m + 1))) % (2))
      {
        int tw = ((time / sf) % (time_M - time_m + 1));
        //#pragma omp parallel num_threads(nthreads)
        {
          //printf(" Change of time block :  %d \n", tw);
          //#pragma omp for collapse(1) schedule(dynamic, 1)
          for (int x0_blk0 = max((x_m + time), xb); x0_blk0 <= min((x_M + time), (xb + xb_size)); x0_blk0 += x0_blk0_size)
          {
            //printf(" Change of inner xblock %d \n", x0_blk0);
            for (int x = x0_blk0; x <= min(min((x_M + time), (xb + xb_size - 1)), (x0_blk0 + x0_blk0_size - 1)); x++)
            {
              //printf(" Updating velocity x %d \n", x - time + 4);
              //printf(" \n PDE update : \n");
#pragma omp simd aligned(tau_sol_xx, tau_sol_xz, tau_sol_zz, v_sol_x, v_sol_z : 32)
              for (int z = z_m; z <= z_M; z += 1)
              {
                //printf(" Updating velocity x %d z: %d \n", x - time + 4, z + 4);
                float r9 = 1.0 / h_z;
                float r8 = 1.0 / h_x;
                v_sol_x[t1][x - time + 4][z + 4] = r8 * (2.7280354210856e-2F * (tau_sol_xx[t0][x - time + 3][z + 4] - tau_sol_xx[t0][x - time + 6][z + 4]) + 7.36569563735987e-1F * (-tau_sol_xx[t0][x - time + 4][z + 4] + tau_sol_xx[t0][x - time + 5][z + 4])) + r9 * (2.7280354210856e-2F * (tau_sol_xz[t0][x - time + 4][z + 2] - tau_sol_xz[t0][x - time + 4][z + 5]) + 7.36569563735987e-1F * (-tau_sol_xz[t0][x - time + 4][z + 3] + tau_sol_xz[t0][x - time + 4][z + 4])) + v_sol_x[t0][x - time + 4][z + 4];
                v_sol_z[t1][x - time + 4][z + 4] = r8 * (2.7280354210856e-2F * (tau_sol_xz[t0][x - time + 2][z + 4] - tau_sol_xz[t0][x - time + 5][z + 4]) + 7.36569563735987e-1F * (-tau_sol_xz[t0][x - time + 3][z + 4] + tau_sol_xz[t0][x - time + 4][z + 4])) + r9 * (2.7280354210856e-2F * (tau_sol_zz[t0][x - time + 4][z + 3] - tau_sol_zz[t0][x - time + 4][z + 6]) + 7.36569563735987e-1F * (-tau_sol_zz[t0][x - time + 4][z + 4] + tau_sol_zz[t0][x - time + 4][z + 5])) + v_sol_z[t0][x - time + 4][z + 4];
              }
            }
          }
        }
        //#pragma omp parallel num_threads(nthreads)
        {
          //#pragma omp for collapse(1) schedule(dynamic, 1)
          for (int x0_blk0 = max((x_m + time), xb - 2); x0_blk0 <= +min((x_M + time), (xb - 2 + xb_size)); x0_blk0 += x0_blk0_size)
          {
            for (int x = x0_blk0; x <= min(min((x_M + time), (xb - 2 + xb_size - 1)), (x0_blk0 + x0_blk0_size - 1)); x++)
            {
              //printf(" Updating stress x %d \n", x - time + 4);
              //#pragma omp simd aligned(tau_sol_xx, tau_sol_xz, tau_sol_zz, v_sol_x, v_sol_z : 32)
              for (int z = z_m; z <= z_M; z += 1)
              {
                //printf(" Updating x %d z: %d \n", x - time + 4, z + 4);
                float r15 = v_sol_x[t1][x - time + 2][z + 4] - v_sol_x[t1][x - time + 5][z + 4];
                float r14 = -v_sol_x[t1][x - time + 3][z + 4] + v_sol_x[t1][x - time + 4][z + 4];
                float r13 = v_sol_z[t1][x - time + 4][z + 2] - v_sol_z[t1][x - time + 4][z + 5];
                float r12 = -v_sol_z[t1][x - time + 4][z + 3] + v_sol_z[t1][x - time + 4][z + 4];
                float r11 = 1.0 / h_z;
                float r10 = 1.0 / h_x;
                tau_sol_xx[t1][x - time + 4][z + 4] = r10 * (9.54594154601839F * r14 + 3.53553390572694e-1F * r15) + r11 * (4.7729707730092F * r12 + 1.76776695286347e-1F * r13) + tau_sol_xx[t0][x - time + 4][z + 4];
                tau_sol_xz[t1][x - time + 4][z + 4] = r10 * (8.83883476431735e-2F * (v_sol_z[t1][x - time + 3][z + 4] - v_sol_z[t1][x - time + 6][z + 4]) + 2.3864853865046F * (-v_sol_z[t1][x - time + 4][z + 4] + v_sol_z[t1][x - time + 5][z + 4])) + r11 * (8.83883476431735e-2F * (v_sol_x[t1][x - time + 4][z + 3] - v_sol_x[t1][x - time + 4][z + 6]) + 2.3864853865046F * (-v_sol_x[t1][x - time + 4][z + 4] + v_sol_x[t1][x - time + 4][z + 5])) + tau_sol_xz[t0][x - time + 4][z + 4];
                tau_sol_zz[t1][x - time + 4][z + 4] = r10 * (4.7729707730092F * r14 + 1.76776695286347e-1F * r15) + r11 * (9.54594154601839F * r12 + 3.53553390572694e-1F * r13) + tau_sol_zz[t0][x - time + 4][z + 4];
              }
              for (int sp_zi = sp_zi_m; sp_zi <= nnz_sp_source_mask[x - time] - 1; sp_zi += 1)
              {
                //printf("\n Source_injection at : ");
                int zind = sp_source_mask[x - time][sp_zi];
                float r0 = save_src_fxx[((time / sf) % (time_M - time_m + 1))][source_id[x - time][zind]] * source_mask[x - time][zind];
                float r1 = save_src_fzz[((time / sf) % (time_M - time_m + 1))][source_id[x - time][zind]] * source_mask[x - time][zind];
                tau_sol_xx[t1][x - time + 4][zind + 4] += r0;
                tau_sol_zz[t1][x - time + 4][zind + 4] += r1;
                //printf(" Time %d , at : %d, %d \n", tw, x - time + 4, zind + 4);
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
