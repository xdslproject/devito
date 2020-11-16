#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "openacc.h"

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

extern "C" int Kernel(struct dataobj *restrict block_sizes_vec, struct dataobj *restrict damp_vec, const float dt, const float h_x, const float h_y, const float h_z, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict save_src_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict usol_vec, struct dataobj *restrict vp_vec, const int sp_zi_m, const int time_M, const int time_m, struct profiler * timers, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m);


int Kernel(struct dataobj *restrict block_sizes_vec, struct dataobj *restrict damp_vec, const float dt, const float h_x, const float h_y, const float h_z, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict save_src_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict usol_vec, struct dataobj *restrict vp_vec, const int sp_zi_m, const int time_M, const int time_m, struct profiler * timers, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m)
{
  int (*restrict block_sizes) __attribute__ ((aligned (64))) = (int (*)) block_sizes_vec->data;
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  int (*restrict nnz_sp_source_mask)[nnz_sp_source_mask_vec->size[1]] __attribute__ ((aligned (64))) = (int (*)[nnz_sp_source_mask_vec->size[1]]) nnz_sp_source_mask_vec->data;
  float (*restrict save_src)[save_src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[save_src_vec->size[1]]) save_src_vec->data;
  int (*restrict source_id)[source_id_vec->size[1]][source_id_vec->size[2]] __attribute__ ((aligned (64))) = (int (*)[source_id_vec->size[1]][source_id_vec->size[2]]) source_id_vec->data;
  float (*restrict source_mask)[source_mask_vec->size[1]][source_mask_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[source_mask_vec->size[1]][source_mask_vec->size[2]]) source_mask_vec->data;
  int (*restrict sp_source_mask)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]] __attribute__ ((aligned (64))) = (int (*)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]]) sp_source_mask_vec->data;
  float (*restrict usol)[usol_vec->size[1]][usol_vec->size[2]][usol_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[usol_vec->size[1]][usol_vec->size[2]][usol_vec->size[3]]) usol_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  #pragma acc enter data copyin(usol[0:usol_vec->size[0]][0:usol_vec->size[1]][0:usol_vec->size[2]][0:usol_vec->size[3]])
  #pragma acc enter data copyin(damp[0:damp_vec->size[0]][0:damp_vec->size[1]][0:damp_vec->size[2]])
  #pragma acc enter data copyin(nnz_sp_source_mask[0:nnz_sp_source_mask_vec->size[0]][0:nnz_sp_source_mask_vec->size[1]])
  #pragma acc enter data copyin(save_src[0:save_src_vec->size[0]][0:save_src_vec->size[1]])
  #pragma acc enter data copyin(source_id[0:source_id_vec->size[0]][0:source_id_vec->size[1]][0:source_id_vec->size[2]])
  #pragma acc enter data copyin(source_mask[0:source_mask_vec->size[0]][0:source_mask_vec->size[1]][0:source_mask_vec->size[2]])
  #pragma acc enter data copyin(sp_source_mask[0:sp_source_mask_vec->size[0]][0:sp_source_mask_vec->size[1]][0:sp_source_mask_vec->size[2]])
  #pragma acc enter data copyin(vp[0:vp_vec->size[0]][0:vp_vec->size[1]][0:vp_vec->size[2]])

  //int xb_size = block_sizes[0];
  //int y0_blk0_size = block_sizes[3];
  //int x0_blk0_size = block_sizes[2];
  //int yb_size = block_sizes[1];
  for (int time = time_m, t2 = (time + 2)%(3), t1 = (time)%(3), t0 = (time + 1)%(3); time <= time_M; time += 1, t2 = (time + 2)%(3), t1 = (time)%(3), t0 = (time + 1)%(3))
  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
#pragma acc parallel loop collapse(3)
{
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
        for (int z = z_m; z <= z_M; z += 1)
        {
          float r9 = -2.5F*usol[t1][x + 4][y + 4][z + 4];
          float r8 = 1.0/dt;
          float r7 = 1.0/(dt*dt);
          float r6 = 1.0/(vp[x + 4][y + 4][z + 4]*vp[x + 4][y + 4][z + 4]);
          usol[t0][x + 4][y + 4][z + 4] = (r6*(-r7*(-2.0F*usol[t1][x + 4][y + 4][z + 4] + usol[t2][x + 4][y + 4][z + 4])) + r8*(damp[x + 1][y + 1][z + 1]*usol[t1][x + 4][y + 4][z + 4]) + (r9 - 8.33333333e-2F*(usol[t1][x + 4][y + 4][z + 2] + usol[t1][x + 4][y + 4][z + 6]) + 1.33333333F*(usol[t1][x + 4][y + 4][z + 3] + usol[t1][x + 4][y + 4][z + 5]))/((h_z*h_z)) + (r9 - 8.33333333e-2F*(usol[t1][x + 4][y + 2][z + 4] + usol[t1][x + 4][y + 6][z + 4]) + 1.33333333F*(usol[t1][x + 4][y + 3][z + 4] + usol[t1][x + 4][y + 5][z + 4]))/((h_y*h_y)) + (r9 - 8.33333333e-2F*(usol[t1][x + 2][y + 4][z + 4] + usol[t1][x + 6][y + 4][z + 4]) + 1.33333333F*(usol[t1][x + 3][y + 4][z + 4] + usol[t1][x + 5][y + 4][z + 4]))/((h_x*h_x)))/(r6*r7 + r8*damp[x + 1][y + 1][z + 1]);
	}
     }
   }
}
#pragma acc parallel loop collapse(2)
{
    for (int x = x_m; x <= x_M; x += 1)
    {
      for (int y = y_m; y <= y_M; y += 1)
      {
	int sp_zi_M = nnz_sp_source_mask[x][y] - 1;
        #pragma acc loop vector
	for (int sp_zi = sp_zi_m; sp_zi <= sp_zi_M; sp_zi += 1)
        {
        int zind = sp_source_mask[x][y][sp_zi];
	int id = source_id[x][y][zind];
        float r0 = save_src[time][id]*source_mask[x][y][zind];
        usol[t0][x + 4][y + 4][zind + 4] += r0;
        }
      }
    }
}
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
  }

  #pragma acc exit data copyout(usol[0:usol_vec->size[0]][0:usol_vec->size[1]][0:usol_vec->size[2]][0:usol_vec->size[3]])
  #pragma acc exit data delete(usol[0:usol_vec->size[0]][0:usol_vec->size[1]][0:usol_vec->size[2]][0:usol_vec->size[3]])
  #pragma acc exit data delete(damp[0:damp_vec->size[0]][0:damp_vec->size[1]][0:damp_vec->size[2]])
  #pragma acc exit data delete(nnz_sp_source_mask[0:nnz_sp_source_mask_vec->size[0]][0:nnz_sp_source_mask_vec->size[1]])
  #pragma acc exit data delete(save_src[0:save_src_vec->size[0]][0:save_src_vec->size[1]])
  #pragma acc exit data delete(source_id[0:source_id_vec->size[0]][0:source_id_vec->size[1]][0:source_id_vec->size[2]])
  #pragma acc exit data delete(source_mask[0:source_mask_vec->size[0]][0:source_mask_vec->size[1]][0:source_mask_vec->size[2]])
  #pragma acc exit data delete(sp_source_mask[0:sp_source_mask_vec->size[0]][0:sp_source_mask_vec->size[1]][0:sp_source_mask_vec->size[2]])
  #pragma acc exit data delete(vp[0:vp_vec->size[0]][0:vp_vec->size[1]][0:vp_vec->size[2]])
  return 0;
}
