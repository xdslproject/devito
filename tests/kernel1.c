  /* Begin section0 */
  #pragma omp parallel num_threads(nthreads_nonaffine)
  {
    int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(-i1x_ltkn - i1x_rtkn + x_M - x_m + 1)/nthreads_nonaffine));
    #pragma omp for collapse(1) schedule(dynamic,chunk_size)
    for (int i1x = i1x_ltkn + x_m; i1x <= -i1x_rtkn + x_M; i1x += 1)
    {
      #pragma omp simd aligned(f,g2:32)
      for (int i1y = i1y_ltkn + y_m; i1y <= -i1y_rtkn + y_M; i1y += 1)
      {
        if (4.0e-1F*g[i1x + 1][i1y + 1] + 6.0e-1F*g2[i1x + 1][i1y + 1] < 3)
        {
          f[i1x + 1][i1y + 1] = 5;
        }
      }
    }
  }
  /* End section0 */