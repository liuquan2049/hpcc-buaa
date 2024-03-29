/* -*- mode: C; tab-width: 2; indent-tabs-mode: nil; fill-column: 79; coding: iso-latin-1-unix -*- */
/* tstdgemm.c
 */

#include <hpcc.h>
#include <sys/times.h>
#include <sys/time.h>
#include "cublas_v2.h"

/* Generates random matrix with entries between 0.0 and 1.0 */
static void
dmatgen(int m, int n, double *a, int lda, int seed) {
  int i, j;
  double *a0 = a, rcp = 1.0 / RAND_MAX;

  srand( seed );

  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++)
      a0[i] = rcp * rand();

    a0 += lda;
  }
}

static double
dnrm_inf(int m, int n, double *a, int lda) {
  int i, j, k, lnx;
  double mx, *a0;

  int nx = 10;
  double x[10];

  mx = 0.0;

  for (i = 0; i < m; i += nx) {
    lnx = Mmin( nx, m-i );
    for (k = 0; k < lnx; ++k) x[k] = 0.0;

    a0 = a + i;

    for (j = 0; j < n; ++j) {
      for (k = 0; k < lnx; ++k)
        x[k] += fabs( a0[k] );

      a0 += lda;
    }

    for (k = 0; k < lnx; ++k)
      if (mx < x[k]) mx = x[k];
  }

  return mx;
}

double wallclock(void)
{
  struct timeval tv;
  struct timezone tz;
  double t;

  gettimeofday(&tv, &tz);
  t = (double)tv.tv_sec;
  t += ((double)tv.tv_usec)/1000000.0;

  return t;
}


int
HPCC_TestDGEMM(HPCC_Params *params, int doIO, double *UGflops, int *Un, int *Ufailure) {
  int myid;
  int n, lda, ldb, ldc, failure = 1;
  double *a, *b, *c, *x, *y, *z, alpha, beta, sres, cnrm, xnrm;
  double *d_a, *d_b, *d_c;
  double Gflops = 0.0, dn, t0, t1;
  long l_n;
  FILE *outFile;
  int seed_a, seed_b, seed_c, seed_x;
  cublasStatus_t stat;
  cublasHandle_t handle;

  if (doIO) {
    outFile = fopen( params->outFname, "a" );
    if (! outFile) {
      outFile = stderr;
      fprintf( outFile, "Cannot open output file.\n" );
      return 1;
    }
  }

  n = (int)(sqrt( params->HPLMaxProcMem / sizeof(double) / 3 + 0.25 ) - 0.5);
  if (n < 0) n = -n; /* if 'n' has overflown an integer */
  l_n = n;
  lda = ldb = ldc = n;

  a = HPCC_XMALLOC( double, l_n * l_n );
  b = HPCC_XMALLOC( double, l_n * l_n );
  c = HPCC_XMALLOC( double, l_n * l_n );

  x = HPCC_XMALLOC( double, l_n );
  y = HPCC_XMALLOC( double, l_n );
  z = HPCC_XMALLOC( double, l_n );

  if (! a || ! b || ! c || ! x || ! y || ! z) {
    goto comp_end;
  }

  seed_a = (int)time( NULL );
  dmatgen( n, n, a, n, seed_a );

  seed_b = (int)time( NULL );
  dmatgen( n, n, b, n, seed_b );

  seed_c = (int)time( NULL );
  dmatgen( n, n, c, n, seed_c );

  seed_x = (int)time( NULL );
  dmatgen( n, 1, x, n, seed_x );

  alpha = a[n / 2];
  beta  = b[n / 2];

//@lq add20170531

  MPI_Comm_rank(MPI_COMM_WORLD,&myid); 
  if (myid % 2)
    cudaSetDevice(1);
  else
    cudaSetDevice(0);
  printf("myid : %d\n",myid); //@hpccdebug
  printf("N    : %d\n",n);    //@hpccdebug

  cudaMalloc((void**)&d_a,(n*n+2)*sizeof(double));
  cudaMalloc((void**)&d_b,n*n*sizeof(double));
  cudaMalloc((void**)&d_c,n*n*sizeof(double));

  //t0 = MPI_Wtime();
  cudaMemcpy(d_a,a,sizeof(double)*n*n,cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,b,sizeof(double)*n*n,cudaMemcpyHostToDevice);
  cudaMemcpy(d_c,c,sizeof(double)*n*n,cudaMemcpyHostToDevice);
  stat = cublasCreate(&handle);

  t0 = MPI_Wtime();
//  HPL_dgemm( HplColumnMajor, HplNoTrans, HplNoTrans, n, n, n, alpha, a, n, b, n, beta, c, n );
//  cublasDgemm(CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,d_a,n,d_b,n,&beta,d_c,n);
  cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&alpha,d_a,n,d_b,n,&beta,d_c,n);
  cudaThreadSynchronize();
  t1 = MPI_Wtime();

  cudaMemcpy(c,d_c,sizeof(double)*n*n,cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle);
//  t1 = wallclock();
//  free(a);
//  free(b);
//  free(c);

  t1 -= t0;
  dn = (double)n;
  if (t1 != 0.0 && t1 != -0.0)
    Gflops = 2.0e-9 * dn * dn * dn / t1;
  else
    Gflops = 0.0;

  cnrm = dnrm_inf( n, n, c, n );
  xnrm = dnrm_inf( n, 1, x, n );

  /* y <- c*x */
  HPL_dgemv( HplColumnMajor, HplNoTrans, n, n, 1.0, c, ldc, x, 1, 0.0, y, 1 );

  /* z <- b*x */
  HPL_dgemv( HplColumnMajor, HplNoTrans, n, n, 1.0, b, ldb, x, 1, 0.0, z, 1 );

  /* y <- alpha * a * z - y */
  HPL_dgemv( HplColumnMajor, HplNoTrans, n, n, alpha, a, lda, z, 1, -1.0, y, 1 );

  dmatgen( n, n, c, n, seed_c );

  /* y <- beta * c_orig * x + y */
  HPL_dgemv( HplColumnMajor, HplNoTrans, n, n, beta, c, ldc, x, 1, 1.0, y, 1 );

  sres = dnrm_inf( n, 1, y, n ) / cnrm / xnrm / n / HPL_dlamch( HPL_MACH_EPS );

  if (doIO) fprintf( outFile, "Scaled residual: %g\n", sres );

  if (sres < params->test.thrsh)
    failure = 0;

  comp_end:

  if (z) HPCC_free( z );
  if (y) HPCC_free( y );
  if (x) HPCC_free( x );
  if (c) HPCC_free( c );
  if (b) HPCC_free( b );
  if (a) HPCC_free( a );

  if (doIO) {
    fflush( outFile );
    fclose( outFile );
  }

  if (UGflops) *UGflops = Gflops;
  if (Un) *Un = n;
  if (Ufailure) *Ufailure = failure;

  return 0;
}
