/* -*- mode: C; tab-width: 2; indent-tabs-mode: nil; -*- */

/*
 * This code has been contributed by the DARPA HPCS program.  Contact
 * David Koester <dkoester@mitre.org> or Bob Lucas <rflucas@isi.edu>
 * if you have questions.
 *
 * GUPS (Giga UPdates per Second) is a measurement that profiles the memory
 * architecture of a system and is a measure of performance similar to MFLOPS.
 * The HPCS HPCchallenge RandomAccess benchmark is intended to exercise the
 * GUPS capability of a system, much like the LINPACK benchmark is intended to
 * exercise the MFLOPS capability of a computer.  In each case, we would
 * expect these benchmarks to achieve close to the "peak" capability of the
 * memory system. The extent of the similarities between RandomAccess and
 * LINPACK are limited to both benchmarks attempting to calculate a peak system
 * capability.
 *
 * GUPS is calculated by identifying the number of memory locations that can be
 * randomly updated in one second, divided by 1 billion (1e9). The term "randomly"
 * means that there is little relationship between one address to be updated and
 * the next, except that they occur in the space of one half the total system
 * memory.  An update is a read-modify-write operation on a table of 64-bit words.
 * An address is generated, the value at that address read from memory, modified
 * by an integer operation (add, and, or, xor) with a literal value, and that
 * new value is written back to memory.
 *
 * We are interested in knowing the GUPS performance of both entire systems and
 * system subcomponents --- e.g., the GUPS rating of a distributed memory
 * multiprocessor the GUPS rating of an SMP node, and the GUPS rating of a
 * single processor.  While there is typically a scaling of FLOPS with processor
 * count, a similar phenomenon may not always occur for GUPS.
 *
 * For additional information on the GUPS metric, the HPCchallenge RandomAccess
 * Benchmark,and the rules to run RandomAccess or modify it to optimize
 * performance -- see http://icl.cs.utk.edu/hpcc/
 *
 */

/*
 * This file contains the computational core of the single cpu version
 * of GUPS.  The inner loop should easily be vectorized by compilers
 * with such support.
 *
 * This core is used by both the single_cpu and star_single_cpu tests.
 */

#include <hpcc.h>
extern "C" {
int HPCC_MPIRandomAccess(HPCC_Params *params);
}
#include <sys/time.h>
#include <sys/resource.h>
#include "RandomAccess.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdint.h>
/* Number of updates to table (suggested: 4x number of table entries) */
#define NUPDATE (4 * TableSize)

static __constant__ uint64_t c_m2[64];
static __device__ uint32_t d_error[1];

double CPUSEC_SAME()
{
   static double              cps = CLOCKS_PER_SEC;
   double                     d;
   clock_t                    t1;
   static clock_t             t0 = 0;
 
   if( t0 == 0 ) t0 = clock();
   t1 = clock() - t0;
   d = (double)(t1) / cps;
   return( d );
}
double RTSEC_SAME()
{
   struct timeval             tp;
   static long                start=0, startu;

   if( !start )
   {
      (void) gettimeofday( &tp, NULL );
      start  = tp.tv_sec;
      startu = tp.tv_usec;
      return( HPL_rzero );
   }
   (void) gettimeofday( &tp, NULL );

   return( (double)( tp.tv_sec - start ) +
           ( (double)( tp.tv_usec-startu ) / 1000000.0 ) );
}

u64Int
HPCC_starts_SAME(s64Int n)
{
  int i, j;
  u64Int m2[64];
  u64Int temp, ran;

  while (n < 0) n += PERIOD;
  while (n > PERIOD) n -= PERIOD;
  if (n == 0) return 0x1;

  temp = 0x1;
  for (i=0; i<64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0);
    temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0);
  }

  for (i=62; i>=0; i--)
    if ((n >> i) & 1)
      break;

  ran = 0x2;
  while (i > 0) {
    temp = 0;
    for (j=0; j<64; j++)
      if ((ran >> j) & 1)
        temp ^= m2[j];
    ran = temp;
    i -= 1;
    if ((n >> i) & 1)
      ran = (ran << 1) ^ ((s64Int) ran < 0 ? POLY : 0);
  }

  return ran;
}

static __global__ void
d_init(size_t n, uint64_t *t)
{
  for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
    i += gridDim.x * blockDim.x) { 
    t[i] = i;
    }
}

static __device__ uint64_t
d_starts(size_t n)
{
  if (n == 0) {
    return 1;
  }

  int i = 63 - __clzll(n);

  uint64_t ran = 2;
  while (i > 0) {
    uint64_t temp = 0;
    for (int j = 0; j < 64; j++) {
      if ((ran >> j) & 1) {
        temp ^= c_m2[j];
      }
    }
    ran = temp;
    i -= 1;
    if ((n >> i) & 1) {
      ran = (ran << 1) ^ ((int64_t) ran < 0 ? POLY : 0);
    }
  }

  return ran;
}

__global__ void
d_bench(size_t n, uint64_t *t)
{
  size_t num_threads = gridDim.x * blockDim.x;
  size_t thread_num = blockIdx.x * blockDim.x + threadIdx.x;
  size_t start = thread_num * 4 * n / num_threads;
  size_t end = (thread_num + 1) * 4 * n / num_threads;
  uint64_t ran;
  ran = d_starts(start);
  for (ptrdiff_t i = thread_num; i < n; i += num_threads ) {
  //for (ptrdiff_t i = start; i < end; ++i) {
    ran = (ran << 1) ^ ((int64_t) ran < 0 ? POLY : 0);
    unsigned long long int *address, old, assumed;
    address = (unsigned long long int *)&t[ran & (n - 1)];
    old = *address;   
    do {
      assumed = old;
      old = atomicCAS(address, assumed, assumed ^ ran);
    } while  (assumed != old);
  }
}

static __global__ void
d_check(size_t TableSize, uint64_t *t)
{
  for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < TableSize;
       i += gridDim.x * blockDim.x) {
    if (t[i] != i) {
      atomicAdd(d_error, 1);
    }
  }
}
static void
starts()
{
  uint64_t m2[64];
  uint64_t temp = 1;
  for (ptrdiff_t i = 0; i < 64; i++) {
    m2[i] = temp;
    temp = (temp << 1) ^ ((int64_t) temp < 0 ? POLY : 0); 
    temp = (temp << 1) ^ ((int64_t) temp < 0 ? POLY : 0); 
  }
  cudaMemcpyToSymbol(c_m2, m2, sizeof(m2));
}


static void
RandomAccessUpdate(u64Int TableSize, u64Int *Table) {
  

  //d_init<<<grid, thread>>>(TableSize, d_t);
  //d_bench<<<grid, thread>>>(TableSize, d_t);



}
extern "C"
int
HPCC_RandomAccess(HPCC_Params *params, int doIO, double *GUPs, int *failure) {
  u64Int i;
  u64Int temp;
  double cputime;               /* CPU time to update table */
  double realtime;              /* Real time to update table */
  double totalMem;
  u64Int *Table;
  u64Int logTableSize, TableSize;
  FILE *outFile = NULL;

  if (doIO) {
    outFile = fopen( params->outFname, "a" );
    if (! outFile) {
      outFile = stderr;
      fprintf( outFile, "Cannot open output file.\n" );
      return 1;
    }
  }

  /* calculate local memory per node for the update table */
  totalMem = params->HPLMaxProcMem;
  totalMem /= sizeof(u64Int);

  /* calculate the size of update array (must be a power of 2) */
  for (totalMem *= 0.5, logTableSize = 0, TableSize = 1;
       totalMem >= 1.0;
       totalMem *= 0.5, logTableSize++, TableSize <<= 1)
    ; /* EMPTY */

  Table = HPCC_XMALLOC( u64Int, TableSize );
  if (! Table) {
    if (doIO) {
      fprintf( outFile, "Failed to allocate memory for the update table (" FSTR64 ").\n", TableSize);
      fclose( outFile );
    }
    return 1;
  }
  params->RandomAccess_N = (s64Int)TableSize;

  /* Print parameters for run */
  if (doIO) {
  fprintf( outFile, "Main table size   = 2^" FSTR64 " = " FSTR64 " words\n", logTableSize,TableSize);
  fprintf( outFile, "Number of updates = " FSTR64 "\n", NUPDATE);
  }

  /* Initialize main table */
  for (i=0; i<TableSize; i++) Table[i] = i;

  /* Initialize gpu */
  
  starts();

  int ndev;
  cudaGetDeviceCount(&ndev);
  int dev = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  cudaSetDevice(dev);
  uint64_t *d_t;
  cudaMalloc((void **)&d_t, TableSize * sizeof(uint64_t)) ;

  dim3 grid(prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.warpSize));
  dim3 thread(prop.warpSize);

  d_init<<<grid, thread>>>(TableSize, d_t);


  /* Begin timing here */
  cputime = -CPUSEC_SAME();
  realtime = -RTSEC_SAME();

  d_bench<<<grid, thread>>>(TableSize, d_t); // core
  cudaMemcpy(Table, d_t, sizeof(uint64_t) * TableSize, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();//@lq
  //RandomAccessUpdate( TableSize, Table );

  /* End timed section */
  cputime += CPUSEC_SAME();
  realtime += RTSEC_SAME();

  /* make sure no division by zero */
  *GUPs = (realtime > 0.0 ? 1.0 / realtime : -1.0);
  *GUPs *= 1e-9*NUPDATE;
  /* Print timing results */
  if (doIO) {
  fprintf( outFile, "CPU time used  = %.6f seconds\n", cputime);
  fprintf( outFile, "Real time used = %.6f seconds\n", realtime);
  fprintf( outFile, "%.9f Billion(10^9) Updates    per second [GUP/s]\n", *GUPs );
  }

  /* Verification of results (in serial or "safe" mode; optional) */
  temp = 0x1;
  for (i=0; i<NUPDATE; i++) {
    temp = (temp << 1) ^ (((s64Int) temp < 0) ? POLY : 0);
    Table[temp & (TableSize-1)] ^= temp;
  }

  temp = 0;
  for (i=0; i<TableSize; i++)
    if (Table[i] != i)
      temp++;

  if (doIO) {
  fprintf( outFile, "Found " FSTR64 " errors in " FSTR64 " locations (%s).\n",
           temp, TableSize, (temp <= 0.01*TableSize) ? "passed" : "failed");
  }
  if (temp <= 0.01*TableSize) *failure = 0;
  else *failure = 1;

  HPCC_free( Table );

  if (doIO) {
    fflush( outFile );
    fclose( outFile );
  }




   //debug
  d_bench<<<grid, thread>>>(TableSize, d_t);
  void *p_error;
  cudaGetSymbolAddress(&p_error, d_error);
  cudaMemset(d_error, 0, sizeof(uint32_t));
  d_check<<<grid, thread>>>(TableSize, d_t);
  uint32_t h_error;
  cudaMemcpy(&h_error, p_error, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  printf("Verification: Found %u errors.\n", h_error);
  //

  cudaFree(d_t);

  return 0;
}

