#ifndef HPCC_STREAM_H
#define HPCC_STREAM_H

int HPCC_Stream(HPCC_Params *params, int doIO, MPI_Comm comm, int world_rank, double *copyGBs, double *scaleGBs, double *addGBs, double *triadGBs,  int *failure);

#endif
