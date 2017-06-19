#!/bin/bash

DATETIME=`hostname`-`date +"%m%d.%H%M%S"`

source /opt/intel/bin/compilervars.sh intel64

export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14

if [ -f ./hpccoutf.txt ];then
    echo file exit
#    mv  hpccoutf.txt results/hpccoutf-$DATE_TIME.txt
fi
#-----------------------------------------------

export LD_LIBRARY_PATH=./hpl/src/cuda:$LD_LIBRARY_PATH

CPU_CORES_PER_GPU=14
# FOR MKL
export MKL_NUM_THREADS=$CPU_CORES_PER_GPU
export MKL_DYNAMIC=FALSE
# FOR OMP
export OMP_NUM_THREADS=$CPU_CORES_PER_GPU

# hint: for 2050 or 2070 card
export CUDA_DGEMM_SPLIT=1.00
# hint: try CUDA_DGEMM_SPLIT - 0.10
export CUDA_DTRSM_SPLIT=0.90

# 1_node 1xP100
#./hpcc

# 1_node 2xP100
#mpirun -hosts gpu3,gpu4,gpu5  -n 6 -ppn 2 ./hpcc
#mpirun  -hosts gpu3,gpu4 -n 4 -ppn 2  ./hpcc 
#mv hpccoutf.txt   ./results/1_node/dgemmhpcc-${DATETIME}.txt

# 5_node 10xP100

mpirun -f host_5node -n 10 -ppn 2 ./hpcc

#mv hpccoutf.txt   ./results/5_node/stream/hpcc-${DATETIME}.txt
#echo mv hpccoutf.txt   ./results/5_node/stream/hpcc-${DATETIME}.txt

#-----------------------------------------------

END_TIME=`date +"%H%M%S"`

cat hpccoutf.txt | grep WR
