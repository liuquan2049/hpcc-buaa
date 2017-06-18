#!/bin/bash

DATETIME=`hostname`-`date +"%m%d.%H%M%S"`
START_TIME=`date +"%H%M%S"`

source /opt/intel/bin/compilervars.sh intel64

export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14

if [ -f ./hpccoutf.txt ];then
    mv  hpccoutf.txt results/
fi
#-----------------------------------------------

export LD_LIBRARY_PATH=./hpl/src/cuda:$LD_LIBRARY_PATH
# 1_node 1xP100
#./hpcc

# 1_node 2xP100
mpirun -hosts gpu3,gpu4,gpu5  -n 6 -ppn 2 ./hpcc
#mv hpccoutf.txt   ./results/1_node/dgemmhpcc-${DATETIME}.txt

# 5_node 10xP100
#mpirun -f host_5node -n 10 -ppn 2 ./hpcc
#mv hpccoutf.txt   ./results/5_node/stream/hpcc-${DATETIME}.txt
#echo mv hpccoutf.txt   ./results/5_node/stream/hpcc-${DATETIME}.txt

#-----------------------------------------------

END_TIME=`date +"%H%M%S"`

echo
echo run time is : $[${END_TIME}-${START_TIME}]
cat hpccoutf.txt | grep WR
