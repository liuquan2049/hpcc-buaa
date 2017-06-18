#!/bin/sh
source /opt/intel/bin/compilervars.sh intel64

cd hpl/src/cuda
make clean
make -j all

cd ../../../

make clean
make -j all
ls
