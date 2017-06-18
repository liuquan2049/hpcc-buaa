
----------
##  编译 

``` bash
$ source /opt/intel/bin/compilervars.sh intel64

$ cd hpcc_top_dir
$ make clean; make -j
```

编译器及选项修改: 

``` bash
$ vim hpl/Make.LinuxIntelIA64Itan2_eccMKL

Makefile修改:
$ vim hpl/lib/arch/build/Makefile.hpcc
```
---------
##  运行

``` bash
$ cd hpcc_top_dir
$ ./hpcc
```

运行脚本:
``` bash
$ ./run.sh  
```

---------
## TODO

1. RandomAccess
2. RandomAccessLCG
3. FFT
4. PTRANS



--------

## DGEMM

### 代码优化

修改 `DGEMM/tstdgemm.c` 中库函数，调用cublasDgemm

### 编译

修改 `hpl/lib/arch/build/Makefile.hpcc 中的编译器和选项

--------

### STREAM

### 代码优化

修改 `STREAM/stream.cu` 调用gpu计算 Copy, Scale, Add, Triad 函数

### 编译

- 修改 `hpl/lib/arch/build/Makefile.hpcc 中的编译器和选项
- nvcc 编译 .cu文件时基于g++ ; 在和基于gcc的编译编译器编译.c文件生成的 .o文件进行链接时会报错 undefined referance to ... 
  修改方式是 .c 调用 .cu中函数时，在 .cu中的函数前加 `extern "C"`; .cu调用.c函数时，暂未找到合适的解决方案，只能把函数放在 .cu文件中重写。

--------

## RandomAccess

cpu/gpu版本切换
hpl/lib/arch/build/Makefile.hpcc

--------
##    FFT

目前修改了`tstfft.c`，即对StarFFT SingleFFT进行优化

### 代码优化

类似DGEMM，修改 `FFT/tstfft.c`中 plan fft destroy等接口，调用cuFFT接口

### 编译

修改 `hpl/lib/arch/build/Makefile.hpcc 中的编译器和选项

### TODO

batch tune


--------
## HPL修改

### 优化

移植HPL-2.0_FERMI
直接使用支持cuda的dgemm动态链接库，位于 {TOPdir}/src/cuda 中

### TODO

写死 N NB
显存占用优化
