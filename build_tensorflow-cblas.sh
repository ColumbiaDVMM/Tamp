#!/bin/bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CC=gcc-4.9
# KERNELS=$(find structured/interface/tensorflow structured/src -type f -name *.cu)
INTERFACE=$(find structured/interface/tensorflow -type f -name *.cc)
SRCS=$(find structured/lib/caffe structured/src -type f -name *.cpp)
INCLUDE="-I/usr/local/cuda/include -I/opt/gflags/include -I/opt/glog/include -I/opt/protobuf/include"
LIBRARY="-L/opt/gflags/lib -L/opt/glog/lib -L/opt/protobuf/lib -L/usr/local/cuda/lib64"
LIBS="-lcblas -latlas -lfftw3 -lfftw3f -lprotobuf"

# nvcc -ccbin=$CC -c -D__CUDACC__ -std=c++11 -Xcompiler -fPIC $INCLUDE -I $TF_INC -I . $KERNELS

$CC -std=c++11 -shared $INTERFACE $SRCS -o layer_op.so -fPIC -I . -I $TF_INC $INCLUDE $LIBRARY -DUSE_FFTW $LIBS

exit 0
