#!/bin/bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

INTERFACE=$(echo structured/interface/tensorflow/{ExecutiveCore_Tensor.cc,layer_op.cc,layer_op_kernel.cc} structured/lib/caffe/util/math_functions.cpp)
SRCS=$(find structured/src -type f -name *.cpp)
INCLUDE=""
LIBRARY=""
LIBS="-lcblas -latlas -lfftw3 -lfftw3f"

g++ -std=c++11 -shared $INTERFACE $SRCS -o layer_op.so -fPIC $INCLUDE -I $TF_INC -I . $LIBRARY -DUSE_FFTW $LIBS

exit 0
