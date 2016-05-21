#!/bin/bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

INTERFACE=$(echo structured/interface/tensorflow/{ExecutiveCore_Tensor.cc,layer_op.cc,layer_op_kernel.cc} structured/lib/caffe/util/math_functions.cpp)
SRCS=$(find structured/src -type f -name *.cpp)
INCLUDE="-I/opt/intel/composer_xe_2013_sp1.2.144/mkl/include -I/opt/intel/composer_xe_2013_sp1.2.144/mkl/include/fftw"
LIBRARY="-L/opt/intel/composer_xe_2013_sp1.2.144/compiler/lib/intel64 -L/opt/intel/composer_xe_2013_sp1.2.144/mkl/lib/intel64"
LIBS="-lmkl_rt"

g++ -std=c++11 -shared $INTERFACE $SRCS -o layer_op.so -fPIC $INCLUDE -I $TF_INC -I . $LIBRARY -DUSE_MKL $LIBS

exit 0
