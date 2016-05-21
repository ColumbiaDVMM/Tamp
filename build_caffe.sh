#!/bin/bash

CAFFE_PATH="caffe"

INTERFACE=$(echo structured/interface/caffe/layerop.cpp)
SRCS=$(find structured/src -type f -name *.cpp)
INCLUDE=""
LIBRARY=""
LIBS=""

g++ -std=c++11 $INTERFACE $SRCS -c -fPIC $INCLUDE -I $CAFFE_PATH/include -I $CAFFE_PATH/.build_release/src -I . $LIBRARY -DUSE_FFTW $LIBS
ar rcs layer_op.a *.o

exit 0
