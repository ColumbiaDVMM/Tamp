#!/bin/bash

CAFFE_PATH="caffe"

INTERFACE=$(find structured/interface/caffe -type f -name *.cpp)
SRCS=$(find structured/src -type f -name *.cpp)
INCLUDE="-I/opt/intel/composer_xe_2013_sp1.2.144/mkl/include -I/opt/intel/composer_xe_2013_sp1.2.144/mkl/include/fftw" 
LIBRARY="-L/opt/intel/composer_xe_2013_sp1.2.144/compiler/lib/intel64 -L/opt/intel/composer_xe_2013_sp1.2.144/mkl/lib/intel64"
LIBS=""

g++ -std=c++11 -shared $INTERFACE $SRCS -o libtampc.so -fPIC $INCLUDE -I $CAFFE_PATH/include -I $CAFFE_PATH/.build_release/src -I . $LIBRARY -MMD -MP -pthread -DCAFFE_VERSION=1.0.0-rc3 -DNDEBUG -DUSE_LEVELDB -DUSE_LMDB -DUSE_MKL $LIBS

exit 0
