#!/bin/bash

CAFFE_PATH="caffe"

INTERFACE=$(find structured/interface/caffe -type f -name *.cpp)
KERNEL=$(find structured/interface/caffe -type f -name *.cu)
SRCS=$(find structured/src -type f -name *.cpp)
OBJS=layerop.o
INCLUDE="-I/usr/local/cuda/include" 
LIBRARY=""
LIBS=""

nvcc -c -D__CUDACC__ -std=c++11 -Xcompiler -fPIC $INCLUDE -I $CAFFE_PATH/include -I $CAFFE_PATH/.build_release/src -I . $KERNEL

g++ -std=c++11 -shared $INTERFACE $SRCS $OBJS -o libtampc.so -fPIC $INCLUDE -I $CAFFE_PATH/include -I $CAFFE_PATH/.build_release/src -I . $LIBRARY -MMD -MP -pthread -DCAFFE_VERSION=1.0.0-rc3 -DNDEBUG -DUSE_LEVELDB -DUSE_LMDB $LIBS

exit 0
