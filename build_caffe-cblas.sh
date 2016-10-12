#!/bin/bash

set -e

CAFFE_PATH="caffe"
CC=gcc-4.9
KERNELS=$(find structured/interface/caffe structured/src -type f -name *.cu)
SRCS=$(find structured/interface/caffe structured/src -type f -name *.cpp)
OBJS="layerop.o circulant.o"
INCLUDE="-I/usr/local/cuda/include -I/opt/gflags/include -I/opt/glog/include -I/opt/protobuf/include -I/usr/local/include -I/usr/include/hdf5/serial"
LIBRARY="-L/opt/gflags/lib -L/opt/glog/lib -L/opt/protobuf/lib -L/usr/local/lib -L/usr/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial"
LIBS=""

nvcc -ccbin=$CC -c -D__CUDACC__ -std=c++11 -Xcompiler -fPIC -I . -I $CAFFE_PATH/include -I $CAFFE_PATH/.build_release/src $INCLUDE $KERNELS

$CC -std=c++11 -shared $SRCS $OBJS -o libtampc.so -fPIC -I . -I $CAFFE_PATH/include -I $CAFFE_PATH/.build_release/src $INCLUDE $LIBRARY -MMD -MP -pthread -DCAFFE_VERSION=1.0.0-rc3 -DNDEBUG -DUSE_LEVELDB -DUSE_LMDB $LIBS

exit 0
