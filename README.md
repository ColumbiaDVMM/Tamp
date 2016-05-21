# TampNet

Tamp is an open source C++ library for reducing the space and time costs of deep neural networks, with the goal of making them more applicable to devices with limited resources.

## Getting Started with TensorFlow

We recommanded an virtualenv installation of TensorFlow. Detail tutorial can be found [here](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)

After successfully installed tensorflow, active virtualenv by:

$ source ~/tensorflow/bin/activate  # If using bash

Or $ source ~/tensorflow/bin/activate.csh  # If using csh

After that, using build script:

$ ./build_tensorflow.sh  # If linking against Intel MKL

Or $ ./build_tensorflow-cblas.sh  # If linking against CBLAS/ATLAS

After successfully build, a dynamic library of model operation can be found at current directory.

## Getting Started with Caffe

Using the our special version of Caffe is necessary is required since Caffe does not have an interface for plugins in original version.

First of all, follow the [instruction](http://caffe.berkeleyvision.org/installation.html) to build an Caffe environment. We need some proto headers generated during building process to make Tamp working.

Next, using build script:

$ ./build_caffe.sh

To generate static linked library for models using Tamp.

Once this is done, turn on Tamp by setting USE_TAMP in Caffe's Makefile.config to include Tamp library and rebuild Caffe.


