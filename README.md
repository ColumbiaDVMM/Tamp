# TampNet

Tamp is an open source C++ library for reducing the space and time costs of deep neural networks, with the goal of making them more applicable to devices with limited resources.

## Getting Started with TensorFlow

We recommanded an virtualenv installation of TensorFlow. Detail tutorial can be found here:

- [Download and Setup](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)

After successfully installed tensorflow, active virtualenv by:

$ source ~/tensorflow/bin/activate  # If using bash

Or

$ source ~/tensorflow/bin/activate.csh  # If using csh

After that, using build script:

$ ./build_tensorflow.sh  # If linking against Intel MKL

Or

$ ./build_tensorflow-cblas.sh  # If linking against CBLAS/ATLAS

After successfully build, a dynamic library of model operation can be found at current directory.

