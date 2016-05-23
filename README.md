# Tamp

Tamp is an open source C++ library for reducing the space and time costs of deep neural networks, with the goal of making them more applicable to devices with limited resources.

## Getting Started with TensorFlow

We recommend an virtualenv installation of TensorFlow. Detail tutorial can be found [here](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)

After successfully installed tensorflow, activate virtualenv by:

```
$ source ~/tensorflow/bin/activate  # If using bash
```

Or

```
$ source ~/tensorflow/bin/activate.csh  # If using csh
```

After that, using build script:

```
$ ./build_tensorflow.sh  # If linking against Intel MKL
```

Or

```
$ ./build_tensorflow-cblas.sh  # If linking against CBLAS/ATLAS
```

## Getting Started with Caffe

Using the our forked version of Caffe is required since the vanilla Caffe does not have an interface for Tamp-styled plugins in original version.

First, follow the [instructions](http://caffe.berkeleyvision.org/installation.html) to build the Caffe environment. Some required protocol buffer headers are generated during this build process for Tamp integration.

Next, using build script, run:

```
$ ./build_caffe.sh
```

Once this is done, turn on Tamp by setting USE_TAMP in Caffe's Makefile.config to include the Tamp library and rebuild Caffe.

## More Help

Online documants can be found at [here](https://github.com/wenri/tamp). Additional documentation is available [here](https://github.com/wenri/tamp/docs) are avaliable.

