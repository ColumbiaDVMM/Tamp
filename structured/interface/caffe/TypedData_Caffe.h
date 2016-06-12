#ifndef _STRUCTURED_TYPED_DATA_CAFFE_
#define _STRUCTURED_TYPED_DATA_CAFFE_

#include "caffe/blob.hpp"
#include "structured/lib/TypedData.h"

namespace structured {
  
template <typename T>
struct TypedDataCaffe : TypedData<T>, caffe::Blob<T> {
  TypedDataCaffe() = default;
  inline TypedDataCaffe(const caffe::Blob<T> &blob): caffe::Blob<T>(blob.shape()) {
    this->ShareData(blob);
    this->ShareDiff(blob);
  }
  inline TypedDataCaffe(caffe::Blob<T> &&blob): caffe::Blob<T>(blob.shape()) {
    this->ShareData(blob);
    this->ShareDiff(blob);
  }
  virtual const T * data() const {
    return this->cpu_data();
  }
  virtual T * data() {
    return this->mutable_cpu_data();
  }
  virtual int64 count() const {
    return caffe::Blob<T>::count();
  }
  virtual int dims() const { return this->num_axes();  }
  virtual int64 dim_size(int index) const { return this->shape(index); }
  inline caffe::Blob<T>& getBlob() { return *this; }
  inline void swapBuffers() {
    swap(this->data_, this->diff_);
  }
  virtual void reshape(const std::vector<int64> shape) {
    std::vector<int> caffeShape;
    for( int64 dim : shape ) caffeShape.push_back(dim);
    this->Reshape(caffeShape);
  }
  virtual std::shared_ptr<BufferedData>fromBuffer(void* buf) const { return nullptr; }
};


}

#endif
