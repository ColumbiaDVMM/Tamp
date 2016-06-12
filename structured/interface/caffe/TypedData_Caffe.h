#ifndef _STRUCTURED_TYPED_DATA_CAFFE_
#define _STRUCTURED_TYPED_DATA_CAFFE_

#include "caffe/blob.hpp"
#include "structured/lib/TypedData.h"

namespace structured {
  
template <typename T>
struct TypedDataCaffe : TypedData<T> {
  TypedDataCaffe() = default;
  inline TypedDataCaffe(const caffe::Blob<T> &blob): _blob(blob.shape()) {
    _blob.ShareData(blob);
  }
  inline TypedDataCaffe(caffe::Blob<T> &&blob): _blob(blob.shape()) {
    _blob.ShareData(blob);
  }
  inline TypedDataCaffe(const TypedDataCaffe<T> &typeddata):
    _blob(typeddata._blob.shape())
  {
    _blob.ShareData(typeddata._blob);
  }
  inline TypedDataCaffe(TypedDataCaffe<T> &&typeddata):
    _blob(typeddata._blob.shape())
  {
    _blob.ShareDate(std::move(typeddata._blob));
  }
  virtual const T * data() const {
    return _blob.cpu_data();
  }
  virtual T * data() {
    return _blob.mutable_cpu_data();
  }
  virtual int64 count() const {
    return this->_blob.count();
  }
  virtual int dims() const { return _blob.num_axes();  }
  virtual int64 dim_size(int index) const { return _blob.shape(index); }
  inline caffe::Blob<T>& getBlob() { return _blob; }
  virtual void reshape(const std::vector<int64> shape) {
    std::vector<int> caffeShape;
    for( int64 dim : shape ) caffeShape.push_back(dim);
    _blob.Reshape(caffeShape);
  }
  virtual std::shared_ptr<BufferedData>fromBuffer(void* buf) const { return nullptr; }
protected:
  caffe::Blob<T> _blob;
};


}

#endif
