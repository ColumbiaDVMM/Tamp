#ifndef _STRUCTURED_TYPED_DATA_CAFFE_
#define _STRUCTURED_TYPED_DATA_CAFFE_

#include "caffe/blob.hpp"
#include "structured/lib/TypedData.h"

namespace structured {
  
template <typename T>
struct TypedDataCaffe : TypedData<T> {
  TypedDataCaffe() = default;
  inline TypedDataCaffe(const Blob<T> &blob) {
    _blob.ShareData(blob);
  }
  inline TypedDataCaffe(Blob<T> &&caffe) {
    _blob.ShareData(blob);
  }
  inline TypedDataCaffe(const TypedDataCaffe<T> &typeddata) { }
  inline TypedDataCaffe(TypedDataCaffe<T> &&typeddata) { }
  virtual const T * data() const {
    return _blob.cpu_data();
  }
  virtual T * data() {
    return _blob.mutable_cpu_data();
  }
  virtual int64 count() const {
    return this->_blob.count();
  }
  virtual int dims() const { return _caffe.num_axes();  }
  virtual int64 dim_size(int index) const { return _caffe.shape(index); }
  inline Blob<T>& getBlob() { return _blob; }
  virtual void reshape(const std::vector<int64> shape) { }
  virtual std::shared_ptr<BufferedData>fromBuffer(void* buf) const { }
protected:
  Blob<T> _blob;  
};


}

#endif
