#ifndef _STRUCTURED_TYPED_DATA_TENSOR_
#define _STRUCTURED_TYPED_DATA_TENSOR_

#include "tensorflow/core/framework/tensor.h"
#include "structured/lib/TypedData.h"

namespace structured {
  
template <typename T>
struct TypedDataTensor : TypedData<T> {
  TypedDataTensor() = default;
  inline TypedDataTensor(const tensorflow::Tensor &tensor): _tensor(tensor) { }
  inline TypedDataTensor(tensorflow::Tensor &&tensor): _tensor(tensor) { }
  inline TypedDataTensor(const TypedDataTensor<T> &typeddata):
    _tensor(typeddata._tensor) { }
  inline TypedDataTensor(TypedDataTensor<T> &&typeddata):
    _tensor(std::move(typeddata._tensor)) { }
  virtual const T * data() const {
    auto _vector = this->_tensor.template flat<T>();
    return _vector.data();
  }
  virtual T * data() {
    auto _vector = this->_tensor.template flat<T>();
    return _vector.data();
  }
  virtual int64 count() const {
    return this->_tensor.NumElements();
  }
  virtual int dims() const { return _tensor.dims();  }
  virtual int64 dim_size(int index) const { return _tensor.dim_size(index); }
  inline tensorflow::Tensor& getTensor() { return _tensor; }
  virtual void reshape(const std::vector<int64> shape) { }
  virtual std::shared_ptr<BufferedData>fromBuffer(void* buf) const { }
protected:
  tensorflow::Tensor _tensor;  
};

  
template <typename T>
struct TypedDataTensorShape : TypedData<T>, tensorflow::TensorShape {
  TypedDataTensorShape() = default;
  inline TypedDataTensorShape(const tensorflow::TensorShape &tensorShape):
    tensorflow::TensorShape(tensorShape) { }
  inline TypedDataTensorShape(tensorflow::TensorShape &&tensorShape):
    tensorflow::TensorShape(tensorShape) { }
  inline TypedDataTensorShape(const TypedDataTensorShape<T> &typeddata):
    tensorflow::TensorShape(typeddata._tensorShape) { }
  inline TypedDataTensorShape(TypedDataTensorShape<T> &&typeddata):
    tensorflow::TensorShape(std::move(typeddata._tensorShape)) { }
  virtual const T * data() const {
    return nullptr;
  }
  virtual T * data() {
    return nullptr;
  }
  virtual int64 count() const {
    return this->num_elements();
  }
  virtual int dims() const { return tensorflow::TensorShape::dims();  }
  virtual int64 dim_size(int index) const {
    return tensorflow::TensorShape::dim_size(index);
  }
  virtual void reshape(const std::vector<int64> shape) {
    this->Clear();
    for( int64 size : shape ) this->AddDim(size);
  }
  virtual std::shared_ptr<BufferedData>fromBuffer(void* buf) const {
    return std::shared_ptr<BufferedData>(
      new TypedDataTensor<T>(*static_cast<tensorflow::Tensor*>(buf)));
  };
};

}

#endif
