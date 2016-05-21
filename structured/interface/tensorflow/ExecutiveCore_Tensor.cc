#include "structured/interface/tensorflow/TypedData_Tensor.h"
#include "structured/interface/tensorflow/ExecutiveCore_Tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <complex>

using namespace tensorflow;
using namespace std;

namespace structured {

template <> shared_ptr<TypedData<float> >
ExecutiveCore::allocateBuffer(const vector<int64>& shape) {
  auto & tensorCore = dynamic_cast<ExecutiveCoreTensor&>(*this);
  TensorShape tensorShape;
  for( int64 dim : shape ) tensorShape.AddDim(dim);
  auto * buffer = new TypedDataTensor<float>;
  tensorCore.context->allocate_temp(DT_FLOAT, tensorShape, &( buffer->getTensor() ));
  return shared_ptr<TypedData<float> >(buffer);
}

template <> shared_ptr<TypedData<double> >
ExecutiveCore::allocateBuffer(const vector<int64>& shape) {
  auto & tensorCore = dynamic_cast<ExecutiveCoreTensor&>(*this);
  TensorShape tensorShape;
  for( int64 dim : shape ) tensorShape.AddDim(dim);
  auto * buffer = new TypedDataTensor<double>;
  tensorCore.context->allocate_temp(DT_DOUBLE, tensorShape, &( buffer->getTensor() ));
  return shared_ptr<TypedData<double> >(buffer);
}

template <> shared_ptr<TypedData<std::complex<float> > >
ExecutiveCore::allocateBuffer(const vector<int64>& shape) {
  auto & tensorCore = dynamic_cast<ExecutiveCoreTensor&>(*this);
  TensorShape tensorShape;
  for( int64 dim : shape ) tensorShape.AddDim(dim);
  auto * buffer = new TypedDataTensor<std::complex<float> >;
  tensorCore.context->allocate_temp(DT_COMPLEX64, tensorShape, &( buffer->getTensor() ));
  return shared_ptr<TypedData<std::complex<float> > >(buffer);
}

template <> shared_ptr<TypedData<std::complex<double> > >
ExecutiveCore::allocateBuffer(const vector<int64>& shape) {
  auto & tensorCore = dynamic_cast<ExecutiveCoreTensor&>(*this);
  TensorShape tensorShape;
  for( int64 dim : shape ) tensorShape.AddDim(dim);
  auto * buffer = new TypedDataTensor<std::complex<double> >;
  tensorCore.context->allocate_temp(DT_COMPLEX128, tensorShape, &( buffer->getTensor() ));
  return shared_ptr<TypedData<std::complex<double> > >(buffer);
}

}
