#include "structured/interface/caffe/TypedData_Caffe.h"
#include "structured/interface/caffe/ExecutiveCore_Caffe.h"
#include <complex>

using namespace std;

namespace structured {

template <> shared_ptr<TypedData<float> >
ExecutiveCore::allocateBuffer(const vector<int64>& shape) {
  auto * buffer = new TypedDataCaffe<float>;
  buffer->reshape(shape);
  return shared_ptr<TypedData<float>>(buffer);
}

template <> shared_ptr<TypedData<double> >
ExecutiveCore::allocateBuffer(const vector<int64>& shape) {
  auto * buffer = new TypedDataCaffe<double>;
  buffer->reshape(shape);
  return shared_ptr<TypedData<double>>(buffer);
}

template <> shared_ptr<TypedData<std::complex<float> > >
ExecutiveCore::allocateBuffer(const vector<int64>& shape) {
  auto * buffer = new TypedDataCaffe<std::complex<float>>;
  buffer->reshape(shape);
  return shared_ptr<TypedData<std::complex<float>>>(buffer);
}

template <> shared_ptr<TypedData<std::complex<double> > >
ExecutiveCore::allocateBuffer(const vector<int64>& shape) {
  auto * buffer = new TypedDataCaffe<std::complex<double>>;
  buffer->reshape(shape);
  return shared_ptr<TypedData<std::complex<double>>>(buffer);
}

}
