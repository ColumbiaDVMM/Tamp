#ifndef _STRUCTURED_LAYER_OP_
#define _STRUCTURED_LAYER_OP_

#include "structured/lib/TypedData.h"
#include "structured/lib/ProcessorTape.h"
#include "structured/interface/tensorflow/ExecutiveCore_Tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include <unordered_map>

namespace structured{

extern std::unordered_map<std::string, ProcessorBase*> ProcessorMap;

namespace functor{

template <typename Device>
struct LayerOpFunctor {
  void operator()(
		  const Device& d,
		  CpuCoreTensor* core,
		  ProcessorTape* atape);
  void operator()(
		  const Device& d,
		  CpuCoreTensor* core,
		  ProcessorTape* atape,
		  ProcessorTape* btape);
};

}
}

#endif
