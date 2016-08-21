#ifndef _STRUCTURED_LAYER_OP_
#define _STRUCTURED_LAYER_OP_

#include "structured/lib/TypedData.h"
#include "structured/lib/ProcessorTape.h"
#include "structured/interface/caffe/ExecutiveCore_Caffe.h"

namespace structured{

  struct CPUDevice{};
  struct GPUDevice{};
  
namespace functor{

template <typename Device>
struct LayerOpFunctor {
  void operator()(
		  const Device& d,
		  ExecutiveCore* core,
		  ProcessorTape* atape);
  void operator()(
		  const Device& d,
		  ExecutiveCore* core,
		  ProcessorTape* atape,
		  ProcessorTape* btape);
};

}
}


#endif
