#ifndef _STRUCTURED_LAYER_OP_
#define _STRUCTURED_LAYER_OP_

#include "structured/lib/TypedData.h"
#include "structured/lib/ProcessorTape.h"
#include "structured/interface/caffe/ExecutiveCore_Caffe.h"

namespace structured{

namespace functor{

template <typename Device>
struct LayerOpFunctor {
  void operator()(
		  const Device& d,
		  ExecutiveCoreTensor* core,
		  ProcessorTape* atape);
  void operator()(
		  const Device& d,
		  ExecutiveCoreTensor* core,
		  ProcessorTape* atape,
		  ProcessorTape* btape);
};

}
}


#endif
