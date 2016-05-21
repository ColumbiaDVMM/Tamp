#ifndef _STRUCTURED_EXECUTIVE_CORE_TENSOR_
#define _STRUCTURED_EXECUTIVE_CORE_TENSOR_

#include "structured/lib/ExecutiveCore.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace structured {
  
struct ExecutiveCoreTensor: ExecutiveCore{
  friend struct ExecutiveCore;
  ExecutiveCoreTensor(ProcessorBase * processor,
       tensorflow::OpKernelContext* ctx, tensorflow::OpKernel* knl):
    ExecutiveCore(processor), context(ctx), kernel(knl) {}
  inline tensorflow::OpKernelContext* getContext() { return context; }
protected:
  tensorflow::OpKernelContext* context;
  tensorflow::OpKernel* kernel;
};

}

#endif
