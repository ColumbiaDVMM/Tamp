#ifndef _STRUCTURED_EXECUTIVE_CORE_TENSOR_
#define _STRUCTURED_EXECUTIVE_CORE_TENSOR_

#include "structured/lib/ExecutiveCore.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace structured {

struct TensorCore{
  friend struct ExecutiveCore;
  TensorCore(tensorflow::OpKernelContext* ctx, tensorflow::OpKernel* knl):
    context(ctx), kernel(knl) {}
  inline tensorflow::OpKernelContext* getContext() { return context; }
protected:
  tensorflow::OpKernelContext* context;
  tensorflow::OpKernel* kernel;
};

struct CpuCoreTensor: CpuCore, TensorCore{
  CpuCoreTensor(ProcessorBase * processor, tensorflow::OpKernelContext* ctx, tensorflow::OpKernel* knl): CpuCore(processor), TensorCore(ctx, knl) { }
};

struct GpuCoreTensor: GpuCore, TensorCore{
  GpuCoreTensor(ProcessorBase * processor, tensorflow::OpKernelContext* ctx, tensorflow::OpKernel* knl): GpuCore(processor), TensorCore(ctx, knl) { }
};

}

#endif
