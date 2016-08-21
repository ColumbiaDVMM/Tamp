#ifndef _STRUCTURED_EXECUTIVE_CORE_CAFFE_
#define _STRUCTURED_EXECUTIVE_CORE_CAFFE_

#include "structured/lib/ExecutiveCore.h"

namespace structured {
  
struct CpuCoreCaffe: CpuCore{
  CpuCoreCaffe(ProcessorBase * processor): CpuCore(processor) { }
};

struct GpuCoreCaffe: GpuCore{
  GpuCoreCaffe(ProcessorBase * processor): GpuCore(processor) { }
};

}

#endif
