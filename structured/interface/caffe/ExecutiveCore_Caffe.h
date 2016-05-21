#ifndef _STRUCTURED_EXECUTIVE_CORE_CAFFE_
#define _STRUCTURED_EXECUTIVE_CORE_CAFFE_

#include "structured/lib/ExecutiveCore.h"

namespace structured {
  
struct ExecutiveCoreCaffe: ExecutiveCore{
  friend struct ExecutiveCore;
  ExecutiveCoreCaffe(ProcessorBase * processor): ExecutiveCore(processor) { }
};

}

#endif
