#ifndef _STRUCTURED_EXECUTIVE_CORE_
#define _STRUCTURED_EXECUTIVE_CORE_

#include "structured/lib/ProcessorBase.h"
#include "structured/lib/ProcessorTape.h"

namespace structured {

struct ExecutiveCore{
  ExecutiveCore(ProcessorBase * processor): _processor(processor) {}
  virtual ~ExecutiveCore() = default;
  virtual int load(ProcessorTape *atape) {
    _processor->Shape(this, atape);
  }
  virtual int execute(ProcessorTape *atape) {
    _processor->Forward(this, atape);
    return 0;
  }
  virtual int execute(const ProcessorTape *atape, ProcessorTape *btape) {
    _processor->Backward(this, atape, btape);
    return 0;
  }
  template <typename T> std::shared_ptr<TypedData<T> >
  allocateBuffer(const std::vector<int64>& shape);
  template <typename T> std::shared_ptr<TypedData<T> >
  allocateShape(const std::vector<int64>& shape);
protected:
  ProcessorBase * _processor;
};

struct CpuCore: ExecutiveCore{
  CpuCore(ProcessorBase * processor): ExecutiveCore(processor) {}
};

struct GpuCore: ExecutiveCore{
  GpuCore(ProcessorBase * processor): ExecutiveCore(processor) {}
};

}

#endif
