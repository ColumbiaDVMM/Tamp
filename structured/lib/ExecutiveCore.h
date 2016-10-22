#ifndef _STRUCTURED_EXECUTIVE_CORE_
#define _STRUCTURED_EXECUTIVE_CORE_

#include "structured/lib/ProcessorBase.h"
#include "structured/lib/ProcessorTape.h"
#include <functional>

namespace structured {

struct GpuCore;
struct CpuCore;
  
struct ExecutiveCore{
  using CpuProcedure = std::function<bool(CpuCore*)>;
  using GpuProcedure = std::function<bool(GpuCore*)>;
  ExecutiveCore(ProcessorBase * processor): _processor(processor) {}
  virtual ~ExecutiveCore() = default;
  virtual int load(ProcessorTape *atape) {
    _processor->Shape(this, atape);
    return 0;
  }
  virtual int execute(ProcessorTape *atape) {
    _processor->Forward(this, atape);
    return 0;
  }
  virtual int execute(const ProcessorTape *atape, ProcessorTape *btape) {
    _processor->Backward(this, atape, btape);
    return 0;
  }
  virtual bool only(const CpuProcedure & func) = 0;
  virtual bool only(const GpuProcedure & func) = 0;
  template <typename T> std::shared_ptr<TypedData<T> >
  allocateBuffer(const std::vector<int64>& shape);
  template <typename T> std::shared_ptr<TypedData<T> >
  allocateShape(const std::vector<int64>& shape);
protected:
  ProcessorBase * _processor;
};

struct CpuCore: ExecutiveCore{
  CpuCore(ProcessorBase * processor): ExecutiveCore(processor) {}
  virtual bool only(const CpuProcedure & func) { return func(this); }
  virtual bool only(const GpuProcedure & func) { return false; }
};

struct GpuCore: ExecutiveCore{
  GpuCore(ProcessorBase * processor): ExecutiveCore(processor) {}
  virtual bool only(const CpuProcedure & func) { return false; }
  virtual bool only(const GpuProcedure & func) { return func(this); }
};

}

#endif
