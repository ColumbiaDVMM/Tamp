#ifndef _STRUCTURED_PROCESSOR_BASE_
#define _STRUCTURED_PROCESSOR_BASE_

#include "structured/lib/TypedData.h"
#include "structured/lib/ProcessorTape.h"
#include "structured/lib/Environment.h"
#include <memory>

namespace structured {

struct ExecutiveCore;

struct ProcessorBase{
  explicit ProcessorBase(Environment * env):
    _num_inputs(env->num_inputs), _num_outputs(env->num_outputs) {}
  virtual void Shape(
     ExecutiveCore * core,
     ProcessorTape * atape) = 0;
  virtual void Forward(
     ExecutiveCore * core,
     const ProcessorTape * atape) = 0;
  virtual void Backward(
     ExecutiveCore * core,
     const ProcessorTape * atape,
     const ProcessorTape * btape) = 0;
  virtual ~ProcessorBase() = default;
  inline int num_inputs() { return _num_inputs; }
  inline int num_outputs() { return _num_outputs; }
protected:
  const int _num_inputs, _num_outputs;
};

}

#endif
