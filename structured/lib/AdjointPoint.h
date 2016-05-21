#ifndef _STRUCTURED_ADJOINT_POINT_
#define _STRUCTURED_ADJOINT_POINT_

#include "structured/lib/Environment.h"
#include "structured/lib/ProcessorBase.h"

namespace structured{

template<typename T>
struct ProcessorRepresentative {
  ProcessorBase* operator() (Environment * env);
};

}
#endif
