#ifndef _STRUCTURED_CREATE_PROCESSOR_
#define _STRUCTURED_CREATE_PROCESSOR_

#include "structured/lib/ProcessorBase.h"
#include "structured/lib/ExecutiveCore.h"
#include "structured/lib/AdjointPoint.h"

#define INSTALL_PROCESSOR(type, processor)			\
  namespace structured {					\
    template <> ProcessorBase*					\
    ProcessorRepresentative<type>::operator()			\
      (Environment * env){					\
      return new processor(env);				\
    }								\
  }

#endif
