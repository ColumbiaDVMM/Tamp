#ifndef _STRUCTURED_PROCESSOR_TAPE_
#define _STRUCTURED_PROCESSOR_TAPE_

#include <memory>

namespace structured {
  
struct ProcessorTape{
  std::vector<std::shared_ptr<const BufferedData > > input;
  std::vector<std::shared_ptr<BufferedData > > output;
};
  
}

#endif
