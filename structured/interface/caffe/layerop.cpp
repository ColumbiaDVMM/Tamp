//This is an empty file.

#include "structured/interface/caffe/layerop.hpp"
#include <iostream>

using namespace std;

namespace structured {


namespace functor {
// Partial specialization MatMulFunctor<Device=CPUDevice, T>.
template <>
void LayerOpFunctor<CPUDevice>::operator()(
		  const CPUDevice& d,
		  ExecutiveCoreCaffe* core,
		  ProcessorTape* atape
		  ) {
  cerr<<"Functor called!\n";
  //core->load(atape);

 
  core->execute(atape);
}

template <>
void LayerOpFunctor<CPUDevice>::operator()(
		  const CPUDevice& d,
		  ExecutiveCoreCaffe* core,
		  ProcessorTape* atape,
		  ProcessorTape* btape
		  ) {

  core->execute(atape, btape);
}

}

  
}
