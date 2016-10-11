#include "structured/interface/tensorflow/layer_op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace structured {

std::unordered_map<std::string, ProcessorBase*> ProcessorMap;

namespace functor {
// Partial specialization MatMulFunctor<Device=CPUDevice, T>.
template <>
void LayerOpFunctor<CPUDevice>::operator()(
		  const CPUDevice& d,
		  CpuCoreTensor* core,
		  ProcessorTape* atape
		  ) {

  core->load(atape);

  Tensor* output_tensor = NULL;

  for(int i=0; i<atape->output.size(); i++){
    auto shape = atape->output[i];
    OP_REQUIRES_OK(core->getContext(),
      core->getContext()->allocate_output(i,
	dynamic_cast<TensorShape&>(*shape), &output_tensor) );
    atape->output[i] = shape->fromBuffer(output_tensor);
  }

  core->execute(atape);
}

template <>
void LayerOpFunctor<CPUDevice>::operator()(
		  const CPUDevice& d,
		  CpuCoreTensor* core,
		  ProcessorTape* atape,
		  ProcessorTape* btape
		  ) {

  core->execute(atape, btape);
}

}
}
