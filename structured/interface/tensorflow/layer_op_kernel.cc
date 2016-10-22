#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "structured/interface/tensorflow/layer_op.h"
#include "structured/interface/tensorflow/TypedData_Tensor.h"
#include "structured/interface/tensorflow/ExecutiveCore_Tensor.h"
#include "structured/lib/AdjointPoint.h"

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("LayerOp")
    .Attr("T: {float, double}")
    .Input("input: T")
    .Input("param: T")
    .Output("output: T");

REGISTER_OP("LayerOpGradient")
    .Attr("T: {float, double}")
    .Attr("op_name: string")
    .Input("input: T")
    .Input("param: T")
    .Input("output: T")
    .Input("top: T")
    .Output("grad_input: T")
    .Output("grad_param: T");

namespace structured{

template <typename Device, typename T>
struct LaunchLayerOp {
    static void launch(
      OpKernelContext* ctx, OpKernel* kernel, ProcessorBase* processor) {

      CpuCoreTensor core(processor, ctx, kernel);
      ProcessorTape atape;
      
      for (int i=0; i<ctx->num_inputs(); i++)
	atape.input.push_back(
	     shared_ptr<const BufferedData >(new TypedDataTensor<T>( ctx->input(i) ))
			      );

      for (int i=0; i<ctx->num_outputs(); i++)
	atape.output.push_back(
	     shared_ptr<BufferedData>(new TypedDataTensorShape<T>)
			       );
      
      functor::LayerOpFunctor<Device>()(ctx->eigen_device<Device>(),
      					   &core, &atape);
    }
    static void launchgrad(
      OpKernelContext* ctx, OpKernel* kernel, ProcessorBase* processor) {

      CpuCoreTensor core(processor, ctx, kernel);
      ProcessorTape atape;
      ProcessorTape btape;
      
      for (int i=0; i<processor->num_inputs(); i++)
	atape.input.push_back(
	   shared_ptr<const BufferedData >(new TypedDataTensor<T>( ctx->input(i) ))
			      );

      for (int i=0; i<processor->num_outputs(); i++)
	atape.output.push_back(
	   shared_ptr< BufferedData >(new TypedDataTensor<T>(
		      ctx->input(i+processor->num_inputs()) ))
			      );

      for (int i=0; i<processor->num_outputs(); i++)
	btape.input.push_back(
	   shared_ptr<const BufferedData >(new TypedDataTensor<T>(
      	      ctx->input(i+processor->num_inputs()+processor->num_outputs())))
			      );

      Tensor* output;
      for(int i=0; i<processor->num_inputs(); i++) {
        OP_REQUIRES_OK(ctx,
		       ctx->allocate_output(i, ctx->input(i).shape(), &output));
	auto output_ptr = new TypedDataTensor<T>(*output);
        btape.output.push_back(
	    shared_ptr<BufferedData>( output_ptr )
	   );
      }
      
      functor::LayerOpFunctor<Device>()(ctx->eigen_device<Device>(),
					&core, &atape, &btape);
      
    }
};

template <typename Device, typename T>
class LayerOp : public OpKernel {
protected:
  ProcessorBase * processor;
public:
  explicit LayerOp(OpKernelConstruction* context) : OpKernel(context) {
    ProcessorRepresentative<T> representative;
 
    LOG(INFO)<<"OpKernelInterfaceInitializaing: "<<this->name()
	     <<", type: "<<this->type_string();

    auto got = ProcessorMap.find(this->name());

    if ( got == ProcessorMap.end() ) {
      Environment env;

      env.num_inputs = this->num_inputs();
      env.num_outputs = this->num_outputs();
    
      this->processor = representative(&env);
      ProcessorMap.insert({this->name(), this->processor});
    
      LOG(INFO)<<"OpKernelProcessorCreated: "<<this->processor;
    } else {
      this->processor = got->second;
    
      LOG(INFO)<<"OpKernelProcessorGot: "<<this->processor;   
    }
  }

  void Compute(OpKernelContext* context) override {
    LaunchLayerOp<Device, T>::launch(context, this, this->processor);
  }
};

template <typename Device, typename T>
class LayerOpGradient : public OpKernel {
protected:
  ProcessorBase * processor;
public:
  explicit LayerOpGradient(OpKernelConstruction* context) : OpKernel(context) {
    string ProcessorName;
    Environment env;
    
    LOG(INFO)<<"OpKernelGradientInterfaceInitializaing: "<<this->name();

    OP_REQUIRES_OK(context,
                   context->GetAttr("op_name", &ProcessorName));
    
    this->processor = ProcessorMap.at(ProcessorName);

    LOG(INFO)<<"OpKernelGradientProcessorGot: "<<this->processor;
  }

  void Compute(OpKernelContext* context) override {
    LaunchLayerOp<Device, T>::launchgrad(context, this, this->processor);
  }
};

}

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("LayerOp").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      structured::LayerOp<CPUDevice, type>)			    \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("LayerOpGradient").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      structured::LayerOpGradient<CPUDevice, type>)

REGISTER_KERNEL(float)
REGISTER_KERNEL(double)  
