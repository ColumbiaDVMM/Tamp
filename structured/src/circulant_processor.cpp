#include "structured/lib/CreateProcessor.h"
#include "structured/lib/caffe/util/math_functions.hpp"
#include "circulant.hpp"
#include <iostream>
#include <complex>
#include <typeinfo>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace structured;
using namespace std;
using namespace caffe;

template <typename Dtype>
struct Processor: ProcessorBase {

int K_ = 1024, N_ = 10, M_ = 100;
Dtype *D_ = nullptr;

Processor(Environment * env) : ProcessorBase(env) {
  cudaMallocManaged(&D_, sizeof(Dtype) * K_);
  // D_ = static_cast<Dtype*>( malloc(sizeof(Dtype) * K_) );
  srand(K_*N_);
  for(int i=0;i<K_;i++)
    D_[i]= (rand()%2) ? (Dtype)1. : (Dtype)-1. ;
  cerr<<"Initialized"<<endl;
}

~Processor() {
  cudaFree(D_);
}

void Shape(
   ExecutiveCore * core,
   ProcessorTape * atape) {

  const auto & input = atape->input[0]->typed<Dtype>();
  const auto & param = atape->input[1]->typed<Dtype>();
  auto & output = atape->output[0]->typed<Dtype>();

  M_ = input.dim_size(0);
  K_ = input.dim_size(1);
  N_ = 384;
  output.reshape({M_, N_});

  cerr<<"Shaped"<<endl;
}

void Forward(
  ExecutiveCore * core,
  const ProcessorTape * atape) {

  const auto input = &atape->input[0]->typed<Dtype>();
  const auto param = &atape->input[1]->typed<Dtype>();
  auto output = &atape->output[0]->typed<Dtype>();
  
  vector<int64> buffer_shape(1, M_*K_);
  
  auto conv_buffer = core->allocateBuffer<complex<Dtype> >(buffer_shape);
  auto param_buffer = core->allocateBuffer<complex<Dtype> >(buffer_shape);
  auto data_buffer = core->allocateBuffer<Dtype>(buffer_shape);

  core->only( [=](GpuCore* core){
      cerr<<"Gpu detected!"<<endl;
      
      CirculantProjection<GpuCore, Dtype>::Compute(
	       D_, M_, K_, N_,
	       input->data(), output->data(), param->data(),
	       conv_buffer->data(), param_buffer->data(), data_buffer->data());

      return true;
    }) ||
  core->only( [=](CpuCore* core){
  
      cerr<<"buffer allocated, insight input: "<<input->data()[0]
	  <<","<<input->data()[1]<<",...,"<<input->data()[K_]<<endl;
      cerr<<"buffer allocated, insight param: "<<param->data()[0]
	  <<","<<param->data()[1]<<",...,"<<param->data()[K_]<<">>";

      CirculantProjection<CpuCore, Dtype>::Compute(
	       D_, M_, K_, N_,
	       input->data(), output->data(), param->data(),
	       conv_buffer->data(), param_buffer->data(), data_buffer->data());
      return true;
    });
}

void Backward(
  ExecutiveCore * core,
  const ProcessorTape * atape,
  const ProcessorTape * btape) {

  cerr<<"In Fwd"<<endl;
  
  const auto input = &atape->input[0]->typed<Dtype>(); 
  const auto param = &atape->input[1]->typed<Dtype>();
  const auto top_diff = &btape->input[0]->typed<Dtype>();
  auto grad_of_input = &btape->output[0]->typed<Dtype>();
  auto grad_of_param = &btape->output[1]->typed<Dtype>();

  cerr<<"Allocating Buffer..."<<endl;
  
  vector<int64> input_buffer_shape(1, N_*K_);
  auto weight_buffer = core->allocateBuffer<Dtype>(input_buffer_shape);
  vector<int64> buffer_shape(1, M_*K_);
  
  auto conv_buffer = core->allocateBuffer<complex<Dtype> >(buffer_shape);
  auto diff_buffer = core->allocateBuffer<complex<Dtype> >(buffer_shape);
  auto param_buffer = core->allocateBuffer<complex<Dtype> >(buffer_shape);
  auto data_buffer = core->allocateBuffer<Dtype>(buffer_shape);

  vector<int64> bias_shape(1, M_);
  auto bias_multiplier = core->allocateBuffer<Dtype>(buffer_shape);
  caffe_set(M_, Dtype(1.), bias_multiplier->data());

  cerr<<"PreCall!"<<endl;
  
  core->only( [=](GpuCore* core){
      cerr<<"Gpu detected!"<<endl;
     
      CUDA_CHECK(cudaMemcpy2D(data_buffer->data(), K_ * sizeof(Dtype),
			      top_diff->data(), N_ * sizeof(Dtype),
			      N_ * sizeof(Dtype), M_, cudaMemcpyDeviceToDevice));
      if (N_ < K_)
	CUDA_CHECK(cudaMemset2D(data_buffer->data() + N_, K_* sizeof(Dtype),
				0, (K_ - N_) * sizeof(Dtype), M_));
    
      caffe_gpu_fft<Dtype>(M_, K_, data_buffer->data(), diff_buffer->data());
      CirculantProjection<GpuCore, Dtype>::GradientOfInput(
	       D_, M_, K_, N_,
	       top_diff->data(), param->data(), grad_of_input->data(),
	       conv_buffer->data(), diff_buffer->data(),
	       param_buffer->data(), data_buffer->data(),
	       weight_buffer->data());;

      CirculantProjection<GpuCore, Dtype>::GradientOfParameter(
	       D_, M_, K_, N_,      
	       top_diff->data(), input->data(), grad_of_param->data(),
	       conv_buffer->data(), diff_buffer->data(),
	       data_buffer->data(),
	       bias_multiplier->data());
  
      return true;
    }) ||
  core->only( [=](CpuCore* core){
      cerr<<"buffer allocated, insight input: "<<input->data()[0]
	  <<","<<input->data()[1]<<",...,"<<input->data()[K_]<<endl;
      cerr<<"grad buf llocated, insight diff: "<<top_diff->data()[0]
	  <<","<<top_diff->data()[1]<<",...,"<<top_diff->data()[N_]<<endl;
      
      CirculantProjection<CpuCore, Dtype>::GradientOfInput(
	       D_, M_, K_, N_,
	       top_diff->data(), param->data(), grad_of_input->data(),
	       weight_buffer->data());;

  /*
  GradOfParamOnCPU(top_diff.data(), input.data(), grad_of_param.data(),
  		   weight_buffer->data());
  
  cerr<<"matrix version veri, insight grad: "<<grad_of_param.data()[0]
      <<","<<grad_of_param.data()[1]<<",...,"<<grad_of_param.data()[N_]<<endl;
  */
  

  // cerr<<"Grad Param num="<<grad_of_param.count()<<endl;
      CirculantProjection<CpuCore, Dtype>::GradientOfParameter(
	       D_, M_, K_, N_,      
	       top_diff->data(), input->data(), grad_of_param->data(),
	       conv_buffer->data(), diff_buffer->data(),
	       data_buffer->data(),
	       bias_multiplier->data());
  /*
  cerr<<"grad buf llocated, insight grad: "<<weight_buffer->data()[0]
      <<","<<weight_buffer->data()[1]<<",...,"<<weight_buffer->data()[N_]<<endl;

  // for(int i=0; i<K_; i++)
  //   if(abs(weight_buffer->data()[i]-grad_of_param.data()[i])>1e-5)
  //     cerr<<"Data disagree!!!!!!!!!!"<<endl;
  
  //  caffe_set(grad_of_input.count(), (Dtype)0., grad_of_input.data());
  //  caffe_set(grad_of_param.count(), (Dtype)0., grad_of_param.data());
  */
      return true;
    });
}

};

INSTALL_PROCESSOR(float, Processor<float>)
INSTALL_PROCESSOR(double, Processor<double>)
