#include "structured/lib/CreateProcessor.h"
#include "structured/lib/caffe/util/math_functions.hpp"
#include <iostream>
#include <complex>

using namespace structured;
using namespace std;
using namespace caffe;

template <typename Dtype>
struct Processor: ProcessorBase {

int K_ = 1024, N_ = 10, M_ = 100;
Dtype *D_ = nullptr;
  
Processor(Environment * env) : ProcessorBase(env) {
  D_ = new Dtype[K_];
  srand(K_*N_);
  for(int i=0;i<K_;i++)
    D_[i]= (rand()%2) ? (Dtype)1. : (Dtype)-1. ;
}

~Processor() {
  delete [] D_;
}

  
inline Dtype Flip(const Dtype* input, int index) const {
  return input[index] * D_[index];
}

void ComputeOnCPU(
	  const Dtype* bottom_data,
	  Dtype* top_data,
	  const Dtype* weight,
	  complex<Dtype>* conv_buffer,
	  complex<Dtype>* param_buffer,
	  Dtype* data_buffer) {

  int Kc = K_ / 2 + 1;

  cerr<<"FFT, FWD, M="<<M_<<", K="<<K_<<", N="<<N_;

  cerr<<"/Forward/Flip";
  for(int i=0; i<M_; i++)
    for(int j=0; j<K_; j++)
      (data_buffer + i*K_)[j] = Flip(bottom_data + i*K_, j);

  cerr<<"/FFT";
  caffe_cpu_fft<Dtype>(1, K_, weight, param_buffer);
  caffe_cpu_fft<Dtype>(M_, K_, data_buffer, conv_buffer);
  cerr<<"/MUL";
  for(int i=0; i<M_; i++)
  {
    caffe_mul<complex<Dtype> >(Kc, param_buffer, conv_buffer + i*Kc, conv_buffer + i*Kc);
  }
  cerr<<"/IFFT"<<endl;

  caffe_cpu_ifft<Dtype>(M_, K_, conv_buffer, data_buffer);
  for(int i=0; i<M_; i++)
    for (int j=0; j<N_; j++)
      (top_data + i*N_)[j] = (data_buffer + i*K_)[j] / K_;
 
}

void GradOfInputOnCPU(
	  const Dtype* top_diff,
	  const Dtype* param_buffer,
	  Dtype* grad,
	  Dtype* weight_buffer) {

     for(int i=0; i<N_; i++)
      for(int j=0; j<K_; j++)
	(weight_buffer + i*K_)[j] = param_buffer[(K_+i-j)%K_] * D_[j];
     
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, weight_buffer, (Dtype)0., grad);

}

void GradOfParamOnCPU(
	  const Dtype* top_diff,
	  const Dtype* bottom_data,
	  Dtype* grad,
	  Dtype* weight_buffer) {

     caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)0., weight_buffer);

     caffe_set(K_, (Dtype)0., grad);

     for(int i=0; i<N_; i++)
      for(int j=0; j<K_; j++)
	grad[(K_+i-j)%K_] += (weight_buffer + i*K_)[j] * D_[j];
          
}

void GradOfParamOnCPUFFT(
	  const Dtype* top_diff,
	  const Dtype* bottom_data,
	  Dtype* grad,
	  complex<Dtype>* conv_buffer,
	  complex<Dtype>* diff_buffer,
	  Dtype* data_buffer,
	  Dtype* bias_multiplier) {
  
    int Kc = K_ / 2 + 1;
 
    // Gradient with respect to weight
  
    for(int i=0; i<M_; i++)
    {
      caffe_copy<Dtype>(N_, top_diff + i*N_, data_buffer + i*K_);
      for(int j=N_; j<K_; j++) (data_buffer + i*K_)[j] = (Dtype)0;
    }
    caffe_cpu_fft<Dtype>(M_, K_, data_buffer, diff_buffer);
    for(int i=0; i<M_; i++)
      for(int j=0; j<K_; j++)
        (data_buffer + i*K_)[(K_-j)%K_] = this->Flip(bottom_data + i*K_, j);
    caffe_cpu_fft<Dtype>(M_, K_, data_buffer, conv_buffer);
    caffe_mul<complex<Dtype> >(M_ * Kc, conv_buffer, diff_buffer, conv_buffer);
    caffe_cpu_ifft<Dtype>(M_, K_, conv_buffer, data_buffer);
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, K_, (Dtype)1./K_, data_buffer,
			  bias_multiplier, (Dtype)0.,
			  grad);
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
  
}

void Forward(
  ExecutiveCore * core,
  const ProcessorTape * atape) {

  const auto & input = atape->input[0]->typed<Dtype>();
  const auto & param = atape->input[1]->typed<Dtype>();
  auto & output = atape->output[0]->typed<Dtype>();
  
  vector<int64> buffer_shape(1, M_*K_);
  
  auto conv_buffer = core->allocateBuffer<complex<Dtype> >(buffer_shape);
  auto param_buffer = core->allocateBuffer<complex<Dtype> >(buffer_shape);
  auto data_buffer = core->allocateBuffer<Dtype>(buffer_shape);

  cerr<<"buffer allocated, insight input: "<<input.data()[0]
      <<","<<input.data()[1]<<",...,"<<input.data()[K_]<<endl;
  cerr<<"buffer allocated, insight param: "<<param.data()[0]
      <<","<<param.data()[1]<<",...,"<<param.data()[K_]<<">>";
  
  ComputeOnCPU(input.data(), output.data(), param.data(),
	       conv_buffer->data(), param_buffer->data(), data_buffer->data());
}

void Backward(
  ExecutiveCore * core,
  const ProcessorTape * atape,
  const ProcessorTape * btape) {
  
  const auto & input = atape->input[0]->typed<Dtype>(); 
  const auto & param = atape->input[1]->typed<Dtype>();
  const auto & top_diff = btape->input[0]->typed<Dtype>();
  auto & grad_of_input = btape->output[0]->typed<Dtype>();
  auto & grad_of_param = btape->output[1]->typed<Dtype>();

  vector<int64> input_buffer_shape(1, N_*K_);
  auto weight_buffer = core->allocateBuffer<Dtype>(input_buffer_shape);


  cerr<<"buffer allocated, insight input: "<<input.data()[0]
      <<","<<input.data()[1]<<",...,"<<input.data()[K_]<<endl;
  cerr<<"grad buf llocated, insight diff: "<<top_diff.data()[0]
      <<","<<top_diff.data()[1]<<",...,"<<top_diff.data()[N_]<<endl;
  
  GradOfInputOnCPU(top_diff.data(), param.data(), grad_of_input.data(), 
		   weight_buffer->data());

  GradOfParamOnCPU(top_diff.data(), input.data(), grad_of_param.data(),
  		   weight_buffer->data());
  
  cerr<<"matrix version veri, insight grad: "<<grad_of_param.data()[0]
      <<","<<grad_of_param.data()[1]<<",...,"<<grad_of_param.data()[N_]<<endl;

  vector<int64> buffer_shape(1, M_*K_);
  
  auto conv_buffer = core->allocateBuffer<complex<Dtype> >(buffer_shape);
  auto diff_buffer = core->allocateBuffer<complex<Dtype> >(buffer_shape);
  auto data_buffer = core->allocateBuffer<Dtype>(buffer_shape);

  vector<int64> bias_shape(1, M_);
  auto bias_multiplier = core->allocateBuffer<Dtype>(buffer_shape);
  caffe_set(M_, Dtype(1.), bias_multiplier->data());

  // cerr<<"Grad Param num="<<grad_of_param.count()<<endl;
  GradOfParamOnCPUFFT(top_diff.data(), input.data(), weight_buffer->data(),
  		   conv_buffer->data(), diff_buffer->data(),
  		   data_buffer->data(),
  		   bias_multiplier->data());
  cerr<<"grad buf llocated, insight grad: "<<weight_buffer->data()[0]
      <<","<<weight_buffer->data()[1]<<",...,"<<weight_buffer->data()[N_]<<endl;

  // for(int i=0; i<K_; i++)
  //   if(abs(weight_buffer->data()[i]-grad_of_param.data()[i])>1e-5)
  //     cerr<<"Data disagree!!!!!!!!!!"<<endl;
  
  //  caffe_set(grad_of_input.count(), (Dtype)0., grad_of_input.data());
  //  caffe_set(grad_of_param.count(), (Dtype)0., grad_of_param.data());
}

};

INSTALL_PROCESSOR(float, Processor<float>)
INSTALL_PROCESSOR(double, Processor<double>)
