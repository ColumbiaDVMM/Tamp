#include "circulant.hpp"
#include "structured/lib/ExecutiveCore.h"
#include "structured/lib/caffe/util/math_functions.hpp"
#include <iostream>

using namespace structured;
using namespace std;
using namespace caffe;

template <typename Dtype>
inline Dtype Flip(const Dtype* D_,
	   const Dtype* input,
	   const int index) {
    return input[index] * D_[index];
}

template <typename TCore, typename Dtype>
void CirculantProjection<TCore, Dtype>::Compute(
   const Dtype* D_, const int M_, const int K_, const int N_,
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
      (data_buffer + i*K_)[j] = Flip(D_, bottom_data + i*K_, j);

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

template <typename TCore, typename Dtype>
void CirculantProjection<TCore, Dtype>::GradientOfInput(
   const Dtype* D_, const int M_, const int K_, const int N_,
   const Dtype* top_diff,
   const Dtype* param_data,
   Dtype* grad,
   Dtype* weight_buffer) {

  for(int i=0; i<N_; i++)
    for(int j=0; j<K_; j++)
      (weight_buffer + i*K_)[j] = param_data[(K_+i-j)%K_] * D_[j];
     
  // Gradient with respect to bottom data
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
			top_diff, weight_buffer, (Dtype)0., grad);

}

  /*  
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
  */

template <typename TCore, typename Dtype>
void CirculantProjection<TCore, Dtype>::GradientOfParameter(
   const Dtype* D_, const int M_, const int K_, const int N_,
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
        (data_buffer + i*K_)[(K_-j)%K_] = Flip(D_, bottom_data + i*K_, j);
    caffe_cpu_fft<Dtype>(M_, K_, data_buffer, conv_buffer);
    caffe_mul<complex<Dtype> >(M_ * Kc, conv_buffer, diff_buffer, conv_buffer);
    caffe_cpu_ifft<Dtype>(M_, K_, conv_buffer, data_buffer);
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, K_, (Dtype)1./K_, data_buffer,
			  bias_multiplier, (Dtype)0.,
			  grad);
}

template class CirculantProjection<CpuCore, float>;
template class CirculantProjection<CpuCore, double>;
