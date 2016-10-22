#include "circulant.hpp"
#include "structured/lib/ExecutiveCore.h"
#include "structured/lib/caffe/util/math_functions.hpp"
#include <vector>
#include <thrust/complex.h>

using namespace structured;
using namespace std;
using namespace caffe;

template <typename Dtype>
__global__ void bat_vmul_knl(const int m, const int n, Dtype* t, const Dtype* r, const Dtype* d) {
  CUDA_KERNEL_LOOP_2D(batch, index, m, n) {
    int off = batch * n;
    (t + off)[index] = (r + off)[index] * d[index];
  }
}

template <typename Dtype>
__global__ void bat_amul_knl(const int m, const int n, Dtype* t, const Dtype* r, const Dtype alpha, const int tn, const int rn) {
  CUDA_KERNEL_LOOP_2D(batch, index, m, n) {
    (t + batch*tn)[index] = alpha * (r + batch*rn)[index];
  }
}

template <typename Dtype>
__global__ void bat_vamul_knl(const int m, const int n, Dtype* t, const Dtype* r, const Dtype* d, const Dtype alpha, const int tn, const int rn) {
  CUDA_KERNEL_LOOP_2D(batch, index, m, n) {
    (t + batch*tn)[index] = alpha * (r + batch*rn)[index] * d[index];
  }
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
  
  bat_vmul_knl<Dtype><<<CAFFE_GET_BLOCKS_2D(M_, K_), CAFFE_CUDA_NUM_THREADS_2D>>>
    (M_, K_, data_buffer, bottom_data, D_);
  CUDA_POST_KERNEL_CHECK;
  
  cerr<<"Forward/GPU_FFT"<<endl;
  caffe_gpu_fft<Dtype>(1, K_, weight, param_buffer);
  caffe_gpu_fft<Dtype>(M_, K_, data_buffer, conv_buffer);
  cerr<<"Forward/MUL"<<endl;

  bat_vmul_knl<thrust::complex<Dtype> >
    <<<CAFFE_GET_BLOCKS_2D(M_, Kc), CAFFE_CUDA_NUM_THREADS_2D>>>
    (M_, Kc,
     reinterpret_cast<thrust::complex<Dtype> *>(conv_buffer),
     reinterpret_cast<thrust::complex<Dtype> *>(conv_buffer),
     reinterpret_cast<thrust::complex<Dtype> *>(param_buffer)
     );
  CUDA_POST_KERNEL_CHECK;
  
  cerr<<"FORWARD/IFFT"<<endl;
  caffe_gpu_ifft<Dtype>(M_, K_, conv_buffer, data_buffer);

  bat_amul_knl<Dtype><<<CAFFE_GET_BLOCKS_2D(M_, K_), CAFFE_CUDA_NUM_THREADS_2D>>>
    (M_, N_, top_data, data_buffer, (Dtype)1./K_, N_, K_);
  CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
__global__ void bat_cirvmul_knl(const int m, const int n, Dtype* t, const Dtype* r, const Dtype* d) {
  CUDA_KERNEL_LOOP_2D(batch, index, m, n) {
    int off = batch * n;
    (t + off)[(n-index)%n]=(r + off)[index] * d[index];
  }
}

template <typename Dtype>
__global__ void circpy_knl(const int n, Dtype* dist, const Dtype* src) {
  CUDA_KERNEL_LOOP(i, n) {
    dist[(n-i)%n] = src[i];
  }
}

template <typename TCore, typename Dtype>
void CirculantProjection<TCore, Dtype>::GradientOfInput(
   const Dtype* D_, const int M_, const int K_, const int N_,
   const Dtype* top_diff,
   const Dtype* param_data,
   Dtype* grad,
   complex<Dtype>* conv_buffer,
   complex<Dtype>* diff_buffer,
   complex<Dtype>* param_buffer,
   Dtype* data_buffer,
   Dtype* weight_buffer) {

  const int Kc = K_ / 2 + 1;

  circpy_knl<Dtype><<<CAFFE_GET_BLOCKS(K_), CAFFE_CUDA_NUM_THREADS>>>
    (K_,  weight_buffer, param_data);
  CUDA_POST_KERNEL_CHECK;
  caffe_gpu_fft<Dtype>(1, K_, weight_buffer, param_buffer);
  bat_vmul_knl<thrust::complex<Dtype> >
    <<<CAFFE_GET_BLOCKS_2D(M_, Kc), CAFFE_CUDA_NUM_THREADS_2D>>>
    (M_, Kc,
     reinterpret_cast<thrust::complex<Dtype> *>(conv_buffer),
     reinterpret_cast<thrust::complex<Dtype> *>(diff_buffer),
     reinterpret_cast<thrust::complex<Dtype> *>(param_buffer)
     );
  CUDA_POST_KERNEL_CHECK;
  caffe_gpu_ifft<Dtype>(M_, K_, conv_buffer, data_buffer);
  bat_vamul_knl<Dtype><<<CAFFE_GET_BLOCKS_2D(M_, K_), CAFFE_CUDA_NUM_THREADS_2D>>>
    (M_, K_, grad, data_buffer, D_, (Dtype)1./K_, K_, K_);
  CUDA_POST_KERNEL_CHECK;

    // Gradient with respect to bottom data

}

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

  const int Kc = K_ / 2 + 1;
  /*
  if (this->param_propagate_down_[0] || propagate_down[0] ){
  
  if (this->param_propagate_down_[0]) {
  */

  cerr<<"Backward/FFT"<<endl;

  bat_cirvmul_knl<Dtype>
      <<<CAFFE_GET_BLOCKS_2D(M_, K_), CAFFE_CUDA_NUM_THREADS_2D>>>
    (M_, K_, data_buffer, bottom_data, D_);
  CUDA_POST_KERNEL_CHECK;
  caffe_gpu_fft<Dtype>(M_, K_, data_buffer, conv_buffer);
  caffe_gpu_mul<complex<Dtype> >(M_ * Kc, conv_buffer, diff_buffer, conv_buffer);
  caffe_gpu_ifft<Dtype>(M_, K_, conv_buffer, data_buffer);
  caffe_gpu_gemv<Dtype>(CblasTrans, M_, K_, (Dtype)1./K_, data_buffer,
			bias_multiplier, (Dtype)0.,
			grad);

}

template class CirculantProjection<GpuCore, float>;
template class CirculantProjection<GpuCore, double>;
