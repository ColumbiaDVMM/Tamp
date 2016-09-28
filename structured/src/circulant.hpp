#ifndef _CIRCULANT_PROJECTION_
#define _CIRCULANT_PROJECTION_
#include <complex>

template <typename TCore, typename Dtype>
struct CirculantProjection{
  static void Compute(const Dtype* D_,
		      const int M_,
		      const int K_,
		      const int N_,
		      const Dtype* bottom_data,
		      Dtype* top_data,
		      const Dtype* weight,
		      std::complex<Dtype>* conv_buffer,
		      std::complex<Dtype>* param_buffer,
		      Dtype* data_buffer);
  static void GradientOfInput(
     const Dtype* D_, const int M_, const int K_,  const int N_,
     const Dtype* top_diff,
     const Dtype* param_buffer,
     Dtype* grad,
     Dtype* weight_buffer);
  static void GradientOfInput(
   const Dtype* D_, const int M_, const int K_, const int N_,
   const Dtype* top_diff,
   const Dtype* param_data,
   Dtype* grad,
   std::complex<Dtype>* conv_buffer,
   std::complex<Dtype>* diff_buffer,
   std::complex<Dtype>* param_buffer,
   Dtype* data_buffer,
   Dtype* weight_buffer);
  static void GradientOfParameter(const Dtype* D_,
				  const int M_,
				  const int K_,
				  const int N_,
				  const Dtype* top_diff,
				  const Dtype* bottom_data,
				  Dtype* grad,
				  std::complex<Dtype>* conv_buffer,
				  std::complex<Dtype>* diff_buffer,
				  Dtype* data_buffer,
				  Dtype* bias_multiplier);
};

#endif
