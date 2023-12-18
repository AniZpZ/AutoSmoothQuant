<<<<<<< HEAD
#include <torch/extension.h>
#include "cublasAlgoMap.h"
#include "cublasINT8MMWrapper.h"
#include <cuda_runtime.h>
=======
// #include "common.h"
#include <torch/torch.h>
#include "cublasAlgoMap.h"
#include "cublasINT8MMWrapper.h"
// #include "cuda_utils.h"
#include <torch/extension.h>
#include <cuda_runtime.h>
// #include <pybind11/pybind11.h>
>>>>>>> 9dd6a68da6feacd809b9f22176961d58252df30b

class FTGEMM {
private:
  cublasINT8MMWrapper *int8_gemm_wrapper = nullptr;

public:
  FTGEMM() {
    // cublasAlgoMap *cublas_algo_map = new cublasAlgoMap("igemm_config.in");
    cublasAlgoMap *cublas_algo_map = new cublasAlgoMap();
<<<<<<< HEAD
=======
    // int sm = getSMVersion();
    // Allocator<AllocatorType::CUDA> allocator(getDevice());
>>>>>>> 9dd6a68da6feacd809b9f22176961d58252df30b
    std::mutex *cublas_wrapper_mutex = new std::mutex();
    bool use_ORDER_COL32_2R_4R4 = false;

    cudaStream_t stream;
    cublasLtHandle_t cublaslt_handle;
    cudaStreamCreate(&stream);
    cublasLtCreate(&cublaslt_handle);

    int8_gemm_wrapper =
        new cublasINT8MMWrapper(cublaslt_handle, stream, cublas_algo_map,
<<<<<<< HEAD
                                cublas_wrapper_mutex, use_ORDER_COL32_2R_4R4);
  };

  ~FTGEMM(){};
  torch::Tensor linear_a8_w8_o32(torch::Tensor input, torch::Tensor weight);
  torch::Tensor linear_a8_w8_o8(torch::Tensor input, torch::Tensor weight,
                                float alpha);
};

torch::Tensor FTGEMM::linear_a8_w8_o32(torch::Tensor input, // INT8
                                       torch::Tensor weight // INT8
=======
                            cublas_wrapper_mutex, use_ORDER_COL32_2R_4R4);
  };

  ~FTGEMM() {};
  torch::Tensor linear_a8_w8_o32(torch::Tensor input, torch::Tensor weight);
  torch::Tensor linear_a8_w8_o8(torch::Tensor input, torch::Tensor weight, float alpha);
};

torch::Tensor FTGEMM::linear_a8_w8_o32(torch::Tensor input,  // INT8
                         torch::Tensor weight // INT8
                         // torch::Tensor bias,    // INT32
                         // float alpha,           // FP32
                         // float beta) {          // FP32
>>>>>>> 9dd6a68da6feacd809b9f22176961d58252df30b
) {
  int m = input.size(0);
  int n = weight.size(0);
  int k = input.size(1);
  torch::Device device = input.device();

  // Allocate memory for output matrix on GPU
  torch::Tensor out = torch::empty({m, n}, torch::kInt32).to(device);
<<<<<<< HEAD
=======
  // bias = bias.to(device).view({1, -1}).repeat({M, 1});
  // auto bias_ = torch::mul(bias, beta);
  // std::cout << bias_ << std::endl;
>>>>>>> 9dd6a68da6feacd809b9f22176961d58252df30b

  // Set data types
  int8_t *input_ptr = input.data_ptr<int8_t>();
  int8_t *weight_ptr = weight.data_ptr<int8_t>();
  int32_t *output_ptr = out.data_ptr<int32_t>();
<<<<<<< HEAD
=======
  // int32_t *output_ptr = bias_ptr;
>>>>>>> 9dd6a68da6feacd809b9f22176961d58252df30b

  int8_gemm_wrapper->Gemm(output_ptr, 1, m, n, k, 0, 0, 0, input_ptr,
                          weight_ptr);
  return out;
}

torch::Tensor FTGEMM::linear_a8_w8_o8(torch::Tensor input,  // INT8
                                      torch::Tensor weight, // INT8
<<<<<<< HEAD
                                      float alpha           // FP32
=======
                                      // torch::Tensor bias,    // INT32
                                      float alpha // FP32
                                      // float beta) {          // FP32
>>>>>>> 9dd6a68da6feacd809b9f22176961d58252df30b
) {
  int m = input.size(0);
  int n = weight.size(0);
  int k = input.size(1);
  torch::Device device = input.device();

  // Allocate memory for output matrix on GPU
  torch::Tensor out = torch::empty({m, n}, torch::kInt8).to(device);
<<<<<<< HEAD
=======
  // bias = bias.to(device).view({1, -1}).repeat({M, 1});
  // auto bias_ = torch::mul(bias, beta);
  // std::cout << bias_ << std::endl;
>>>>>>> 9dd6a68da6feacd809b9f22176961d58252df30b

  // Set data types
  int8_t *input_ptr = input.data_ptr<int8_t>();
  int8_t *weight_ptr = weight.data_ptr<int8_t>();
  int8_t *output_ptr = out.data_ptr<int8_t>();
<<<<<<< HEAD
=======
  // int32_t *output_ptr = bias_ptr;
>>>>>>> 9dd6a68da6feacd809b9f22176961d58252df30b

  int8_gemm_wrapper->Gemm(output_ptr, 1, m, n, k, 0, 0, 0, alpha, input_ptr,
                          weight_ptr);
  return out;
}

<<<<<<< HEAD
PYBIND11_MODULE(ftgemm, m) {
  pybind11::class_<FTGEMM>(m, "FTGEMM")
      .def(pybind11::init<>())
      .def("linear_a8_w8_o32", &FTGEMM::linear_a8_w8_o32)
      .def("linear_a8_w8_o8", &FTGEMM::linear_a8_w8_o8);
}
=======
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<FTGEMM>(m, "FTGEMM")
    .def(pybind11::init<>())
    .def("linear_a8_w8_o32", &FTGEMM::linear_a8_w8_o32)
    .def("linear_a8_w8_o8", &FTGEMM::linear_a8_w8_o8);
}
>>>>>>> 9dd6a68da6feacd809b9f22176961d58252df30b
