# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# find_package(pybind11 REQUIRED)
# find_package(CUDA REQUIRED)
# find_package(Python REQUIRED)
# find_package(Torch REQUIRED)

set(CUDA_INCLUDE_DIR "/usr/local/cuda/include" CACHE PATH "CUTLASS Header Library")
# set(PYBIND_INCLUDE_DIR "/usr/local/lib/python3.9/site-packages/pybind11/include" CACHE PATH "PYBIND Header Library")
# set(PYBIND_INCLUDE_DIR "//home/hadoop-hmart-waimai-rank/int-linear/torch_int/ft_gemm/pybind11/include" CACHE PATH "PYBIND Header Library")
set(PYTHON_INCLUDE_DIR "/usr/local/include/python3.9" CACHE PATH "PYTHON Header Library")
set(TORCH_INCLUDE_DIR1 "/usr/local/lib/python3.9/site-packages/torch/include" CACHE PATH "Torch")
set(TORCH_INCLUDE_DIR2 "/usr/local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include" CACHE PATH "Torch")
# set(PYBIND11_DIR "/usr/local/lib/python3.9/site-packages/pybind11")
include_directories(${CUDA_INCLUDE_DIR} ${PYBIND_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} ${TORCH_INCLUDE_DIR1} ${TORCH_INCLUDE_DIR2})
set(TORCH_LIBRARIES "/usr/local/lib/python3.9/site-packages/torch/lib")
link_directories("/usr/local/cuda/lib64" "/usr/local/lib/python3.9/site-packages/torch/lib" "/usr/local/lib/python3.9")
set(PYTHON_EXECUTABLE "/usr/local/bin/python3")

add_library(cuda_utils STATIC cuda_utils.cc)
set_property(TARGET cuda_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cuda_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cuda_utils PUBLIC -lcudart)

add_library(cublasAlgoMap STATIC cublasAlgoMap.cc)
set_property(TARGET cublasAlgoMap PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasAlgoMap PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cublasAlgoMap PUBLIC -lcublas -lcudart -lcurand cuda_utils)

add_library(cublasMMWrapper STATIC cublasMMWrapper.cc)
set_property(TARGET cublasMMWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasMMWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cublasMMWrapper PUBLIC -lcublas -lcudart -lcurand cublasAlgoMap cuda_utils)

add_library(cublasINT8MMWrapper STATIC cublasINT8MMWrapper.cc)
set_property(TARGET cublasINT8MMWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasINT8MMWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(cublasINT8MMWrapper PUBLIC -lcublasLt -lcudart -lcurand -lcublas cublasAlgoMap cublasMMWrapper cuda_utils)

add_subdirectory(pybind11)
# add_library(ftgemm SHARED bindings.cpp common.h)
# add_library(ftgemm SHARED bindings.cpp)
pybind11_add_module(ftgemm MODULE bindings.cpp)
# set_property(TARGET ftgemm PROPERTY POSITION_INDEPENDENT_CODE  ON)
# set_property(TARGET ftgemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ftgemm PUBLIC ${PYTHON_LIBRARIES} torch -lcudart -lcublas -lcublasLt -ldl cublasINT8MMWrapper cublasAlgoMap)






