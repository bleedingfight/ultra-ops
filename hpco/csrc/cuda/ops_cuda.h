#pragma once
#include <ATen/Operators.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>
namespace hpco {

at::Tensor mymuladd_cuda(const at::Tensor &a, const at::Tensor &b, double c);
at::Tensor mymul_cuda(const at::Tensor &a, const at::Tensor &b);
void myadd_out_cuda(const at::Tensor &a, const at::Tensor &b, at::Tensor &out);
void elu_out_cuda(const at::Tensor &a, at::Tensor &out);
at::Tensor elu_cuda(const at::Tensor &a);
} // namespace hpco
