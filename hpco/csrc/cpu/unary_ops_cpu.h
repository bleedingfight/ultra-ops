#pragma once
#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
namespace hpco {
at::Tensor elu_cpu(const at::Tensor &a);
void elu_out_cpu(const at::Tensor &a, at::Tensor &out);
} // namespace hpco
