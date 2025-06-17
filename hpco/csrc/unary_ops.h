#pragma once
#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
namespace hpco {
at::Tensor elu_cpu(const at::Tensor &a);
// An example of an operator that mutates one of its inputs.
void elu_out_cpu(const at::Tensor &a, at::Tensor &out);
} // namespace hpco
