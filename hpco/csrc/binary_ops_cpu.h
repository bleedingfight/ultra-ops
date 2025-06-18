#pragma once
#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
namespace hpco {

at::Tensor mymuladd_cpu(const at::Tensor &a, const at::Tensor &b, double c);
at::Tensor mymul_cpu(const at::Tensor &a, const at::Tensor &b);
void myadd_out_cpu(const at::Tensor &a, const at::Tensor &b, at::Tensor &out);
} // namespace hpco
