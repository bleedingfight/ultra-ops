#include "unary_ops.h"
#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

namespace hpco {
// Defines the operators
TORCH_LIBRARY(hpco, m) {
    m.def("elu(Tensor a) -> Tensor");
    m.def("elu_out(Tensor a, Tensor(a!) out) -> ()");
}

// Registers CPU implementations for elu,elu_out
TORCH_LIBRARY_IMPL(hpco, CPU, m) {
    m.impl("elu", &elu_cpu);
    m.impl("elu_out", &elu_out_cpu);
}

} // namespace hpco
