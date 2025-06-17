#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject *PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,   /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        NULL, /* methods */
    };
    return PyModule_Create(&module_def);
}
}

namespace hpco {

at::Tensor mymuladd_cpu(const at::Tensor &a, const at::Tensor &b, double c) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
    }
    return result;
}

at::Tensor mymul_cpu(const at::Tensor &a, const at::Tensor &b) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); i++) {
        result_ptr[i] = a_ptr[i] * b_ptr[i];
    }
    return result;
}

// An example of an operator that mutates one of its inputs.
void myadd_out_cpu(const at::Tensor &a, const at::Tensor &b, at::Tensor &out) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(b.sizes() == out.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_CHECK(out.dtype() == at::kFloat);
    TORCH_CHECK(out.is_contiguous());
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = out.data_ptr<float>();
    for (int64_t i = 0; i < out.numel(); i++) {
        result_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

at::Tensor elu_cpu(const at::Tensor &a) {
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
    const float *a_ptr = a_contig.data_ptr<float>();
    float *result_ptr = result.data_ptr<float>();
    float alpha = 1.f;
    for (int64_t i = 0; i < result.numel(); i++) {
        result_ptr[i] =
            a_ptr[i] >= 0 ? a_ptr[i]
                          : alpha * static_cast<float>(std::exp(a_ptr[i] - 1));
    }
    return result;
}

// An example of an operator that mutates one of its inputs.
void elu_out_cpu(const at::Tensor &a, at::Tensor &out) {
    TORCH_CHECK(a.sizes() == out.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(out.dtype() == at::kFloat);
    TORCH_CHECK(out.is_contiguous());
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CPU);
    at::Tensor a_contig = a.contiguous();
    const float *a_ptr = a_contig.data_ptr<float>();
    float *result_ptr = out.data_ptr<float>();
    float alpha = 1.f;
    for (int64_t i = 0; i < out.numel(); i++) {
        result_ptr[i] =
            a_ptr[i] >= 0 ? a_ptr[i]
                          : alpha * static_cast<float>(std::exp(a_ptr[i] - 1));
    }
}

// Defines the operators
TORCH_LIBRARY(hpco, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
    m.def("mymul(Tensor a, Tensor b) -> Tensor");
    m.def("elu(Tensor a) -> Tensor");
    m.def("elu_out(Tensor a, Tensor(a!) out) -> ()");
    m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
}

// Registers CPU implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(hpco, CPU, m) {
    m.impl("mymuladd", &mymuladd_cpu);
    m.impl("mymul", &mymul_cpu);
    m.impl("myadd_out", &myadd_out_cpu);
    m.impl("elu", &elu_cpu);
    m.impl("elu_out", &elu_out_cpu);
}

} // namespace hpco
