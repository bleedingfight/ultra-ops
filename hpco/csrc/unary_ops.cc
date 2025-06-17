#include "unary_ops.h"
namespace hpco {
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

} // namespace hpco
