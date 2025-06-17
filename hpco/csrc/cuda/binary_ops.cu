#include "binary_ops.cuh"
#include "ops_cuda.h"
namespace hpco {

at::Tensor mymuladd_cuda(const at::Tensor &a, const at::Tensor &b, double c) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = result.data_ptr<float>();

    int numel = a_contig.numel();
    muladd_kernel<<<(numel + 255) / 256, 256>>>(numel, a_ptr, b_ptr, c,
                                                result_ptr);
    return result;
}

at::Tensor mymul_cuda(const at::Tensor &a, const at::Tensor &b) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = result.data_ptr<float>();
    int numel = a_contig.numel();
    mul_kernel<<<(numel + 255) / 256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
    return result;
}

void myadd_out_cuda(const at::Tensor &a, const at::Tensor &b, at::Tensor &out) {
    TORCH_CHECK(a.sizes() == b.sizes());
    TORCH_CHECK(b.sizes() == out.sizes());
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(b.dtype() == at::kFloat);
    TORCH_CHECK(out.dtype() == at::kFloat);
    TORCH_CHECK(out.is_contiguous());
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    const float *a_ptr = a_contig.data_ptr<float>();
    const float *b_ptr = b_contig.data_ptr<float>();
    float *result_ptr = out.data_ptr<float>();
    int numel = a_contig.numel();
    add_kernel<<<(numel + 255) / 256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
}

} // namespace hpco
