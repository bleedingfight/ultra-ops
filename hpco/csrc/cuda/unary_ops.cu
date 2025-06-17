#include "ops_cuda.h"
#include "unary_ops.cuh"
namespace hpco {

void elu_out_cuda(const at::Tensor &a, at::Tensor &out) {
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_CHECK(out.dtype() == at::kFloat);
    TORCH_CHECK(out.is_contiguous());
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    const float *a_ptr = a_contig.data_ptr<float>();
    float *result_ptr = out.data_ptr<float>();
    int numel = a_contig.numel();
    elu_kernel_fp32<<<(numel + 255) / 256, 256>>>(result_ptr, a_ptr, numel);
}

at::Tensor elu_cuda(const at::Tensor &a) {
    TORCH_CHECK(a.dtype() == at::kFloat);

    at::Tensor a_cont = a.contiguous();
    at::Tensor out = at::empty(a_cont.sizes(), a_cont.options());
    TORCH_CHECK(out.dtype() == at::kFloat);
    TORCH_CHECK(out.is_contiguous());
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    const float *a_ptr = a_cont.data_ptr<float>();
    float *result_ptr = out.data_ptr<float>();
    int numel = a_cont.numel();
    elu_kernel_fp32<<<(numel + 255) / 256, 256>>>(result_ptr, a_ptr, numel);
    return out;
}

} // namespace hpco
