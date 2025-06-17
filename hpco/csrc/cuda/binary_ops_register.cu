
#include "ops_cuda.h"
namespace hpco {
// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(hpco, CUDA, m) {
    m.impl("mymuladd", &mymuladd_cuda);
    m.impl("mymul", &mymul_cuda);
    m.impl("myadd_out", &myadd_out_cuda);
}
} // namespace hpco
