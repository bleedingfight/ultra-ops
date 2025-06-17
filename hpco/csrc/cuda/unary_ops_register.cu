#include "ops_cuda.h"

namespace hpco {
// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(hpco, CUDA, m) { m.impl("elu", &elu_cuda); }
} // namespace hpco
