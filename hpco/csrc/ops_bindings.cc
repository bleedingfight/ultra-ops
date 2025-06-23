#include "cpu/binary_ops_cpu.h"
#include "cpu/unary_ops_cpu.h"
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
// Defines the operators
TORCH_LIBRARY(hpco, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
    m.def("mymul(Tensor a, Tensor b) -> Tensor");
    m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
    m.def("elu(Tensor a) -> Tensor");
    m.def("elu_out(Tensor a, Tensor(a!) out) -> ()");
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
