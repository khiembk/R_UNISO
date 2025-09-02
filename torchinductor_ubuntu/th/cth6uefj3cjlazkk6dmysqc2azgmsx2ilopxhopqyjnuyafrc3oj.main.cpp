
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const int64_t* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = 0;
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                            auto tmp1 = 32128L;
                            auto tmp2 = c10::convert<int64_t>(tmp1);
                            auto tmp3 = int64_t(tmp0 + tmp2);
                            auto tmp4 = tmp0 < 0;
                            auto tmp5 = tmp4 ? tmp3 : tmp0;
                            auto tmp6 = tmp5;
                            auto tmp7 = c10::convert<int64_t>(tmp6);
                            TORCH_CHECK((0 <= tmp7) & (tmp7 < 32128L), "index out of bounds: 0 <= tmp7 < 32128L");
                            auto tmp9 = in_ptr1[static_cast<int64_t>(x1 + 512L*tmp5)];
                            auto tmp10 = float(tmp9 * tmp9);
                            tmp_acc0 = tmp_acc0 + tmp10;
                        }
                    }
                }
                out_ptr0[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = in_ptr2[static_cast<int64_t>(x1)];
                        auto tmp1 = in_ptr0[static_cast<int64_t>(x0)];
                        auto tmp11 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp2 = 32128L;
                        auto tmp3 = c10::convert<int64_t>(tmp2);
                        auto tmp4 = int64_t(tmp1 + tmp3);
                        auto tmp5 = tmp1 < 0;
                        auto tmp6 = tmp5 ? tmp4 : tmp1;
                        auto tmp7 = tmp6;
                        auto tmp8 = c10::convert<int64_t>(tmp7);
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 32128L), "index out of bounds: 0 <= tmp8 < 32128L");
                        auto tmp10 = in_ptr1[static_cast<int64_t>(x1 + 512L*tmp6)];
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = float(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = float(tmp10 * tmp16);
                        auto tmp18 = float(tmp0 * tmp17);
                        out_ptr1[static_cast<int64_t>(x1 + 512L*x0)] = tmp18;
                    }
                }
            }
        }
    }
}

// Python bindings to call kernel():
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <sstream>
#include <cstdlib>

#ifndef _MSC_VER
#if __cplusplus < 202002L
// C++20 (earlier) code
// https://en.cppreference.com/w/cpp/language/attributes/likely
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#endif
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

// This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
// We manually link it below to workaround issues with fbcode build.
static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
    static_assert(std::is_pointer_v<T>, "arg type must be pointer or long");
    return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
}
template <> inline int64_t parse_arg<int64_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == -1 && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return result;
}
template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
    if(unlikely(result == reinterpret_cast<void*>(-1) && PyErr_Occurred()))
        throw std::runtime_error("expected int arg");
    return reinterpret_cast<uintptr_t>(result);
}



static PyObject* kernel_py(PyObject* self, PyObject* args) {
    try {
        if(unlikely(!PyTuple_CheckExact(args)))
            throw std::runtime_error("tuple args required");
        if(unlikely(PyTuple_GET_SIZE(args) != 5))
            throw std::runtime_error("requires 5 args");
        kernel(parse_arg<int64_t*>(args, 0), parse_arg<float*>(args, 1), parse_arg<float*>(args, 2), parse_arg<float*>(args, 3), parse_arg<float*>(args, 4)); Py_RETURN_NONE;
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}

static PyMethodDef py_methods[] = {
    {"kernel", kernel_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef py_module =
    {PyModuleDef_HEAD_INIT, "kernel", NULL, -1, py_methods};

PyMODINIT_FUNC PyInit_kernel(void) {
    const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
    if(!str_addr) {
        PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
        return nullptr;
    }
    std::istringstream iss(str_addr);
    uintptr_t addr = 0;
    iss >> addr;
    _torchinductor_pyobject_tensor_data_ptr =
        reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
    PyObject* module = PyModule_Create(&py_module);
    if (module == NULL) {
        return NULL;
    }
    #ifdef Py_GIL_DISABLED
        PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
    #endif
    return module;
}
