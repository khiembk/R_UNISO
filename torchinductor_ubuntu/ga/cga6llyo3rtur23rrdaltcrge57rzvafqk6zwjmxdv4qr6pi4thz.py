# AOT ID: ['16_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.cpp_pybinding(['const int64_t*', 'const float*', 'const float*', 'float*', 'float*'], '''
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
''')


cpp_fused__softmax__to_copy_add_mul_rsub_1 = async_compile.cpp_pybinding(['const float*', 'const int64_t*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       const int64_t* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp35 = in_ptr1[static_cast<int64_t>(x2)];
                            auto tmp0 = x2 + ((-1L)*x1);
                            auto tmp1 = c10::convert<int64_t>(tmp0);
                            auto tmp2 = static_cast<int64_t>(0);
                            auto tmp3 = tmp1 > tmp2;
                            auto tmp4 = c10::convert<int64_t>(tmp3);
                            auto tmp5 = static_cast<int64_t>(16);
                            auto tmp6 = int64_t(tmp4 * tmp5);
                            auto tmp7 = int64_t(tmp6 + tmp2);
                            auto tmp8 = std::abs(x1 + ((-1L)*x2));
                            auto tmp9 = c10::convert<int64_t>(tmp8);
                            auto tmp10 = static_cast<int64_t>(8);
                            auto tmp11 = tmp9 < tmp10;
                            auto tmp12 = c10::convert<float>(tmp8);
                            auto tmp13 = static_cast<float>(0.125);
                            auto tmp14 = float(tmp12 * tmp13);
                            auto tmp15 = std::log(tmp14);
                            auto tmp16 = static_cast<float>(0.36067376022224085);
                            auto tmp17 = float(tmp15 * tmp16);
                            auto tmp18 = static_cast<float>(8.0);
                            auto tmp19 = float(tmp17 * tmp18);
                            auto tmp20 = c10::convert<int64_t>(tmp19);
                            auto tmp21 = int64_t(tmp20 + tmp10);
                            auto tmp22 = static_cast<int64_t>(15);
                            auto tmp23 = min_propagate_nan(tmp21, tmp22);
                            auto tmp24 = tmp11 ? tmp9 : tmp23;
                            auto tmp25 = int64_t(tmp7 + tmp24);
                            auto tmp26 = 32L;
                            auto tmp27 = c10::convert<int64_t>(tmp26);
                            auto tmp28 = int64_t(tmp25 + tmp27);
                            auto tmp29 = tmp25 < 0;
                            auto tmp30 = tmp29 ? tmp28 : tmp25;
                            auto tmp31 = tmp30;
                            auto tmp32 = c10::convert<int64_t>(tmp31);
                            TORCH_CHECK((0 <= tmp32) & (tmp32 < 32L), "index out of bounds: 0 <= tmp32 < 32L");
                            auto tmp34 = in_ptr0[static_cast<int64_t>(x0 + 8L*tmp30)];
                            auto tmp36 = c10::convert<float>(tmp35);
                            auto tmp37 = static_cast<float>(1.0);
                            auto tmp38 = float(tmp37 - tmp36);
                            auto tmp39 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp40 = float(tmp38 * tmp39);
                            auto tmp41 = float(tmp34 + tmp40);
                            out_ptr0[static_cast<int64_t>(x2 + 64L*x1 + 4096L*x0)] = tmp41;
                        }
                    }
                }
            }
        }
    }
    {
        std::unique_ptr<float []> buf_local_buffer_data_0 = std::make_unique<float []>(64L);
        float* local_buffer_data_0 = buf_local_buffer_data_0.get();
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr2[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp1 = out_ptr0[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                        }
                    }
                }
                out_ptr1[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            {
                float tmp_acc0 = 0;
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr2[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp1 = out_ptr0[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp3 = out_ptr1[static_cast<int64_t>(x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp4 = float(tmp2 - tmp3);
                            auto tmp5 = std::exp(tmp4);
                            local_buffer_data_0[static_cast<int64_t>(x1)] = tmp5;
                            tmp_acc0 = tmp_acc0 + tmp5;
                        }
                    }
                }
                out_ptr2[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = local_buffer_data_0[static_cast<int64_t>(x1)];
                        auto tmp1 = out_ptr2[static_cast<int64_t>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr3[static_cast<int64_t>(x1 + 64L*x0)] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_2 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x2 + 64L*x0 + 4096L*x1)];
                            out_ptr0[static_cast<int64_t>(x2 + 64L*x1 + 512L*x0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_3 = async_compile.cpp_pybinding(['const int64_t*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const int64_t* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                            auto tmp10 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp1 = 32128L;
                            auto tmp2 = c10::convert<int64_t>(tmp1);
                            auto tmp3 = int64_t(tmp0 + tmp2);
                            auto tmp4 = tmp0 < 0;
                            auto tmp5 = tmp4 ? tmp3 : tmp0;
                            auto tmp6 = tmp5;
                            auto tmp7 = c10::convert<int64_t>(tmp6);
                            TORCH_CHECK((0 <= tmp7) & (tmp7 < 32128L), "index out of bounds: 0 <= tmp7 < 32128L");
                            auto tmp9 = in_ptr1[static_cast<int64_t>(x1 + 512L*tmp5)];
                            auto tmp11 = float(tmp9 + tmp10);
                            auto tmp12 = float(tmp11 * tmp11);
                            tmp_acc0 = tmp_acc0 + tmp12;
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
                        auto tmp0 = in_ptr3[static_cast<int64_t>(x1)];
                        auto tmp1 = in_ptr0[static_cast<int64_t>(x0)];
                        auto tmp11 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp13 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp2 = 32128L;
                        auto tmp3 = c10::convert<int64_t>(tmp2);
                        auto tmp4 = int64_t(tmp1 + tmp3);
                        auto tmp5 = tmp1 < 0;
                        auto tmp6 = tmp5 ? tmp4 : tmp1;
                        auto tmp7 = tmp6;
                        auto tmp8 = c10::convert<int64_t>(tmp7);
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 32128L), "index out of bounds: 0 <= tmp8 < 32128L");
                        auto tmp10 = in_ptr1[static_cast<int64_t>(x1 + 512L*tmp6)];
                        auto tmp12 = float(tmp10 + tmp11);
                        auto tmp14 = static_cast<float>(512.0);
                        auto tmp15 = tmp13 / tmp14;
                        auto tmp16 = static_cast<float>(1e-06);
                        auto tmp17 = float(tmp15 + tmp16);
                        auto tmp18 = 1 / std::sqrt(tmp17);
                        auto tmp19 = float(tmp12 * tmp18);
                        auto tmp20 = float(tmp0 * tmp19);
                        out_ptr1[static_cast<int64_t>(x1 + 512L*x0)] = tmp20;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_4 = async_compile.cpp_pybinding(['float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(float* in_out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(131072L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = std::max(tmp0, decltype(tmp0)(0));
                    in_out_ptr0[static_cast<int64_t>(x0)] = tmp1;
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_5 = async_compile.cpp_pybinding(['const int64_t*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const int64_t* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                            auto tmp10 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp12 = in_ptr3[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp1 = 32128L;
                            auto tmp2 = c10::convert<int64_t>(tmp1);
                            auto tmp3 = int64_t(tmp0 + tmp2);
                            auto tmp4 = tmp0 < 0;
                            auto tmp5 = tmp4 ? tmp3 : tmp0;
                            auto tmp6 = tmp5;
                            auto tmp7 = c10::convert<int64_t>(tmp6);
                            TORCH_CHECK((0 <= tmp7) & (tmp7 < 32128L), "index out of bounds: 0 <= tmp7 < 32128L");
                            auto tmp9 = in_ptr1[static_cast<int64_t>(x1 + 512L*tmp5)];
                            auto tmp11 = float(tmp9 + tmp10);
                            auto tmp13 = float(tmp11 + tmp12);
                            auto tmp14 = float(tmp13 * tmp13);
                            tmp_acc0 = tmp_acc0 + tmp14;
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
                        auto tmp0 = in_ptr4[static_cast<int64_t>(x1)];
                        auto tmp1 = in_ptr0[static_cast<int64_t>(x0)];
                        auto tmp11 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp13 = in_ptr3[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp15 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp2 = 32128L;
                        auto tmp3 = c10::convert<int64_t>(tmp2);
                        auto tmp4 = int64_t(tmp1 + tmp3);
                        auto tmp5 = tmp1 < 0;
                        auto tmp6 = tmp5 ? tmp4 : tmp1;
                        auto tmp7 = tmp6;
                        auto tmp8 = c10::convert<int64_t>(tmp7);
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 32128L), "index out of bounds: 0 <= tmp8 < 32128L");
                        auto tmp10 = in_ptr1[static_cast<int64_t>(x1 + 512L*tmp6)];
                        auto tmp12 = float(tmp10 + tmp11);
                        auto tmp14 = float(tmp12 + tmp13);
                        auto tmp16 = static_cast<float>(512.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-06);
                        auto tmp19 = float(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = float(tmp14 * tmp20);
                        auto tmp22 = float(tmp0 * tmp21);
                        out_ptr1[static_cast<int64_t>(x1 + 512L*x0)] = tmp22;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_6 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        std::unique_ptr<float []> buf_local_buffer_data_0 = std::make_unique<float []>(64L);
        float* local_buffer_data_0 = buf_local_buffer_data_0.get();
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                        }
                    }
                }
                out_ptr0[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            {
                float tmp_acc0 = 0;
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp4 = float(tmp2 - tmp3);
                            auto tmp5 = std::exp(tmp4);
                            local_buffer_data_0[static_cast<int64_t>(x1)] = tmp5;
                            tmp_acc0 = tmp_acc0 + tmp5;
                        }
                    }
                }
                out_ptr1[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = local_buffer_data_0[static_cast<int64_t>(x1)];
                        auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr2[static_cast<int64_t>(x1 + 64L*x0)] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_7 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x2 + 64L*x0 + 4096L*x1)];
                            out_ptr0[static_cast<int64_t>(x2 + 64L*x1 + 512L*x0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_8 = async_compile.cpp_pybinding(['float*', 'const int64_t*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(float* in_out_ptr0,
                       const int64_t* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                            auto tmp10 = in_out_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp12 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp14 = in_ptr3[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp1 = 32128L;
                            auto tmp2 = c10::convert<int64_t>(tmp1);
                            auto tmp3 = int64_t(tmp0 + tmp2);
                            auto tmp4 = tmp0 < 0;
                            auto tmp5 = tmp4 ? tmp3 : tmp0;
                            auto tmp6 = tmp5;
                            auto tmp7 = c10::convert<int64_t>(tmp6);
                            TORCH_CHECK((0 <= tmp7) & (tmp7 < 32128L), "index out of bounds: 0 <= tmp7 < 32128L");
                            auto tmp9 = in_ptr1[static_cast<int64_t>(x1 + 512L*tmp5)];
                            auto tmp11 = float(tmp9 + tmp10);
                            auto tmp13 = float(tmp11 + tmp12);
                            auto tmp15 = float(tmp13 + tmp14);
                            auto tmp16 = float(tmp15 * tmp15);
                            in_out_ptr0[static_cast<int64_t>(x1 + 512L*x0)] = tmp15;
                            tmp_acc0 = tmp_acc0 + tmp16;
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
                        auto tmp0 = in_ptr4[static_cast<int64_t>(x1)];
                        auto tmp1 = in_out_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp2 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp3 = static_cast<float>(512.0);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = static_cast<float>(1e-06);
                        auto tmp6 = float(tmp4 + tmp5);
                        auto tmp7 = 1 / std::sqrt(tmp6);
                        auto tmp8 = float(tmp1 * tmp7);
                        auto tmp9 = float(tmp0 * tmp8);
                        out_ptr1[static_cast<int64_t>(x1 + 512L*x0)] = tmp9;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_9 = async_compile.cpp_pybinding(['float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(float* in_out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(131072L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = std::max(tmp0, decltype(tmp0)(0));
                    in_out_ptr0[static_cast<int64_t>(x0)] = tmp1;
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_10 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
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
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp3 = float(tmp2 * tmp2);
                            tmp_acc0 = tmp_acc0 + tmp3;
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
                        auto tmp1 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp2 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp4 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp3 = float(tmp1 + tmp2);
                        auto tmp5 = static_cast<float>(512.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = float(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = float(tmp3 * tmp9);
                        auto tmp11 = float(tmp0 * tmp10);
                        out_ptr1[static_cast<int64_t>(x1 + 512L*x0)] = tmp11;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_11 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        std::unique_ptr<float []> buf_local_buffer_data_0 = std::make_unique<float []>(64L);
        float* local_buffer_data_0 = buf_local_buffer_data_0.get();
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                        }
                    }
                }
                out_ptr0[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            {
                float tmp_acc0 = 0;
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp4 = float(tmp2 - tmp3);
                            auto tmp5 = std::exp(tmp4);
                            local_buffer_data_0[static_cast<int64_t>(x1)] = tmp5;
                            tmp_acc0 = tmp_acc0 + tmp5;
                        }
                    }
                }
                out_ptr1[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = local_buffer_data_0[static_cast<int64_t>(x1)];
                        auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr2[static_cast<int64_t>(x1 + 64L*x0)] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_12 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x2 + 64L*x0 + 4096L*x1)];
                            out_ptr0[static_cast<int64_t>(x2 + 64L*x1 + 512L*x0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_13 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp3 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp4 = float(tmp2 + tmp3);
                            auto tmp5 = float(tmp4 * tmp4);
                            tmp_acc0 = tmp_acc0 + tmp5;
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
                        auto tmp0 = in_ptr3[static_cast<int64_t>(x1)];
                        auto tmp1 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp2 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp4 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp6 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp3 = float(tmp1 + tmp2);
                        auto tmp5 = float(tmp3 + tmp4);
                        auto tmp7 = static_cast<float>(512.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-06);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = float(tmp5 * tmp11);
                        auto tmp13 = float(tmp0 * tmp12);
                        out_ptr1[static_cast<int64_t>(x1 + 512L*x0)] = tmp13;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_14 = async_compile.cpp_pybinding(['float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(float* in_out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(131072L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = std::max(tmp0, decltype(tmp0)(0));
                    in_out_ptr0[static_cast<int64_t>(x0)] = tmp1;
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_15 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp3 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp5 = in_ptr3[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp4 = float(tmp2 + tmp3);
                            auto tmp6 = float(tmp4 + tmp5);
                            auto tmp7 = float(tmp6 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
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
                        auto tmp0 = in_ptr4[static_cast<int64_t>(x1)];
                        auto tmp1 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp2 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp4 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp6 = in_ptr3[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp8 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp3 = float(tmp1 + tmp2);
                        auto tmp5 = float(tmp3 + tmp4);
                        auto tmp7 = float(tmp5 + tmp6);
                        auto tmp9 = static_cast<float>(512.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = float(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = float(tmp7 * tmp13);
                        auto tmp15 = float(tmp0 * tmp14);
                        out_ptr1[static_cast<int64_t>(x1 + 512L*x0)] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_16 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        std::unique_ptr<float []> buf_local_buffer_data_0 = std::make_unique<float []>(64L);
        float* local_buffer_data_0 = buf_local_buffer_data_0.get();
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                        }
                    }
                }
                out_ptr0[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            {
                float tmp_acc0 = 0;
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp4 = float(tmp2 - tmp3);
                            auto tmp5 = std::exp(tmp4);
                            local_buffer_data_0[static_cast<int64_t>(x1)] = tmp5;
                            tmp_acc0 = tmp_acc0 + tmp5;
                        }
                    }
                }
                out_ptr1[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = local_buffer_data_0[static_cast<int64_t>(x1)];
                        auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr2[static_cast<int64_t>(x1 + 64L*x0)] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_17 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x2 + 64L*x0 + 4096L*x1)];
                            out_ptr0[static_cast<int64_t>(x2 + 64L*x1 + 512L*x0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_18 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(32768L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = in_ptr0[static_cast<int64_t>(x0)];
                    auto tmp3 = in_ptr1[static_cast<int64_t>(x0)];
                    auto tmp5 = in_ptr2[static_cast<int64_t>(x0)];
                    auto tmp7 = in_ptr3[static_cast<int64_t>(x0)];
                    auto tmp2 = float(tmp0 + tmp1);
                    auto tmp4 = float(tmp2 + tmp3);
                    auto tmp6 = float(tmp4 + tmp5);
                    auto tmp8 = float(tmp6 + tmp7);
                    in_out_ptr0[static_cast<int64_t>(x0)] = tmp8;
                }
            }
        }
    }
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
                            auto tmp0 = in_out_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp1 = float(tmp0 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp1;
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
                        auto tmp0 = in_ptr4[static_cast<int64_t>(x1)];
                        auto tmp1 = in_out_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp2 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp3 = static_cast<float>(512.0);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = static_cast<float>(1e-06);
                        auto tmp6 = float(tmp4 + tmp5);
                        auto tmp7 = 1 / std::sqrt(tmp6);
                        auto tmp8 = float(tmp1 * tmp7);
                        auto tmp9 = float(tmp0 * tmp8);
                        out_ptr1[static_cast<int64_t>(x1 + 512L*x0)] = tmp9;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_19 = async_compile.cpp_pybinding(['float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(float* in_out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(131072L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = std::max(tmp0, decltype(tmp0)(0));
                    in_out_ptr0[static_cast<int64_t>(x0)] = tmp1;
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_20 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
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
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp3 = float(tmp2 * tmp2);
                            tmp_acc0 = tmp_acc0 + tmp3;
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
                        auto tmp1 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp2 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp4 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp3 = float(tmp1 + tmp2);
                        auto tmp5 = static_cast<float>(512.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = float(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = float(tmp3 * tmp9);
                        auto tmp11 = float(tmp0 * tmp10);
                        out_ptr1[static_cast<int64_t>(x1 + 512L*x0)] = tmp11;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_21 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        std::unique_ptr<float []> buf_local_buffer_data_0 = std::make_unique<float []>(64L);
        float* local_buffer_data_0 = buf_local_buffer_data_0.get();
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                        }
                    }
                }
                out_ptr0[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            {
                float tmp_acc0 = 0;
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp4 = float(tmp2 - tmp3);
                            auto tmp5 = std::exp(tmp4);
                            local_buffer_data_0[static_cast<int64_t>(x1)] = tmp5;
                            tmp_acc0 = tmp_acc0 + tmp5;
                        }
                    }
                }
                out_ptr1[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = local_buffer_data_0[static_cast<int64_t>(x1)];
                        auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr2[static_cast<int64_t>(x1 + 64L*x0)] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_22 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x2 + 64L*x0 + 4096L*x1)];
                            out_ptr0[static_cast<int64_t>(x2 + 64L*x1 + 512L*x0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_23 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp3 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp4 = float(tmp2 + tmp3);
                            auto tmp5 = float(tmp4 * tmp4);
                            tmp_acc0 = tmp_acc0 + tmp5;
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
                        auto tmp0 = in_ptr3[static_cast<int64_t>(x1)];
                        auto tmp1 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp2 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp4 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp6 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp3 = float(tmp1 + tmp2);
                        auto tmp5 = float(tmp3 + tmp4);
                        auto tmp7 = static_cast<float>(512.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-06);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = float(tmp5 * tmp11);
                        auto tmp13 = float(tmp0 * tmp12);
                        out_ptr1[static_cast<int64_t>(x1 + 512L*x0)] = tmp13;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_24 = async_compile.cpp_pybinding(['float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(float* in_out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(131072L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = std::max(tmp0, decltype(tmp0)(0));
                    in_out_ptr0[static_cast<int64_t>(x0)] = tmp1;
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_25 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp3 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp5 = in_ptr3[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp4 = float(tmp2 + tmp3);
                            auto tmp6 = float(tmp4 + tmp5);
                            auto tmp7 = float(tmp6 * tmp6);
                            tmp_acc0 = tmp_acc0 + tmp7;
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
                        auto tmp0 = in_ptr4[static_cast<int64_t>(x1)];
                        auto tmp1 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp2 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp4 = in_ptr2[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp6 = in_ptr3[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp8 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp3 = float(tmp1 + tmp2);
                        auto tmp5 = float(tmp3 + tmp4);
                        auto tmp7 = float(tmp5 + tmp6);
                        auto tmp9 = static_cast<float>(512.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-06);
                        auto tmp12 = float(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = float(tmp7 * tmp13);
                        auto tmp15 = float(tmp0 * tmp14);
                        out_ptr1[static_cast<int64_t>(x1 + 512L*x0)] = tmp15;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_26 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        std::unique_ptr<float []> buf_local_buffer_data_0 = std::make_unique<float []>(64L);
        float* local_buffer_data_0 = buf_local_buffer_data_0.get();
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
        {
            {
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                        }
                    }
                }
                out_ptr0[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            {
                float tmp_acc0 = 0;
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 64L*x0)];
                            auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp4 = float(tmp2 - tmp3);
                            auto tmp5 = std::exp(tmp4);
                            local_buffer_data_0[static_cast<int64_t>(x1)] = tmp5;
                            tmp_acc0 = tmp_acc0 + tmp5;
                        }
                    }
                }
                out_ptr1[static_cast<int64_t>(x0)] = tmp_acc0;
            }
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = local_buffer_data_0[static_cast<int64_t>(x1)];
                        auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr2[static_cast<int64_t>(x1 + 64L*x0)] = tmp2;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_27 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x2 + 64L*x0 + 4096L*x1)];
                            out_ptr0[static_cast<int64_t>(x2 + 64L*x1 + 512L*x0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_28 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(32768L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = in_ptr0[static_cast<int64_t>(x0)];
                    auto tmp3 = in_ptr1[static_cast<int64_t>(x0)];
                    auto tmp5 = in_ptr2[static_cast<int64_t>(x0)];
                    auto tmp7 = in_ptr3[static_cast<int64_t>(x0)];
                    auto tmp2 = float(tmp0 + tmp1);
                    auto tmp4 = float(tmp2 + tmp3);
                    auto tmp6 = float(tmp4 + tmp5);
                    auto tmp8 = float(tmp6 + tmp7);
                    in_out_ptr0[static_cast<int64_t>(x0)] = tmp8;
                }
            }
        }
    }
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
                            auto tmp0 = in_out_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp1 = float(tmp0 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp1;
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
                        auto tmp0 = in_ptr4[static_cast<int64_t>(x1)];
                        auto tmp1 = in_out_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp2 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp3 = static_cast<float>(512.0);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = static_cast<float>(1e-06);
                        auto tmp6 = float(tmp4 + tmp5);
                        auto tmp7 = 1 / std::sqrt(tmp6);
                        auto tmp8 = float(tmp1 * tmp7);
                        auto tmp9 = float(tmp0 * tmp8);
                        out_ptr1[static_cast<int64_t>(x1 + 512L*x0)] = tmp9;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_relu_29 = async_compile.cpp_pybinding(['float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(float* in_out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(131072L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = std::max(tmp0, decltype(tmp0)(0));
                    in_out_ptr0[static_cast<int64_t>(x0)] = tmp1;
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_30 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp1 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                            auto tmp2 = float(tmp0 + tmp1);
                            auto tmp3 = float(tmp2 * tmp2);
                            tmp_acc0 = tmp_acc0 + tmp3;
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
                        auto tmp1 = in_out_ptr0[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp2 = in_ptr1[static_cast<int64_t>(x1 + 512L*x0)];
                        auto tmp4 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp3 = float(tmp1 + tmp2);
                        auto tmp5 = static_cast<float>(512.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = float(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = float(tmp3 * tmp9);
                        auto tmp11 = float(tmp0 * tmp10);
                        in_out_ptr0[static_cast<int64_t>(x1 + 512L*x0)] = tmp11;
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 64), (64, 1))
    assert_size_stride(arg1_1, (32128, 512), (512, 1))
    assert_size_stride(arg2_1, (1, 64), (64, 1))
    assert_size_stride(arg3_1, (512, ), (1, ))
    assert_size_stride(arg4_1, (512, 512), (512, 1))
    assert_size_stride(arg5_1, (512, 512), (512, 1))
    assert_size_stride(arg6_1, (512, 512), (512, 1))
    assert_size_stride(arg7_1, (32, 8), (8, 1))
    assert_size_stride(arg8_1, (512, 512), (512, 1))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (2048, 512), (512, 1))
    assert_size_stride(arg11_1, (512, 2048), (2048, 1))
    assert_size_stride(arg12_1, (512, ), (1, ))
    assert_size_stride(arg13_1, (512, 512), (512, 1))
    assert_size_stride(arg14_1, (512, 512), (512, 1))
    assert_size_stride(arg15_1, (512, 512), (512, 1))
    assert_size_stride(arg16_1, (512, 512), (512, 1))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (2048, 512), (512, 1))
    assert_size_stride(arg19_1, (512, 2048), (2048, 1))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (512, 512), (512, 1))
    assert_size_stride(arg22_1, (512, 512), (512, 1))
    assert_size_stride(arg23_1, (512, 512), (512, 1))
    assert_size_stride(arg24_1, (512, 512), (512, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (2048, 512), (512, 1))
    assert_size_stride(arg27_1, (512, 2048), (2048, 1))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, 512), (512, 1))
    assert_size_stride(arg30_1, (512, 512), (512, 1))
    assert_size_stride(arg31_1, (512, 512), (512, 1))
    assert_size_stride(arg32_1, (512, 512), (512, 1))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (2048, 512), (512, 1))
    assert_size_stride(arg35_1, (512, 2048), (2048, 1))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, 512), (512, 1))
    assert_size_stride(arg38_1, (512, 512), (512, 1))
    assert_size_stride(arg39_1, (512, 512), (512, 1))
    assert_size_stride(arg40_1, (512, 512), (512, 1))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (2048, 512), (512, 1))
    assert_size_stride(arg43_1, (512, 2048), (2048, 1))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, 512), (512, 1))
    assert_size_stride(arg46_1, (512, 512), (512, 1))
    assert_size_stride(arg47_1, (512, 512), (512, 1))
    assert_size_stride(arg48_1, (512, 512), (512, 1))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (2048, 512), (512, 1))
    assert_size_stride(arg51_1, (512, 2048), (2048, 1))
    assert_size_stride(arg52_1, (512, ), (1, ))
    buf0 = empty_strided_cpu((1, 64, 1), (64, 1, 64), torch.float32)
    buf1 = empty_strided_cpu((1, 64, 512), (32768, 512, 1), torch.float32)
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_0(arg0_1, arg1_1, arg3_1, buf0, buf1)
    del arg3_1
    buf2 = empty_strided_cpu((64, 512), (512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (64, 512), (512, 1), 0), reinterpret_tensor(arg4_1, (512, 512), (1, 512), 0), out=buf2)
    del arg4_1
    buf3 = empty_strided_cpu((64, 512), (512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (64, 512), (512, 1), 0), reinterpret_tensor(arg5_1, (512, 512), (1, 512), 0), out=buf3)
    del arg5_1
    buf4 = empty_strided_cpu((8, 64, 64), (4096, 64, 1), torch.float32)
    # Topologically Sorted Source Nodes: [scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf2, (8, 64, 64), (64, 512, 1), 0), reinterpret_tensor(buf3, (8, 64, 64), (64, 1, 512), 0), out=buf4)
    buf5 = reinterpret_tensor(buf3, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf3  # reuse
    buf6 = empty_strided_cpu((1, 8, 64, 1), (512, 64, 1, 512), torch.float32)
    buf8 = empty_strided_cpu((1, 8, 64, 1), (512, 64, 1, 512), torch.float32)
    buf10 = reinterpret_tensor(buf2, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf2  # reuse
    cpp_fused__softmax__to_copy_add_mul_rsub_1(arg7_1, arg2_1, buf4, buf5, buf6, buf8, buf10)
    del arg2_1
    del arg7_1
    buf9 = reinterpret_tensor(buf4, (64, 512), (512, 1), 0); del buf4  # reuse
    # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (64, 512), (512, 1), 0), reinterpret_tensor(arg6_1, (512, 512), (1, 512), 0), out=buf9)
    del arg6_1
    buf11 = reinterpret_tensor(buf1, (8, 64, 64), (4096, 64, 1), 0); del buf1  # reuse
    # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf10, (8, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf9, (8, 64, 64), (64, 512, 1), 0), out=buf11)
    buf12 = reinterpret_tensor(buf9, (1, 64, 8, 64), (32768, 512, 64, 1), 0); del buf9  # reuse
    cpp_fused_clone_2(buf11, buf12)
    buf13 = reinterpret_tensor(buf11, (64, 512), (512, 1), 0); del buf11  # reuse
    # Topologically Sorted Source Nodes: [attn_output_1], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf12, (64, 512), (512, 1), 0), reinterpret_tensor(arg8_1, (512, 512), (1, 512), 0), out=buf13)
    del arg8_1
    buf14 = buf0; del buf0  # reuse
    buf15 = reinterpret_tensor(buf12, (1, 64, 512), (32768, 512, 1), 0); del buf12  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_3(arg0_1, arg1_1, buf13, arg9_1, buf14, buf15)
    del arg9_1
    buf16 = empty_strided_cpu((64, 2048), (2048, 1), torch.float32)
    # Topologically Sorted Source Nodes: [hidden_states_6], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf15, (64, 512), (512, 1), 0), reinterpret_tensor(arg10_1, (512, 2048), (1, 512), 0), out=buf16)
    del arg10_1
    buf17 = reinterpret_tensor(buf16, (1, 64, 2048), (131072, 2048, 1), 0); del buf16  # reuse
    cpp_fused_relu_4(buf17)
    buf18 = reinterpret_tensor(buf15, (64, 512), (512, 1), 0); del buf15  # reuse
    # Topologically Sorted Source Nodes: [hidden_states_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf17, (64, 2048), (2048, 1), 0), reinterpret_tensor(arg11_1, (2048, 512), (1, 2048), 0), out=buf18)
    del arg11_1
    buf19 = buf14; del buf14  # reuse
    buf20 = reinterpret_tensor(buf10, (1, 64, 512), (32768, 512, 1), 0); del buf10  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_5(arg0_1, arg1_1, buf13, buf18, arg12_1, buf19, buf20)
    del arg12_1
    buf21 = empty_strided_cpu((64, 512), (512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (64, 512), (512, 1), 0), reinterpret_tensor(arg13_1, (512, 512), (1, 512), 0), out=buf21)
    del arg13_1
    buf22 = empty_strided_cpu((64, 512), (512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (64, 512), (512, 1), 0), reinterpret_tensor(arg14_1, (512, 512), (1, 512), 0), out=buf22)
    del arg14_1
    buf23 = empty_strided_cpu((8, 64, 64), (4096, 64, 1), torch.float32)
    # Topologically Sorted Source Nodes: [scores_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf21, (8, 64, 64), (64, 512, 1), 0), reinterpret_tensor(buf22, (8, 64, 64), (64, 1, 512), 0), out=buf23)
    buf24 = buf8; del buf8  # reuse
    buf26 = buf6; del buf6  # reuse
    buf28 = reinterpret_tensor(buf22, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf22  # reuse
    cpp_fused__softmax_add_6(buf23, buf5, buf24, buf26, buf28)
    buf27 = reinterpret_tensor(buf23, (64, 512), (512, 1), 0); del buf23  # reuse
    # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf20, (64, 512), (512, 1), 0), reinterpret_tensor(arg15_1, (512, 512), (1, 512), 0), out=buf27)
    del arg15_1
    buf29 = reinterpret_tensor(buf20, (8, 64, 64), (4096, 64, 1), 0); del buf20  # reuse
    # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf28, (8, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf27, (8, 64, 64), (64, 512, 1), 0), out=buf29)
    buf30 = reinterpret_tensor(buf28, (1, 64, 8, 64), (32768, 512, 64, 1), 0); del buf28  # reuse
    cpp_fused_clone_7(buf29, buf30)
    buf31 = reinterpret_tensor(buf29, (64, 512), (512, 1), 0); del buf29  # reuse
    # Topologically Sorted Source Nodes: [attn_output_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf30, (64, 512), (512, 1), 0), reinterpret_tensor(arg16_1, (512, 512), (1, 512), 0), out=buf31)
    del arg16_1
    buf32 = reinterpret_tensor(buf13, (1, 64, 512), (32768, 512, 1), 0); del buf13  # reuse
    buf33 = buf19; del buf19  # reuse
    buf34 = reinterpret_tensor(buf30, (1, 64, 512), (32768, 512, 1), 0); del buf30  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_8(buf32, arg0_1, arg1_1, buf18, buf31, arg17_1, buf33, buf34)
    del arg0_1
    del arg17_1
    del arg1_1
    buf35 = reinterpret_tensor(buf17, (64, 2048), (2048, 1), 0); del buf17  # reuse
    # Topologically Sorted Source Nodes: [hidden_states_16], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf34, (64, 512), (512, 1), 0), reinterpret_tensor(arg18_1, (512, 2048), (1, 512), 0), out=buf35)
    del arg18_1
    buf36 = reinterpret_tensor(buf35, (1, 64, 2048), (131072, 2048, 1), 0); del buf35  # reuse
    cpp_fused_relu_9(buf36)
    buf37 = reinterpret_tensor(buf34, (64, 512), (512, 1), 0); del buf34  # reuse
    # Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf36, (64, 2048), (2048, 1), 0), reinterpret_tensor(arg19_1, (2048, 512), (1, 2048), 0), out=buf37)
    del arg19_1
    buf38 = buf33; del buf33  # reuse
    buf39 = reinterpret_tensor(buf31, (1, 64, 512), (32768, 512, 1), 0); del buf31  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_10(buf32, buf37, arg20_1, buf38, buf39)
    del arg20_1
    buf40 = buf18; del buf18  # reuse
    # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (64, 512), (512, 1), 0), reinterpret_tensor(arg21_1, (512, 512), (1, 512), 0), out=buf40)
    del arg21_1
    buf41 = buf27; del buf27  # reuse
    # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (64, 512), (512, 1), 0), reinterpret_tensor(arg22_1, (512, 512), (1, 512), 0), out=buf41)
    del arg22_1
    buf42 = reinterpret_tensor(buf21, (8, 64, 64), (4096, 64, 1), 0); del buf21  # reuse
    # Topologically Sorted Source Nodes: [scores_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf40, (8, 64, 64), (64, 512, 1), 0), reinterpret_tensor(buf41, (8, 64, 64), (64, 1, 512), 0), out=buf42)
    buf43 = buf26; del buf26  # reuse
    buf45 = buf24; del buf24  # reuse
    buf47 = reinterpret_tensor(buf41, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf41  # reuse
    cpp_fused__softmax_add_11(buf42, buf5, buf43, buf45, buf47)
    buf46 = reinterpret_tensor(buf42, (64, 512), (512, 1), 0); del buf42  # reuse
    # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf39, (64, 512), (512, 1), 0), reinterpret_tensor(arg23_1, (512, 512), (1, 512), 0), out=buf46)
    del arg23_1
    buf48 = reinterpret_tensor(buf39, (8, 64, 64), (4096, 64, 1), 0); del buf39  # reuse
    # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf47, (8, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf46, (8, 64, 64), (64, 512, 1), 0), out=buf48)
    buf49 = reinterpret_tensor(buf47, (1, 64, 8, 64), (32768, 512, 64, 1), 0); del buf47  # reuse
    cpp_fused_clone_12(buf48, buf49)
    buf50 = reinterpret_tensor(buf48, (64, 512), (512, 1), 0); del buf48  # reuse
    # Topologically Sorted Source Nodes: [attn_output_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (64, 512), (512, 1), 0), reinterpret_tensor(arg24_1, (512, 512), (1, 512), 0), out=buf50)
    del arg24_1
    buf51 = buf38; del buf38  # reuse
    buf52 = reinterpret_tensor(buf49, (1, 64, 512), (32768, 512, 1), 0); del buf49  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_13(buf32, buf37, buf50, arg25_1, buf51, buf52)
    del arg25_1
    buf53 = reinterpret_tensor(buf36, (64, 2048), (2048, 1), 0); del buf36  # reuse
    # Topologically Sorted Source Nodes: [hidden_states_26], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf52, (64, 512), (512, 1), 0), reinterpret_tensor(arg26_1, (512, 2048), (1, 512), 0), out=buf53)
    del arg26_1
    buf54 = reinterpret_tensor(buf53, (1, 64, 2048), (131072, 2048, 1), 0); del buf53  # reuse
    cpp_fused_relu_14(buf54)
    buf55 = reinterpret_tensor(buf52, (64, 512), (512, 1), 0); del buf52  # reuse
    # Topologically Sorted Source Nodes: [hidden_states_29], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf54, (64, 2048), (2048, 1), 0), reinterpret_tensor(arg27_1, (2048, 512), (1, 2048), 0), out=buf55)
    del arg27_1
    buf56 = buf51; del buf51  # reuse
    buf57 = reinterpret_tensor(buf46, (1, 64, 512), (32768, 512, 1), 0); del buf46  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_15(buf32, buf37, buf50, buf55, arg28_1, buf56, buf57)
    del arg28_1
    buf58 = buf40; del buf40  # reuse
    # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (64, 512), (512, 1), 0), reinterpret_tensor(arg29_1, (512, 512), (1, 512), 0), out=buf58)
    del arg29_1
    buf59 = empty_strided_cpu((64, 512), (512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (64, 512), (512, 1), 0), reinterpret_tensor(arg30_1, (512, 512), (1, 512), 0), out=buf59)
    del arg30_1
    buf60 = empty_strided_cpu((8, 64, 64), (4096, 64, 1), torch.float32)
    # Topologically Sorted Source Nodes: [scores_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf58, (8, 64, 64), (64, 512, 1), 0), reinterpret_tensor(buf59, (8, 64, 64), (64, 1, 512), 0), out=buf60)
    buf61 = buf45; del buf45  # reuse
    buf63 = buf43; del buf43  # reuse
    buf65 = reinterpret_tensor(buf59, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf59  # reuse
    cpp_fused__softmax_add_16(buf60, buf5, buf61, buf63, buf65)
    buf64 = reinterpret_tensor(buf60, (64, 512), (512, 1), 0); del buf60  # reuse
    # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (64, 512), (512, 1), 0), reinterpret_tensor(arg31_1, (512, 512), (1, 512), 0), out=buf64)
    del arg31_1
    buf66 = reinterpret_tensor(buf57, (8, 64, 64), (4096, 64, 1), 0); del buf57  # reuse
    # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf65, (8, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf64, (8, 64, 64), (64, 512, 1), 0), out=buf66)
    buf67 = reinterpret_tensor(buf65, (1, 64, 8, 64), (32768, 512, 64, 1), 0); del buf65  # reuse
    cpp_fused_clone_17(buf66, buf67)
    buf68 = reinterpret_tensor(buf66, (64, 512), (512, 1), 0); del buf66  # reuse
    # Topologically Sorted Source Nodes: [attn_output_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (64, 512), (512, 1), 0), reinterpret_tensor(arg32_1, (512, 512), (1, 512), 0), out=buf68)
    del arg32_1
    buf69 = buf32; del buf32  # reuse
    buf70 = buf56; del buf56  # reuse
    buf71 = reinterpret_tensor(buf67, (1, 64, 512), (32768, 512, 1), 0); del buf67  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_18(buf69, buf37, buf50, buf55, buf68, arg33_1, buf70, buf71)
    del arg33_1
    buf72 = reinterpret_tensor(buf54, (64, 2048), (2048, 1), 0); del buf54  # reuse
    # Topologically Sorted Source Nodes: [hidden_states_36], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf71, (64, 512), (512, 1), 0), reinterpret_tensor(arg34_1, (512, 2048), (1, 512), 0), out=buf72)
    del arg34_1
    buf73 = reinterpret_tensor(buf72, (1, 64, 2048), (131072, 2048, 1), 0); del buf72  # reuse
    cpp_fused_relu_19(buf73)
    buf74 = reinterpret_tensor(buf71, (64, 512), (512, 1), 0); del buf71  # reuse
    # Topologically Sorted Source Nodes: [hidden_states_39], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf73, (64, 2048), (2048, 1), 0), reinterpret_tensor(arg35_1, (2048, 512), (1, 2048), 0), out=buf74)
    del arg35_1
    buf75 = buf70; del buf70  # reuse
    buf76 = reinterpret_tensor(buf68, (1, 64, 512), (32768, 512, 1), 0); del buf68  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_20(buf69, buf74, arg36_1, buf75, buf76)
    del arg36_1
    buf77 = buf55; del buf55  # reuse
    # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (64, 512), (512, 1), 0), reinterpret_tensor(arg37_1, (512, 512), (1, 512), 0), out=buf77)
    del arg37_1
    buf78 = buf50; del buf50  # reuse
    # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (64, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 512), (1, 512), 0), out=buf78)
    del arg38_1
    buf79 = reinterpret_tensor(buf37, (8, 64, 64), (4096, 64, 1), 0); del buf37  # reuse
    # Topologically Sorted Source Nodes: [scores_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf77, (8, 64, 64), (64, 512, 1), 0), reinterpret_tensor(buf78, (8, 64, 64), (64, 1, 512), 0), out=buf79)
    buf80 = buf63; del buf63  # reuse
    buf82 = buf61; del buf61  # reuse
    buf84 = reinterpret_tensor(buf78, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf78  # reuse
    cpp_fused__softmax_add_21(buf79, buf5, buf80, buf82, buf84)
    buf83 = reinterpret_tensor(buf79, (64, 512), (512, 1), 0); del buf79  # reuse
    # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (64, 512), (512, 1), 0), reinterpret_tensor(arg39_1, (512, 512), (1, 512), 0), out=buf83)
    del arg39_1
    buf85 = reinterpret_tensor(buf76, (8, 64, 64), (4096, 64, 1), 0); del buf76  # reuse
    # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf84, (8, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf83, (8, 64, 64), (64, 512, 1), 0), out=buf85)
    buf86 = reinterpret_tensor(buf84, (1, 64, 8, 64), (32768, 512, 64, 1), 0); del buf84  # reuse
    cpp_fused_clone_22(buf85, buf86)
    buf87 = reinterpret_tensor(buf85, (64, 512), (512, 1), 0); del buf85  # reuse
    # Topologically Sorted Source Nodes: [attn_output_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf86, (64, 512), (512, 1), 0), reinterpret_tensor(arg40_1, (512, 512), (1, 512), 0), out=buf87)
    del arg40_1
    buf88 = buf75; del buf75  # reuse
    buf89 = reinterpret_tensor(buf86, (1, 64, 512), (32768, 512, 1), 0); del buf86  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_23(buf69, buf74, buf87, arg41_1, buf88, buf89)
    del arg41_1
    buf90 = reinterpret_tensor(buf73, (64, 2048), (2048, 1), 0); del buf73  # reuse
    # Topologically Sorted Source Nodes: [hidden_states_46], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf89, (64, 512), (512, 1), 0), reinterpret_tensor(arg42_1, (512, 2048), (1, 512), 0), out=buf90)
    del arg42_1
    buf91 = reinterpret_tensor(buf90, (1, 64, 2048), (131072, 2048, 1), 0); del buf90  # reuse
    cpp_fused_relu_24(buf91)
    buf92 = reinterpret_tensor(buf89, (64, 512), (512, 1), 0); del buf89  # reuse
    # Topologically Sorted Source Nodes: [hidden_states_49], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (64, 2048), (2048, 1), 0), reinterpret_tensor(arg43_1, (2048, 512), (1, 2048), 0), out=buf92)
    del arg43_1
    buf93 = buf88; del buf88  # reuse
    buf94 = reinterpret_tensor(buf83, (1, 64, 512), (32768, 512, 1), 0); del buf83  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_25(buf69, buf74, buf87, buf92, arg44_1, buf93, buf94)
    del arg44_1
    buf95 = buf77; del buf77  # reuse
    # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (64, 512), (512, 1), 0), reinterpret_tensor(arg45_1, (512, 512), (1, 512), 0), out=buf95)
    del arg45_1
    buf96 = buf64; del buf64  # reuse
    # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (64, 512), (512, 1), 0), reinterpret_tensor(arg46_1, (512, 512), (1, 512), 0), out=buf96)
    del arg46_1
    buf97 = reinterpret_tensor(buf58, (8, 64, 64), (4096, 64, 1), 0); del buf58  # reuse
    # Topologically Sorted Source Nodes: [scores_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf95, (8, 64, 64), (64, 512, 1), 0), reinterpret_tensor(buf96, (8, 64, 64), (64, 1, 512), 0), out=buf97)
    del buf95
    buf98 = buf82; del buf82  # reuse
    buf100 = buf80; del buf80  # reuse
    buf102 = reinterpret_tensor(buf96, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf96  # reuse
    cpp_fused__softmax_add_26(buf97, buf5, buf98, buf100, buf102)
    del buf100
    del buf5
    del buf98
    buf101 = reinterpret_tensor(buf97, (64, 512), (512, 1), 0); del buf97  # reuse
    # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (64, 512), (512, 1), 0), reinterpret_tensor(arg47_1, (512, 512), (1, 512), 0), out=buf101)
    del arg47_1
    buf103 = reinterpret_tensor(buf94, (8, 64, 64), (4096, 64, 1), 0); del buf94  # reuse
    # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf102, (8, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf101, (8, 64, 64), (64, 512, 1), 0), out=buf103)
    del buf101
    buf104 = reinterpret_tensor(buf102, (1, 64, 8, 64), (32768, 512, 64, 1), 0); del buf102  # reuse
    cpp_fused_clone_27(buf103, buf104)
    buf105 = reinterpret_tensor(buf103, (64, 512), (512, 1), 0); del buf103  # reuse
    # Topologically Sorted Source Nodes: [attn_output_11], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf104, (64, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 512), (1, 512), 0), out=buf105)
    del arg48_1
    buf106 = buf69; del buf69  # reuse
    buf107 = buf93; del buf93  # reuse
    buf108 = reinterpret_tensor(buf104, (1, 64, 512), (32768, 512, 1), 0); del buf104  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_28(buf106, buf74, buf87, buf92, buf105, arg49_1, buf107, buf108)
    del arg49_1
    del buf105
    del buf74
    del buf87
    del buf92
    buf109 = reinterpret_tensor(buf91, (64, 2048), (2048, 1), 0); del buf91  # reuse
    # Topologically Sorted Source Nodes: [hidden_states_56], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf108, (64, 512), (512, 1), 0), reinterpret_tensor(arg50_1, (512, 2048), (1, 512), 0), out=buf109)
    del arg50_1
    buf110 = reinterpret_tensor(buf109, (1, 64, 2048), (131072, 2048, 1), 0); del buf109  # reuse
    cpp_fused_relu_29(buf110)
    buf111 = reinterpret_tensor(buf108, (64, 512), (512, 1), 0); del buf108  # reuse
    # Topologically Sorted Source Nodes: [hidden_states_59], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf110, (64, 2048), (2048, 1), 0), reinterpret_tensor(arg51_1, (2048, 512), (1, 2048), 0), out=buf111)
    del arg51_1
    del buf110
    buf112 = buf107; del buf107  # reuse
    buf113 = buf106; del buf106  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_30(buf113, buf111, arg52_1, buf112)
    del arg52_1
    return (buf113, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 64), (64, 1), device='cpu', dtype=torch.int64)
    arg1_1 = rand_strided((32128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1, 64), (64, 1), device='cpu', dtype=torch.int64)
    arg3_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((32, 8), (8, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
