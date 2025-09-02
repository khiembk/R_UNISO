# AOT ID: ['5_inference']
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
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/ww/cww2krno77atvhnkweffxzgjvgge7qzsuv3sgfsnm4bhot2obctn.py
# Topologically Sorted Source Nodes: [dropout, pow_1, mean, add, rsqrt, mul_3, mul_4], Original ATen: [aten.clone, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   dropout => clone
#   mean => mean
#   mul_3 => mul_3
#   mul_4 => mul_4
#   pow_1 => pow_1
#   rsqrt => rsqrt
# Graph fragment:
#   %clone : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%arg0_1,), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%clone, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clone, %rsqrt), kwargs = {})
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, %mul_3), kwargs = {})
triton_per_fused_add_clone_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_per_fused_add_clone_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 385536}}
)
@triton.jit
def triton_per_fused_add_clone_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, out_ptr0, out_ptr2, xnumel, r0_numel):
    xnumel = 50
    XBLOCK: tl.constexpr = 1
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = 384.0
    tmp8 = (tmp5 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp0 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(out_ptr0 + (r0_1 + 384*x0), tmp0, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 384*x0), tmp13, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/n5/cn52fcjj2m6nnbikgglz6cqq3bmkjztro6kkgkbw3gsnvoaa7fg3.py
# Topologically Sorted Source Nodes: [mul, to_1, sub, mul_1], Original ATen: [aten.mul, aten._to_copy, aten.rsub]
# Source node to ATen node mapping:
#   mul => mul
#   mul_1 => mul_1
#   sub => sub
#   to_1 => convert_element_type_1
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_4, %unsqueeze_6), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %convert_element_type_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, -3.4028234663852886e+38), kwargs = {})
triton_poi_fused__to_copy_mul_rsub_1 = async_compile.triton('triton_poi_fused__to_copy_mul_rsub_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_rsub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_rsub_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2500
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 50)
    x1 = xindex // 50
    x2 = xindex
    tmp4 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 <= tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = -3.4028234663852886e+38
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/ps/cps3dy6b4gjssvefnh3qbsvakocez5se4m7l5z6y36r6gkc6n3bz.py
# Topologically Sorted Source Nodes: [to_2, sub_1, mul_2], Original ATen: [aten._to_copy, aten.rsub, aten.mul]
# Source node to ATen node mapping:
#   mul_2 => mul_2
#   sub_1 => sub_1
#   to_2 => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_8, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %convert_element_type_2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, -3.4028234663852886e+38), kwargs = {})
triton_poi_fused__to_copy_mul_rsub_2 = async_compile.triton('triton_poi_fused__to_copy_mul_rsub_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_rsub_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_mul_rsub_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = -3.4028234663852886e+38
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 50, 384), (19200, 384, 1))
    assert_size_stride(arg1_1, (1, 50), (50, 1))
    assert_size_stride(arg2_1, (1, 50), (50, 1))
    assert_size_stride(arg3_1, (384, ), (1, ))
    assert_size_stride(arg4_1, (256, 384), (384, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dropout, pow_1, mean, add, rsqrt, mul_3, mul_4], Original ATen: [aten.clone, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_mean_mul_pow_rsqrt_0.run(arg0_1, arg3_1, buf0, buf2, 50, 384, stream=stream0)
        del arg0_1
        del arg3_1
        buf3 = empty_strided_cuda((50, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (50, 384), (384, 1), 0), reinterpret_tensor(arg4_1, (384, 256), (1, 384), 0), out=buf3)
        del arg4_1
        buf4 = empty_strided_cuda((1, 1, 50, 50), (2500, 2500, 50, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, to_1, sub, mul_1], Original ATen: [aten.mul, aten._to_copy, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_rsub_1.run(arg1_1, buf4, 2500, stream=stream0)
        del arg1_1
        buf5 = empty_strided_cuda((1, 1, 1, 50), (50, 50, 50, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to_2, sub_1, mul_2], Original ATen: [aten._to_copy, aten.rsub, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_rsub_2.run(arg2_1, buf5, 50, stream=stream0)
        del arg2_1
    return (buf2, reinterpret_tensor(buf3, (1, 8, 50, 32), (12800, 32, 256, 1), 0), buf4, buf0, buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 50, 384), (19200, 384, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 50), (50, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((1, 50), (50, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
