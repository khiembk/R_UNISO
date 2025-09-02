# AOT ID: ['10_inference']
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/il/cildhqyjewa3xn63csmsjtiou7vzexocpmii2lz6t45ysjn3a5bp.py
# Topologically Sorted Source Nodes: [add, pow_1, mean, add_1, rsqrt, mul, mul_1], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   mean => mean
#   mul => mul
#   mul_1 => mul_1
#   pow_1 => pow_1
#   rsqrt => rsqrt
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %view_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, %mul), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 526336}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 512*x0), None)
    tmp7 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp8 = 512.0
    tmp9 = (tmp6 / tmp8)
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp2 * tmp12
    tmp14 = tmp7 * tmp13
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp14, None)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/th/cthmbw3xlb7fnle57qziutxvx6g7ycpx4audmpxgju7pdm5r7tpa.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_3,), kwargs = {})
triton_poi_fused_relu_1 = async_compile.triton('triton_poi_fused_relu_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1572864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/m6/cm6ncetewmgk5qsolkak7uceryayez6utaznewcxtvw3qmueefb4.py
# Topologically Sorted Source Nodes: [add, add_2, pow_2, mean_1, add_3, rsqrt_1, mul_2, mul_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_2 => add_2
#   add_3 => add_3
#   mean_1 => mean_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %view_1), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_5), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_2, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg6_1, %mul_2), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_2 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 657408}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 512*x0), None)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 512*x0), None)
    tmp9 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [R0_BLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = 512.0
    tmp11 = (tmp8 / tmp10)
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp4 * tmp14
    tmp16 = tmp9 * tmp15
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp16, None)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/bw/cbwaasw7yvayosz5t6jgzhxzzv66vphvv74ctzt7e3avy5rwzfmh.py
# Topologically Sorted Source Nodes: [iadd], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   iadd => add_4
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %arg10_1), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_4, -1), kwargs = {})
triton_per_fused_add_3 = async_compile.triton('triton_per_fused_add_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'y': 8, 'x': 64, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 131072, 'x': 8192, 'r0_': 131072}}
)
@triton.jit
def triton_per_fused_add_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8
    xnumel = 64
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, None, :]
    r0_offset = 0
    r0_mask = tl.full([YBLOCK, XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 64*x1 + 4096*y0), xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (y0 + 8*r0_2 + 512*x1), xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [YBLOCK, XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [YBLOCK, XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask & ymask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 2)[:, :, None]
    tmp9 = tmp3 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [YBLOCK, XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask & ymask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 2)[:, :, None]
    tl.store(out_ptr0 + (x1 + 64*y0), tmp8, xmask & ymask)
    tl.store(out_ptr1 + (x1 + 64*y0), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/7f/c7fydtvgc2ryykzkj4wnk3cmbeju5cyfoey5m4w2sv6riksngfma.py
# Topologically Sorted Source Nodes: [iadd, softmax], Original ATen: [aten.add, aten._softmax]
# Source node to ATen node mapping:
#   iadd => add_4
#   softmax => div
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %arg10_1), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_2), kwargs = {})
#   %exp_default_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %getitem_3), kwargs = {})
triton_poi_fused__softmax_add_4 = async_compile.triton('triton_poi_fused__softmax_add_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'z': 8, 'y': 64, 'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'znumel': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'ZBLOCK': 'constexpr', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid3D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'z': 131072, 'y': 4096, 'x': 393216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, znumel, ynumel, xnumel, ZBLOCK : tl.constexpr, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    znumel = 8
    ynumel = 64
    xnumel = 64
    zoffset = tl.program_id(2) * ZBLOCK
    zindex = zoffset + tl.arange(0, ZBLOCK)[:, None, None]
    zmask = zindex < znumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = yindex
    z0 = zindex
    tmp0 = tl.load(in_out_ptr0 + (x2 + 64*y1 + 4096*z0), xmask & ymask & zmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (z0 + 8*x2 + 512*y1), xmask & ymask & zmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y1 + 64*z0), ymask & zmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (y1 + 64*z0), ymask & zmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp7 = (tmp5 / tmp6)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 64*y1 + 4096*z0), tmp7, xmask & ymask & zmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/3b/c3biuhc432rtj7twxpmhddkdea6k2suwvn5nzcygmtwyom4wqd2v.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 8)
    x2 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 4096*x1), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/wd/cwdhcmrlnsbjtobzmwwysl4wtplzsnpz5uqtgductf72pfm5ppxz.py
# Topologically Sorted Source Nodes: [add, add_2, add_4, pow_3, mean_2, add_5, rsqrt_2, mul_4, mul_5], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_2 => add_2
#   add_4 => add_5
#   add_5 => add_6
#   mean_2 => mean_2
#   mul_4 => mul_4
#   mul_5 => mul_5
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %view_1), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_5), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_25), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_5, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %rsqrt_2), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg12_1, %mul_4), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_6 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 788480}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 512*x0), None)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 512*x0), None)
    tmp5 = tl.load(in_ptr3 + (r0_1 + 512*x0), None)
    tmp11 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [R0_BLOCK])
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp12 = 512.0
    tmp13 = (tmp10 / tmp12)
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp6 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp18, None)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/n6/cn64ooodypc2sqdcjoreugxb2bxlaoas7xzwknk5tv27weteroqm.py
# Topologically Sorted Source Nodes: [add, add_2, add_4, add_6, pow_4, mean_3, add_7, rsqrt_3, mul_6, mul_7], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_2 => add_2
#   add_4 => add_5
#   add_6 => add_7
#   add_7 => add_8
#   mean_3 => mean_3
#   mul_6 => mul_6
#   mul_7 => mul_7
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %view_1), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_5), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_25), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_29), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_7, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, %rsqrt_3), kwargs = {})
#   %mul_7 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg15_1, %mul_6), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 1181696}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 512*x0), None)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 512*x0), None)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 512*x0), None)
    tmp13 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp14 = 512.0
    tmp15 = (tmp12 / tmp14)
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp8 * tmp18
    tmp20 = tmp13 * tmp19
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp8, None)
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp20, None)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/4s/c4s4ferwvah5xaubhxer33tjb3czudx2i7v6g2dylzbepknt4knz.py
# Topologically Sorted Source Nodes: [add_8, pow_5, mean_4, add_9, rsqrt_4, mul_8, mul_9], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_8 => add_10
#   add_9 => add_11
#   mean_4 => mean_4
#   mul_8 => mul_8
#   mul_9 => mul_9
#   pow_5 => pow_5
#   rsqrt_4 => rsqrt_4
# Graph fragment:
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %view_49), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_10, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %rsqrt_4), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg20_1, %mul_8), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_8 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 788480}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_8(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 512*x0), None)
    tmp7 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp8 = 512.0
    tmp9 = (tmp6 / tmp8)
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp2 * tmp12
    tmp14 = tmp7 * tmp13
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp2, None)
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp14, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 64, 512), (32768, 512, 1))
    assert_size_stride(arg1_1, (512, 512), (512, 1))
    assert_size_stride(arg2_1, (1, 64, 512), (32768, 512, 1))
    assert_size_stride(arg3_1, (512, ), (1, ))
    assert_size_stride(arg4_1, (2048, 512), (512, 1))
    assert_size_stride(arg5_1, (512, 2048), (2048, 1))
    assert_size_stride(arg6_1, (512, ), (1, ))
    assert_size_stride(arg7_1, (512, 512), (512, 1))
    assert_size_stride(arg8_1, (512, 512), (512, 1))
    assert_size_stride(arg9_1, (512, 512), (512, 1))
    assert_size_stride(arg10_1, (1, 8, 64, 64), (8, 1, 512, 8))
    assert_size_stride(arg11_1, (512, 512), (512, 1))
    assert_size_stride(arg12_1, (512, ), (1, ))
    assert_size_stride(arg13_1, (2048, 512), (512, 1))
    assert_size_stride(arg14_1, (512, 2048), (2048, 1))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (512, 512), (512, 1))
    assert_size_stride(arg17_1, (512, 512), (512, 1))
    assert_size_stride(arg18_1, (512, 512), (512, 1))
    assert_size_stride(arg19_1, (512, 512), (512, 1))
    assert_size_stride(arg20_1, (512, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg0_1, (64, 512), (512, 1), 0), reinterpret_tensor(arg1_1, (512, 512), (1, 512), 0), out=buf0)
        del arg0_1
        del arg1_1
        buf2 = empty_strided_cuda((1, 64, 512), (32768, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, pow_1, mean, add_1, rsqrt, mul, mul_1], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_0.run(arg2_1, buf0, arg3_1, buf2, 64, 512, stream=stream0)
        del arg3_1
        buf3 = empty_strided_cuda((64, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (64, 512), (512, 1), 0), reinterpret_tensor(arg4_1, (512, 2048), (1, 512), 0), out=buf3)
        del arg4_1
        buf4 = reinterpret_tensor(buf3, (1, 64, 2048), (131072, 2048, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf4, 131072, stream=stream0)
        buf5 = reinterpret_tensor(buf2, (64, 512), (512, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (64, 2048), (2048, 1), 0), reinterpret_tensor(arg5_1, (2048, 512), (1, 2048), 0), out=buf5)
        del arg5_1
        buf7 = empty_strided_cuda((1, 64, 512), (32768, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, add_2, pow_2, mean_1, add_3, rsqrt_1, mul_2, mul_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_2.run(arg2_1, buf0, buf5, arg6_1, buf7, 64, 512, stream=stream0)
        del arg6_1
        buf8 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (64, 512), (512, 1), 0), reinterpret_tensor(arg7_1, (512, 512), (1, 512), 0), out=buf8)
        del arg7_1
        buf9 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (64, 512), (512, 1), 0), reinterpret_tensor(arg8_1, (512, 512), (1, 512), 0), out=buf9)
        del arg8_1
        buf10 = empty_strided_cuda((8, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (8, 64, 64), (64, 512, 1), 0), reinterpret_tensor(buf9, (8, 64, 64), (64, 1, 512), 0), out=buf10)
        del buf8
        buf11 = empty_strided_cuda((1, 8, 64, 1), (512, 64, 1, 512), torch.float32)
        buf12 = empty_strided_cuda((1, 8, 64, 1), (512, 64, 1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [iadd], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_3.run(buf10, arg10_1, buf11, buf12, 8, 64, 64, stream=stream0)
        buf13 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (64, 512), (512, 1), 0), reinterpret_tensor(arg9_1, (512, 512), (1, 512), 0), out=buf13)
        del arg9_1
        buf14 = reinterpret_tensor(buf10, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [iadd, softmax], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_4.run(buf14, arg10_1, buf11, buf12, 8, 64, 64, stream=stream0)
        buf15 = reinterpret_tensor(buf7, (8, 64, 64), (4096, 64, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf14, (8, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf13, (8, 64, 64), (64, 512, 1), 0), out=buf15)
        buf16 = reinterpret_tensor(buf14, (1, 64, 8, 64), (32768, 512, 64, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf15, buf16, 32768, stream=stream0)
        buf17 = reinterpret_tensor(buf15, (64, 512), (512, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (64, 512), (512, 1), 0), reinterpret_tensor(arg11_1, (512, 512), (1, 512), 0), out=buf17)
        del arg11_1
        buf19 = reinterpret_tensor(buf16, (1, 64, 512), (32768, 512, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [add, add_2, add_4, pow_3, mean_2, add_5, rsqrt_2, mul_4, mul_5], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_6.run(arg2_1, buf0, buf5, buf17, arg12_1, buf19, 64, 512, stream=stream0)
        del arg12_1
        buf20 = reinterpret_tensor(buf4, (64, 2048), (2048, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (64, 512), (512, 1), 0), reinterpret_tensor(arg13_1, (512, 2048), (1, 512), 0), out=buf20)
        del arg13_1
        buf21 = reinterpret_tensor(buf20, (1, 64, 2048), (131072, 2048, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [relu_1], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf21, 131072, stream=stream0)
        buf22 = reinterpret_tensor(buf19, (64, 512), (512, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (64, 2048), (2048, 1), 0), reinterpret_tensor(arg14_1, (2048, 512), (1, 2048), 0), out=buf22)
        del arg14_1
        del buf21
        buf23 = reinterpret_tensor(buf0, (1, 64, 512), (32768, 512, 1), 0); del buf0  # reuse
        buf25 = reinterpret_tensor(buf13, (1, 64, 512), (32768, 512, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [add, add_2, add_4, add_6, pow_4, mean_3, add_7, rsqrt_3, mul_6, mul_7], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_7.run(buf23, arg2_1, buf5, buf17, buf22, arg15_1, buf25, 64, 512, stream=stream0)
        del arg15_1
        del arg2_1
        buf26 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (64, 512), (512, 1), 0), reinterpret_tensor(arg16_1, (512, 512), (1, 512), 0), out=buf26)
        del arg16_1
        buf27 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (64, 512), (512, 1), 0), reinterpret_tensor(arg17_1, (512, 512), (1, 512), 0), out=buf27)
        del arg17_1
        buf28 = reinterpret_tensor(buf17, (8, 64, 64), (4096, 64, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf26, (8, 64, 64), (64, 512, 1), 0), reinterpret_tensor(buf27, (8, 64, 64), (64, 1, 512), 0), out=buf28)
        del buf26
        buf29 = buf12; del buf12  # reuse
        buf30 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [iadd_1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_3.run(buf28, arg10_1, buf29, buf30, 8, 64, 64, stream=stream0)
        buf31 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (64, 512), (512, 1), 0), reinterpret_tensor(arg18_1, (512, 512), (1, 512), 0), out=buf31)
        del arg18_1
        buf32 = reinterpret_tensor(buf28, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [iadd_1, softmax_1], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_4.run(buf32, arg10_1, buf29, buf30, 8, 64, 64, stream=stream0)
        del arg10_1
        del buf29
        del buf30
        buf33 = reinterpret_tensor(buf25, (8, 64, 64), (4096, 64, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf32, (8, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf31, (8, 64, 64), (64, 512, 1), 0), out=buf33)
        del buf31
        buf34 = reinterpret_tensor(buf32, (1, 64, 8, 64), (32768, 512, 64, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf33, buf34, 32768, stream=stream0)
        buf35 = reinterpret_tensor(buf33, (64, 512), (512, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (64, 512), (512, 1), 0), reinterpret_tensor(arg19_1, (512, 512), (1, 512), 0), out=buf35)
        del arg19_1
        buf36 = buf23; del buf23  # reuse
        buf38 = reinterpret_tensor(buf34, (1, 64, 512), (32768, 512, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [add_8, pow_5, mean_4, add_9, rsqrt_4, mul_8, mul_9], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_8.run(buf36, buf35, arg20_1, buf38, 64, 512, stream=stream0)
        del arg20_1
        del buf35
    return (buf38, buf36, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 64, 512), (32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 64, 512), (32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1, 8, 64, 64), (8, 1, 512, 8), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
