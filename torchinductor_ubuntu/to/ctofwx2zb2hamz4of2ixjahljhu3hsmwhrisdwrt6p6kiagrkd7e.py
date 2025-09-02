# AOT ID: ['6_inference']
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/gs/cgscz7snavm33fdoafcgbjt7z3jguuay3yszv3klt6rj5gv776id.py
# Topologically Sorted Source Nodes: [add_2, iadd, softmax], Original ATen: [aten.add, aten._softmax]
# Source node to ATen node mapping:
#   add_2 => add_2
#   iadd => add_3
#   softmax => div_2
# Graph fragment:
#   %add_2 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_2, %arg5_1), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_8, %add_2), kwargs = {})
#   %prepare_softmax_online_default_11 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_3, -1), kwargs = {})
#   %sub_tensor_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_22), kwargs = {})
#   %exp_default_11 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_11,), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_11, %getitem_23), kwargs = {})
triton_per_fused__softmax_add_0 = async_compile.triton('triton_per_fused__softmax_add_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 400
    r0_numel = 50
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 50)
    x1 = xindex // 50
    tmp0 = tl.load(in_ptr0 + (r0_2 + 50*x3), r0_mask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr2 + (r0_2 + 50*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = (-1)*((0) * ((0) <= (r0_2 + ((-1)*x0))) + (r0_2 + ((-1)*x0)) * ((r0_2 + ((-1)*x0)) < (0)))
    tmp2 = tl.full([1, 1], 16, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tmp1.to(tl.float32)
    tmp5 = 0.0625
    tmp6 = tmp4 * tmp5
    tmp7 = tl_math.log(tmp6)
    tmp8 = 0.48089834696298783
    tmp9 = tmp7 * tmp8
    tmp10 = 16.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.int64)
    tmp13 = tmp12 + tmp2
    tmp14 = tl.full([1, 1], 31, tl.int64)
    tmp15 = triton_helpers.minimum(tmp13, tmp14)
    tmp16 = tl.where(tmp3, tmp1, tmp15)
    tmp17 = tl.full([1, 1], 0, tl.int64)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([XBLOCK, R0_BLOCK], 32, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert(((0 <= tmp22) & (tmp22 < 32)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp22 < 32")
    tmp24 = tl.load(in_ptr1 + (x1 + 8*tmp22), r0_mask & xmask, eviction_policy='evict_last')
    tmp26 = tmp24 + tmp25
    tmp27 = tmp0 + tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
    tmp32 = tl.where(r0_mask & xmask, tmp30, float("-inf"))
    tmp33 = triton_helpers.max2(tmp32, 1)[:, None]
    tmp34 = tmp28 - tmp33
    tmp35 = tl_math.exp(tmp34)
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, R0_BLOCK])
    tmp38 = tl.where(r0_mask & xmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None]
    tmp40 = tmp27 - tmp33
    tmp41 = tl_math.exp(tmp40)
    tmp42 = (tmp41 / tmp39)
    tl.store(out_ptr2 + (r0_2 + 50*x0 + 2528*x1), tmp42, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/yj/cyji2sorngyqhsgfvf22beipm2nswxsff37x74573nql44rdncd4.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_6,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 153600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 8)
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 1600*x1), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/pb/cpb2akpv43fq2mp5cna45tzor23ja647xkhkuhuy3odems2suvyb.py
# Topologically Sorted Source Nodes: [add_3, pow_1, mean, add_4, rsqrt, mul_1, mul_2], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_3 => add_4
#   add_4 => add_5
#   mean => mean
#   mul_1 => mul_1
#   mul_2 => mul_2
#   pow_1 => pow_1
#   rsqrt => rsqrt
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg7_1, %view_16), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_4, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg8_1, %mul_1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 308736}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_2(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel):
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
    tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = 384.0
    tmp10 = (tmp7 / tmp9)
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp2 * tmp13
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp15, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/s7/cs72dqgrz6wtxvrvlzsnrjvomcdgyurqcedygwq72a5yyfdt6ycv.py
# Topologically Sorted Source Nodes: [zeros, add_5, iadd_1, softmax_1], Original ATen: [aten.zeros, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   add_5 => add_6
#   iadd_1 => add_7
#   softmax_1 => div_3
#   zeros => full_default_2
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 8, 50, 50], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_6 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default_2, %arg13_1), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_28, %add_6), kwargs = {})
#   %prepare_softmax_online_default_10 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_7, -1), kwargs = {})
#   %sub_tensor_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_20), kwargs = {})
#   %exp_default_10 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_10,), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_10, %getitem_21), kwargs = {})
triton_per_fused__softmax_add_zeros_3 = async_compile.triton('triton_per_fused__softmax_add_zeros_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_zeros_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 240200}}
)
@triton.jit
def triton_per_fused__softmax_add_zeros_3(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 400
    r0_numel = 50
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x2 = (xindex % 50)
    x3 = xindex // 50
    tmp0 = tl.load(in_ptr0 + (r0_1 + 50*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = 0.0
    tmp3 = tmp2 + tmp1
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask & xmask, tmp7, float("-inf"))
    tmp10 = triton_helpers.max2(tmp9, 1)[:, None]
    tmp11 = tmp5 - tmp10
    tmp12 = tl_math.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp4 - tmp10
    tmp18 = tl_math.exp(tmp17)
    tmp19 = (tmp18 / tmp16)
    tl.store(out_ptr2 + (r0_1 + 50*x2 + 2528*x3), tmp19, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/yj/cyj5oncudbbs3vs5cmtgfrw4k7nyal5kytujjp6zfcy33j5i3iae.py
# Topologically Sorted Source Nodes: [add_3, add_6, pow_2, mean_1, add_7, rsqrt_1, mul_3, mul_4], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_3 => add_4
#   add_6 => add_8
#   add_7 => add_9
#   mean_1 => mean_1
#   mul_3 => mul_3
#   mul_4 => mul_4
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg7_1, %view_16), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_36), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_8, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %rsqrt_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg15_1, %mul_3), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_4 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 385536}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel):
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
    tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [R0_BLOCK])
    tmp8 = tl.where(r0_mask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = 384.0
    tmp12 = (tmp9 / tmp11)
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp4 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp17, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/4m/c4mmm5mwva4sx3z7xo5nnpbxxq6ett42guri7re7xg6uk6vsdc6c.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_38,), kwargs = {})
triton_poi_fused_relu_5 = async_compile.triton('triton_poi_fused_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 307200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_5(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/y2/cy2xl37rph3vwojw7mg3c2eblyaop4ej2czjobs7sfuen6b5pvho.py
# Topologically Sorted Source Nodes: [add_3, add_6, add_8, pow_3, mean_2, add_9, rsqrt_2, mul_5, mul_6], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_3 => add_4
#   add_6 => add_8
#   add_8 => add_10
#   add_9 => add_11
#   mean_2 => mean_2
#   mul_5 => mul_5
#   mul_6 => mul_6
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg7_1, %view_16), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_36), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_40), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_10, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %rsqrt_2), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg18_1, %mul_5), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 462336}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
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
    tmp1 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp12 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = 384.0
    tmp14 = (tmp11 / tmp13)
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp6 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp19, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/3l/c3lpv4r3xx4xkmxbtmbk4aan2zq2tlmxkm62wknizexlkhh5zmal.py
# Topologically Sorted Source Nodes: [add_3, add_6, add_8, add_10, pow_4, mean_3, add_11, rsqrt_3, mul_7, mul_8], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_10 => add_13
#   add_11 => add_14
#   add_3 => add_4
#   add_6 => add_8
#   add_8 => add_10
#   mean_3 => mean_3
#   mul_7 => mul_7
#   mul_8 => mul_8
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg7_1, %view_16), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_36), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_40), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_60), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_13, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %rsqrt_3), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg23_1, %mul_7), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 692736}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
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
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 384.0
    tmp16 = (tmp13 / tmp15)
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp8 * tmp19
    tmp21 = tmp14 * tmp20
    tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp8, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp21, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/iw/ciw55q4u2tuyk64gfpup2zz7tc3rjr5eosfzk5llxfmoahpxoekm.py
# Topologically Sorted Source Nodes: [add_12, add_14, add_16, add_18, pow_8, mean_7, add_19, rsqrt_7, mul_15, mul_16], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_12 => add_16
#   add_14 => add_18
#   add_16 => add_21
#   add_18 => add_24
#   add_19 => add_25
#   mean_7 => mean_7
#   mul_15 => mul_15
#   mul_16 => mul_16
#   pow_8 => pow_8
#   rsqrt_7 => rsqrt_7
# Graph fragment:
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_80), kwargs = {})
#   %add_18 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_84), kwargs = {})
#   %add_21 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %view_104), kwargs = {})
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_124), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_24, 2), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [-1], True), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_7, 1e-06), kwargs = {})
#   %rsqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %rsqrt_7), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg41_1, %mul_15), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 692736}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 384.0
    tmp16 = (tmp13 / tmp15)
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp8 * tmp19
    tmp21 = tmp14 * tmp20
    tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp8, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp21, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/fa/cfaqlg4gx7wvw6mo6kvkdn74n3babdcj75mjuvwai3vh2acvw6eg.py
# Topologically Sorted Source Nodes: [add_36, pow_17, mean_16, add_37, rsqrt_16, mul_33, mul_34], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_36 => add_48
#   add_37 => add_49
#   mean_16 => mean_16
#   mul_33 => mul_33
#   mul_34 => mul_34
#   pow_17 => pow_17
#   rsqrt_16 => rsqrt_16
# Graph fragment:
#   %add_48 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %view_256), kwargs = {})
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_48, 2), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_17, [-1], True), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_16, 1e-06), kwargs = {})
#   %rsqrt_16 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_49,), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %rsqrt_16), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg80_1, %mul_33), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 462336}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_9(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = 384.0
    tmp10 = (tmp7 / tmp9)
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp2 * tmp13
    tmp15 = tmp8 * tmp14
    tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp2, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp15, r0_mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 50, 384), (19200, 384, 1))
    assert_size_stride(arg1_1, (256, 384), (384, 1))
    assert_size_stride(arg2_1, (256, 384), (384, 1))
    assert_size_stride(arg3_1, (1, 8, 50, 32), (12800, 32, 256, 1))
    assert_size_stride(arg4_1, (32, 8), (8, 1))
    assert_size_stride(arg5_1, (1, 1, 50, 50), (2500, 2500, 50, 1))
    assert_size_stride(arg6_1, (384, 256), (256, 1))
    assert_size_stride(arg7_1, (1, 50, 384), (19200, 384, 1))
    assert_size_stride(arg8_1, (384, ), (1, ))
    assert_size_stride(arg9_1, (256, 384), (384, 1))
    assert_size_stride(arg10_1, (1, 50, 384), (19200, 384, 1))
    assert_size_stride(arg11_1, (256, 384), (384, 1))
    assert_size_stride(arg12_1, (256, 384), (384, 1))
    assert_size_stride(arg13_1, (1, 1, 1, 50), (50, 50, 50, 1))
    assert_size_stride(arg14_1, (384, 256), (256, 1))
    assert_size_stride(arg15_1, (384, ), (1, ))
    assert_size_stride(arg16_1, (512, 384), (384, 1))
    assert_size_stride(arg17_1, (384, 512), (512, 1))
    assert_size_stride(arg18_1, (384, ), (1, ))
    assert_size_stride(arg19_1, (256, 384), (384, 1))
    assert_size_stride(arg20_1, (256, 384), (384, 1))
    assert_size_stride(arg21_1, (256, 384), (384, 1))
    assert_size_stride(arg22_1, (384, 256), (256, 1))
    assert_size_stride(arg23_1, (384, ), (1, ))
    assert_size_stride(arg24_1, (256, 384), (384, 1))
    assert_size_stride(arg25_1, (256, 384), (384, 1))
    assert_size_stride(arg26_1, (256, 384), (384, 1))
    assert_size_stride(arg27_1, (384, 256), (256, 1))
    assert_size_stride(arg28_1, (384, ), (1, ))
    assert_size_stride(arg29_1, (512, 384), (384, 1))
    assert_size_stride(arg30_1, (384, 512), (512, 1))
    assert_size_stride(arg31_1, (384, ), (1, ))
    assert_size_stride(arg32_1, (256, 384), (384, 1))
    assert_size_stride(arg33_1, (256, 384), (384, 1))
    assert_size_stride(arg34_1, (256, 384), (384, 1))
    assert_size_stride(arg35_1, (384, 256), (256, 1))
    assert_size_stride(arg36_1, (384, ), (1, ))
    assert_size_stride(arg37_1, (256, 384), (384, 1))
    assert_size_stride(arg38_1, (256, 384), (384, 1))
    assert_size_stride(arg39_1, (256, 384), (384, 1))
    assert_size_stride(arg40_1, (384, 256), (256, 1))
    assert_size_stride(arg41_1, (384, ), (1, ))
    assert_size_stride(arg42_1, (512, 384), (384, 1))
    assert_size_stride(arg43_1, (384, 512), (512, 1))
    assert_size_stride(arg44_1, (384, ), (1, ))
    assert_size_stride(arg45_1, (256, 384), (384, 1))
    assert_size_stride(arg46_1, (256, 384), (384, 1))
    assert_size_stride(arg47_1, (256, 384), (384, 1))
    assert_size_stride(arg48_1, (384, 256), (256, 1))
    assert_size_stride(arg49_1, (384, ), (1, ))
    assert_size_stride(arg50_1, (256, 384), (384, 1))
    assert_size_stride(arg51_1, (256, 384), (384, 1))
    assert_size_stride(arg52_1, (256, 384), (384, 1))
    assert_size_stride(arg53_1, (384, 256), (256, 1))
    assert_size_stride(arg54_1, (384, ), (1, ))
    assert_size_stride(arg55_1, (512, 384), (384, 1))
    assert_size_stride(arg56_1, (384, 512), (512, 1))
    assert_size_stride(arg57_1, (384, ), (1, ))
    assert_size_stride(arg58_1, (256, 384), (384, 1))
    assert_size_stride(arg59_1, (256, 384), (384, 1))
    assert_size_stride(arg60_1, (256, 384), (384, 1))
    assert_size_stride(arg61_1, (384, 256), (256, 1))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (256, 384), (384, 1))
    assert_size_stride(arg64_1, (256, 384), (384, 1))
    assert_size_stride(arg65_1, (256, 384), (384, 1))
    assert_size_stride(arg66_1, (384, 256), (256, 1))
    assert_size_stride(arg67_1, (384, ), (1, ))
    assert_size_stride(arg68_1, (512, 384), (384, 1))
    assert_size_stride(arg69_1, (384, 512), (512, 1))
    assert_size_stride(arg70_1, (384, ), (1, ))
    assert_size_stride(arg71_1, (256, 384), (384, 1))
    assert_size_stride(arg72_1, (256, 384), (384, 1))
    assert_size_stride(arg73_1, (256, 384), (384, 1))
    assert_size_stride(arg74_1, (384, 256), (256, 1))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (256, 384), (384, 1))
    assert_size_stride(arg77_1, (256, 384), (384, 1))
    assert_size_stride(arg78_1, (256, 384), (384, 1))
    assert_size_stride(arg79_1, (384, 256), (256, 1))
    assert_size_stride(arg80_1, (384, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((50, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg0_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg1_1, (384, 256), (1, 384), 0), out=buf0)
        del arg1_1
        buf1 = empty_strided_cuda((8, 50, 50), (2500, 50, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg3_1, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf0, (8, 32, 50), (32, 1, 256), 0), out=buf1)
        del arg3_1
        buf4 = empty_strided_cuda((1, 8, 50, 50), (20224, 2528, 50, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_2, iadd, softmax], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_0.run(buf1, arg4_1, arg5_1, buf4, 400, 50, stream=stream0)
        buf5 = empty_strided_cuda((50, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg0_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg2_1, (384, 256), (1, 384), 0), out=buf5)
        del arg0_1
        del arg2_1
        buf6 = empty_strided_cuda((8, 50, 32), (1600, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf5, (8, 50, 32), (32, 256, 1), 0), out=buf6)
        buf7 = empty_strided_cuda((1, 50, 8, 32), (12800, 256, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf6, buf7, 12800, stream=stream0)
        buf8 = empty_strided_cuda((50, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (50, 256), (256, 1), 0), reinterpret_tensor(arg6_1, (256, 384), (1, 256), 0), out=buf8)
        del arg6_1
        buf10 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_3, pow_1, mean, add_4, rsqrt, mul_1, mul_2], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_2.run(arg7_1, buf8, arg8_1, buf10, 50, 384, stream=stream0)
        del arg8_1
        buf11 = reinterpret_tensor(buf7, (50, 256), (256, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (50, 384), (384, 1), 0), reinterpret_tensor(arg9_1, (384, 256), (1, 384), 0), out=buf11)
        del arg9_1
        buf12 = reinterpret_tensor(buf6, (50, 256), (256, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg10_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg11_1, (384, 256), (1, 384), 0), out=buf12)
        del arg11_1
        buf13 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf12, (8, 32, 50), (32, 1, 256), 0), out=buf13)
        buf17 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [zeros, add_5, iadd_1, softmax_1], Original ATen: [aten.zeros, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_zeros_3.run(buf13, arg13_1, buf17, 400, 50, stream=stream0)
        buf16 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg10_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg12_1, (384, 256), (1, 384), 0), out=buf16)
        del arg12_1
        buf18 = empty_strided_cuda((8, 50, 32), (1600, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf17, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf16, (8, 50, 32), (32, 256, 1), 0), out=buf18)
        buf19 = empty_strided_cuda((1, 50, 8, 32), (12800, 256, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf18, buf19, 12800, stream=stream0)
        buf20 = reinterpret_tensor(buf10, (50, 384), (384, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (50, 256), (256, 1), 0), reinterpret_tensor(arg14_1, (256, 384), (1, 256), 0), out=buf20)
        del arg14_1
        buf22 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_3, add_6, pow_2, mean_1, add_7, rsqrt_1, mul_3, mul_4], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_4.run(arg7_1, buf8, buf20, arg15_1, buf22, 50, 384, stream=stream0)
        del arg15_1
        buf23 = empty_strided_cuda((50, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (50, 384), (384, 1), 0), reinterpret_tensor(arg16_1, (384, 512), (1, 384), 0), out=buf23)
        del arg16_1
        buf24 = reinterpret_tensor(buf23, (1, 50, 512), (25600, 512, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf24, 25600, stream=stream0)
        buf25 = reinterpret_tensor(buf22, (50, 384), (384, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (50, 512), (512, 1), 0), reinterpret_tensor(arg17_1, (512, 384), (1, 512), 0), out=buf25)
        del arg17_1
        buf27 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_3, add_6, add_8, pow_3, mean_2, add_9, rsqrt_2, mul_5, mul_6], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_6.run(arg7_1, buf8, buf20, buf25, arg18_1, buf27, 50, 384, stream=stream0)
        del arg18_1
        buf28 = reinterpret_tensor(buf19, (50, 256), (256, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (50, 384), (384, 1), 0), reinterpret_tensor(arg19_1, (384, 256), (1, 384), 0), out=buf28)
        del arg19_1
        buf29 = reinterpret_tensor(buf18, (50, 256), (256, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (50, 384), (384, 1), 0), reinterpret_tensor(arg20_1, (384, 256), (1, 384), 0), out=buf29)
        del arg20_1
        buf30 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf29, (8, 32, 50), (32, 1, 256), 0), out=buf30)
        buf33 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [add_2, iadd_2, softmax_2], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_0.run(buf30, arg4_1, arg5_1, buf33, 400, 50, stream=stream0)
        buf34 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (50, 384), (384, 1), 0), reinterpret_tensor(arg21_1, (384, 256), (1, 384), 0), out=buf34)
        del arg21_1
        buf35 = empty_strided_cuda((8, 50, 32), (1600, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf34, (8, 50, 32), (32, 256, 1), 0), out=buf35)
        buf36 = empty_strided_cuda((1, 50, 8, 32), (12800, 256, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf35, buf36, 12800, stream=stream0)
        buf37 = reinterpret_tensor(buf27, (50, 384), (384, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (50, 256), (256, 1), 0), reinterpret_tensor(arg22_1, (256, 384), (1, 256), 0), out=buf37)
        del arg22_1
        buf38 = reinterpret_tensor(buf8, (1, 50, 384), (19200, 384, 1), 0); del buf8  # reuse
        buf40 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_3, add_6, add_8, add_10, pow_4, mean_3, add_11, rsqrt_3, mul_7, mul_8], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_7.run(buf38, arg7_1, buf20, buf25, buf37, arg23_1, buf40, 50, 384, stream=stream0)
        del arg23_1
        del arg7_1
        buf41 = reinterpret_tensor(buf36, (50, 256), (256, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (50, 384), (384, 1), 0), reinterpret_tensor(arg24_1, (384, 256), (1, 384), 0), out=buf41)
        del arg24_1
        buf42 = reinterpret_tensor(buf35, (50, 256), (256, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg10_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg25_1, (384, 256), (1, 384), 0), out=buf42)
        del arg25_1
        buf43 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf41, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf42, (8, 32, 50), (32, 1, 256), 0), out=buf43)
        buf47 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [zeros, add_5, iadd_3, softmax_3], Original ATen: [aten.zeros, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_zeros_3.run(buf43, arg13_1, buf47, 400, 50, stream=stream0)
        buf46 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg10_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg26_1, (384, 256), (1, 384), 0), out=buf46)
        del arg26_1
        buf48 = empty_strided_cuda((8, 50, 32), (1600, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf47, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf46, (8, 50, 32), (32, 256, 1), 0), out=buf48)
        buf49 = empty_strided_cuda((1, 50, 8, 32), (12800, 256, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf48, buf49, 12800, stream=stream0)
        buf50 = reinterpret_tensor(buf40, (50, 384), (384, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (50, 256), (256, 1), 0), reinterpret_tensor(arg27_1, (256, 384), (1, 256), 0), out=buf50)
        del arg27_1
        buf52 = reinterpret_tensor(buf37, (1, 50, 384), (19200, 384, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [add_12, pow_5, mean_4, add_13, rsqrt_4, mul_9, mul_10], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_2.run(buf38, buf50, arg28_1, buf52, 50, 384, stream=stream0)
        del arg28_1
        buf53 = reinterpret_tensor(buf24, (50, 512), (512, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (50, 384), (384, 1), 0), reinterpret_tensor(arg29_1, (384, 512), (1, 384), 0), out=buf53)
        del arg29_1
        buf54 = reinterpret_tensor(buf53, (1, 50, 512), (25600, 512, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [relu_1], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf54, 25600, stream=stream0)
        buf55 = reinterpret_tensor(buf52, (50, 384), (384, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (50, 512), (512, 1), 0), reinterpret_tensor(arg30_1, (512, 384), (1, 512), 0), out=buf55)
        del arg30_1
        buf57 = reinterpret_tensor(buf25, (1, 50, 384), (19200, 384, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [add_12, add_14, pow_6, mean_5, add_15, rsqrt_5, mul_11, mul_12], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_4.run(buf38, buf50, buf55, arg31_1, buf57, 50, 384, stream=stream0)
        del arg31_1
        buf58 = reinterpret_tensor(buf49, (50, 256), (256, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (50, 384), (384, 1), 0), reinterpret_tensor(arg32_1, (384, 256), (1, 384), 0), out=buf58)
        del arg32_1
        buf59 = reinterpret_tensor(buf48, (50, 256), (256, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (50, 384), (384, 1), 0), reinterpret_tensor(arg33_1, (384, 256), (1, 384), 0), out=buf59)
        del arg33_1
        buf60 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf58, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf59, (8, 32, 50), (32, 1, 256), 0), out=buf60)
        buf63 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [add_2, iadd_4, softmax_4], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_0.run(buf60, arg4_1, arg5_1, buf63, 400, 50, stream=stream0)
        buf64 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (50, 384), (384, 1), 0), reinterpret_tensor(arg34_1, (384, 256), (1, 384), 0), out=buf64)
        del arg34_1
        buf65 = empty_strided_cuda((8, 50, 32), (1600, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf63, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf64, (8, 50, 32), (32, 256, 1), 0), out=buf65)
        buf66 = empty_strided_cuda((1, 50, 8, 32), (12800, 256, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf65, buf66, 12800, stream=stream0)
        buf67 = reinterpret_tensor(buf57, (50, 384), (384, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (50, 256), (256, 1), 0), reinterpret_tensor(arg35_1, (256, 384), (1, 256), 0), out=buf67)
        del arg35_1
        buf69 = reinterpret_tensor(buf20, (1, 50, 384), (19200, 384, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [add_12, add_14, add_16, pow_7, mean_6, add_17, rsqrt_6, mul_13, mul_14], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_6.run(buf38, buf50, buf55, buf67, arg36_1, buf69, 50, 384, stream=stream0)
        del arg36_1
        buf70 = reinterpret_tensor(buf66, (50, 256), (256, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (50, 384), (384, 1), 0), reinterpret_tensor(arg37_1, (384, 256), (1, 384), 0), out=buf70)
        del arg37_1
        buf71 = reinterpret_tensor(buf65, (50, 256), (256, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg10_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg38_1, (384, 256), (1, 384), 0), out=buf71)
        del arg38_1
        buf72 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf70, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf71, (8, 32, 50), (32, 1, 256), 0), out=buf72)
        buf76 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [zeros, add_5, iadd_5, softmax_5], Original ATen: [aten.zeros, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_zeros_3.run(buf72, arg13_1, buf76, 400, 50, stream=stream0)
        buf75 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg10_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg39_1, (384, 256), (1, 384), 0), out=buf75)
        del arg39_1
        buf77 = empty_strided_cuda((8, 50, 32), (1600, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf76, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf75, (8, 50, 32), (32, 256, 1), 0), out=buf77)
        buf78 = empty_strided_cuda((1, 50, 8, 32), (12800, 256, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf77, buf78, 12800, stream=stream0)
        buf79 = reinterpret_tensor(buf69, (50, 384), (384, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (50, 256), (256, 1), 0), reinterpret_tensor(arg40_1, (256, 384), (1, 256), 0), out=buf79)
        del arg40_1
        buf80 = buf38; del buf38  # reuse
        buf82 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_12, add_14, add_16, add_18, pow_8, mean_7, add_19, rsqrt_7, mul_15, mul_16], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_8.run(buf80, buf50, buf55, buf67, buf79, arg41_1, buf82, 50, 384, stream=stream0)
        del arg41_1
        buf83 = reinterpret_tensor(buf54, (50, 512), (512, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (50, 384), (384, 1), 0), reinterpret_tensor(arg42_1, (384, 512), (1, 384), 0), out=buf83)
        del arg42_1
        buf84 = reinterpret_tensor(buf83, (1, 50, 512), (25600, 512, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [relu_2], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf84, 25600, stream=stream0)
        buf85 = reinterpret_tensor(buf82, (50, 384), (384, 1), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (50, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 384), (1, 512), 0), out=buf85)
        del arg43_1
        buf87 = reinterpret_tensor(buf79, (1, 50, 384), (19200, 384, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [add_20, pow_9, mean_8, add_21, rsqrt_8, mul_17, mul_18], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_2.run(buf80, buf85, arg44_1, buf87, 50, 384, stream=stream0)
        del arg44_1
        buf88 = reinterpret_tensor(buf78, (50, 256), (256, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (50, 384), (384, 1), 0), reinterpret_tensor(arg45_1, (384, 256), (1, 384), 0), out=buf88)
        del arg45_1
        buf89 = reinterpret_tensor(buf77, (50, 256), (256, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (50, 384), (384, 1), 0), reinterpret_tensor(arg46_1, (384, 256), (1, 384), 0), out=buf89)
        del arg46_1
        buf90 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf88, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf89, (8, 32, 50), (32, 1, 256), 0), out=buf90)
        buf93 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [add_2, iadd_6, softmax_6], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_0.run(buf90, arg4_1, arg5_1, buf93, 400, 50, stream=stream0)
        buf94 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (50, 384), (384, 1), 0), reinterpret_tensor(arg47_1, (384, 256), (1, 384), 0), out=buf94)
        del arg47_1
        buf95 = empty_strided_cuda((8, 50, 32), (1600, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf93, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf94, (8, 50, 32), (32, 256, 1), 0), out=buf95)
        buf96 = empty_strided_cuda((1, 50, 8, 32), (12800, 256, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf95, buf96, 12800, stream=stream0)
        buf97 = reinterpret_tensor(buf87, (50, 384), (384, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf96, (50, 256), (256, 1), 0), reinterpret_tensor(arg48_1, (256, 384), (1, 256), 0), out=buf97)
        del arg48_1
        buf99 = reinterpret_tensor(buf67, (1, 50, 384), (19200, 384, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [add_20, add_22, pow_10, mean_9, add_23, rsqrt_9, mul_19, mul_20], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_4.run(buf80, buf85, buf97, arg49_1, buf99, 50, 384, stream=stream0)
        del arg49_1
        buf100 = reinterpret_tensor(buf96, (50, 256), (256, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (50, 384), (384, 1), 0), reinterpret_tensor(arg50_1, (384, 256), (1, 384), 0), out=buf100)
        del arg50_1
        buf101 = reinterpret_tensor(buf95, (50, 256), (256, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg10_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg51_1, (384, 256), (1, 384), 0), out=buf101)
        del arg51_1
        buf102 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf100, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf101, (8, 32, 50), (32, 1, 256), 0), out=buf102)
        buf106 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [zeros, add_5, iadd_7, softmax_7], Original ATen: [aten.zeros, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_zeros_3.run(buf102, arg13_1, buf106, 400, 50, stream=stream0)
        buf105 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg10_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg52_1, (384, 256), (1, 384), 0), out=buf105)
        del arg52_1
        buf107 = empty_strided_cuda((8, 50, 32), (1600, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf106, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf105, (8, 50, 32), (32, 256, 1), 0), out=buf107)
        buf108 = empty_strided_cuda((1, 50, 8, 32), (12800, 256, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_7], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf107, buf108, 12800, stream=stream0)
        buf109 = reinterpret_tensor(buf99, (50, 384), (384, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (50, 256), (256, 1), 0), reinterpret_tensor(arg53_1, (256, 384), (1, 256), 0), out=buf109)
        del arg53_1
        buf111 = reinterpret_tensor(buf55, (1, 50, 384), (19200, 384, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [add_20, add_22, add_24, pow_11, mean_10, add_25, rsqrt_10, mul_21, mul_22], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_6.run(buf80, buf85, buf97, buf109, arg54_1, buf111, 50, 384, stream=stream0)
        del arg54_1
        buf112 = reinterpret_tensor(buf84, (50, 512), (512, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (50, 384), (384, 1), 0), reinterpret_tensor(arg55_1, (384, 512), (1, 384), 0), out=buf112)
        del arg55_1
        buf113 = reinterpret_tensor(buf112, (1, 50, 512), (25600, 512, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [relu_3], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf113, 25600, stream=stream0)
        buf114 = reinterpret_tensor(buf111, (50, 384), (384, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (50, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 384), (1, 512), 0), out=buf114)
        del arg56_1
        buf115 = buf80; del buf80  # reuse
        buf117 = reinterpret_tensor(buf50, (1, 50, 384), (19200, 384, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [add_20, add_22, add_24, add_26, pow_12, mean_11, add_27, rsqrt_11, mul_23, mul_24], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_8.run(buf115, buf85, buf97, buf109, buf114, arg57_1, buf117, 50, 384, stream=stream0)
        del arg57_1
        buf118 = reinterpret_tensor(buf108, (50, 256), (256, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (50, 384), (384, 1), 0), reinterpret_tensor(arg58_1, (384, 256), (1, 384), 0), out=buf118)
        del arg58_1
        buf119 = reinterpret_tensor(buf107, (50, 256), (256, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (50, 384), (384, 1), 0), reinterpret_tensor(arg59_1, (384, 256), (1, 384), 0), out=buf119)
        del arg59_1
        buf120 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf118, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf119, (8, 32, 50), (32, 1, 256), 0), out=buf120)
        buf123 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [add_2, iadd_8, softmax_8], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_0.run(buf120, arg4_1, arg5_1, buf123, 400, 50, stream=stream0)
        buf124 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (50, 384), (384, 1), 0), reinterpret_tensor(arg60_1, (384, 256), (1, 384), 0), out=buf124)
        del arg60_1
        buf125 = empty_strided_cuda((8, 50, 32), (1600, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf123, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf124, (8, 50, 32), (32, 256, 1), 0), out=buf125)
        buf126 = empty_strided_cuda((1, 50, 8, 32), (12800, 256, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf125, buf126, 12800, stream=stream0)
        buf127 = reinterpret_tensor(buf117, (50, 384), (384, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (50, 256), (256, 1), 0), reinterpret_tensor(arg61_1, (256, 384), (1, 256), 0), out=buf127)
        del arg61_1
        buf129 = reinterpret_tensor(buf97, (1, 50, 384), (19200, 384, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [add_28, pow_13, mean_12, add_29, rsqrt_12, mul_25, mul_26], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_2.run(buf115, buf127, arg62_1, buf129, 50, 384, stream=stream0)
        del arg62_1
        buf130 = reinterpret_tensor(buf126, (50, 256), (256, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (50, 384), (384, 1), 0), reinterpret_tensor(arg63_1, (384, 256), (1, 384), 0), out=buf130)
        del arg63_1
        buf131 = reinterpret_tensor(buf125, (50, 256), (256, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg10_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg64_1, (384, 256), (1, 384), 0), out=buf131)
        del arg64_1
        buf132 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf130, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf131, (8, 32, 50), (32, 1, 256), 0), out=buf132)
        buf136 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [zeros, add_5, iadd_9, softmax_9], Original ATen: [aten.zeros, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_zeros_3.run(buf132, arg13_1, buf136, 400, 50, stream=stream0)
        buf135 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg10_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg65_1, (384, 256), (1, 384), 0), out=buf135)
        del arg65_1
        buf137 = empty_strided_cuda((8, 50, 32), (1600, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf136, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf135, (8, 50, 32), (32, 256, 1), 0), out=buf137)
        buf138 = empty_strided_cuda((1, 50, 8, 32), (12800, 256, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_9], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf137, buf138, 12800, stream=stream0)
        buf139 = reinterpret_tensor(buf129, (50, 384), (384, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (50, 256), (256, 1), 0), reinterpret_tensor(arg66_1, (256, 384), (1, 256), 0), out=buf139)
        del arg66_1
        buf141 = reinterpret_tensor(buf85, (1, 50, 384), (19200, 384, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [add_28, add_30, pow_14, mean_13, add_31, rsqrt_13, mul_27, mul_28], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_4.run(buf115, buf127, buf139, arg67_1, buf141, 50, 384, stream=stream0)
        del arg67_1
        buf142 = reinterpret_tensor(buf113, (50, 512), (512, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (50, 384), (384, 1), 0), reinterpret_tensor(arg68_1, (384, 512), (1, 384), 0), out=buf142)
        del arg68_1
        buf143 = reinterpret_tensor(buf142, (1, 50, 512), (25600, 512, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [relu_4], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_5.run(buf143, 25600, stream=stream0)
        buf144 = reinterpret_tensor(buf141, (50, 384), (384, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (50, 512), (512, 1), 0), reinterpret_tensor(arg69_1, (512, 384), (1, 512), 0), out=buf144)
        del arg69_1
        del buf143
        buf146 = reinterpret_tensor(buf114, (1, 50, 384), (19200, 384, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [add_28, add_30, add_32, pow_15, mean_14, add_33, rsqrt_14, mul_29, mul_30], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_6.run(buf115, buf127, buf139, buf144, arg70_1, buf146, 50, 384, stream=stream0)
        del arg70_1
        buf147 = reinterpret_tensor(buf138, (50, 256), (256, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (50, 384), (384, 1), 0), reinterpret_tensor(arg71_1, (384, 256), (1, 384), 0), out=buf147)
        del arg71_1
        buf148 = reinterpret_tensor(buf137, (50, 256), (256, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (50, 384), (384, 1), 0), reinterpret_tensor(arg72_1, (384, 256), (1, 384), 0), out=buf148)
        del arg72_1
        buf149 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf147, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf148, (8, 32, 50), (32, 1, 256), 0), out=buf149)
        buf152 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [add_2, iadd_10, softmax_10], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_0.run(buf149, arg4_1, arg5_1, buf152, 400, 50, stream=stream0)
        del arg4_1
        del arg5_1
        buf153 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (50, 384), (384, 1), 0), reinterpret_tensor(arg73_1, (384, 256), (1, 384), 0), out=buf153)
        del arg73_1
        buf154 = empty_strided_cuda((8, 50, 32), (1600, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf152, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf153, (8, 50, 32), (32, 256, 1), 0), out=buf154)
        buf155 = empty_strided_cuda((1, 50, 8, 32), (12800, 256, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf154, buf155, 12800, stream=stream0)
        buf156 = reinterpret_tensor(buf146, (50, 384), (384, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [linear_52], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (50, 256), (256, 1), 0), reinterpret_tensor(arg74_1, (256, 384), (1, 256), 0), out=buf156)
        del arg74_1
        buf157 = buf115; del buf115  # reuse
        buf159 = reinterpret_tensor(buf109, (1, 50, 384), (19200, 384, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [add_28, add_30, add_32, add_34, pow_16, mean_15, add_35, rsqrt_15, mul_31, mul_32], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_8.run(buf157, buf127, buf139, buf144, buf156, arg75_1, buf159, 50, 384, stream=stream0)
        del arg75_1
        del buf127
        del buf139
        del buf144
        buf160 = reinterpret_tensor(buf155, (50, 256), (256, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (50, 384), (384, 1), 0), reinterpret_tensor(arg76_1, (384, 256), (1, 384), 0), out=buf160)
        del arg76_1
        buf161 = reinterpret_tensor(buf154, (50, 256), (256, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg10_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg77_1, (384, 256), (1, 384), 0), out=buf161)
        del arg77_1
        buf162 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf160, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf161, (8, 32, 50), (32, 1, 256), 0), out=buf162)
        buf166 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [zeros, add_5, iadd_11, softmax_11], Original ATen: [aten.zeros, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_zeros_3.run(buf162, arg13_1, buf166, 400, 50, stream=stream0)
        del arg13_1
        del buf162
        buf165 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg10_1, (50, 384), (384, 1), 0), reinterpret_tensor(arg78_1, (384, 256), (1, 384), 0), out=buf165)
        del arg10_1
        del arg78_1
        buf167 = empty_strided_cuda((8, 50, 32), (1600, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf166, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf165, (8, 50, 32), (32, 256, 1), 0), out=buf167)
        del buf166
        buf168 = empty_strided_cuda((1, 50, 8, 32), (12800, 256, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_11], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf167, buf168, 12800, stream=stream0)
        del buf167
        buf169 = reinterpret_tensor(buf159, (50, 384), (384, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (50, 256), (256, 1), 0), reinterpret_tensor(arg79_1, (256, 384), (1, 256), 0), out=buf169)
        del arg79_1
        del buf168
        buf170 = buf157; del buf157  # reuse
        buf172 = reinterpret_tensor(buf156, (1, 50, 384), (19200, 384, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [add_36, pow_17, mean_16, add_37, rsqrt_16, mul_33, mul_34], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf170, buf169, arg80_1, buf172, 50, 384, stream=stream0)
        del arg80_1
        del buf169
    return (buf172, buf170, reinterpret_tensor(buf0, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf5, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf12, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf16, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf29, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf34, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf42, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf46, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf59, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf64, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf71, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf75, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf89, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf94, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf101, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf105, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf119, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf124, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf131, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf135, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf148, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf153, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf161, (1, 8, 50, 32), (12800, 32, 256, 1), 0), reinterpret_tensor(buf165, (1, 8, 50, 32), (12800, 32, 256, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 50, 384), (19200, 384, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 8, 50, 32), (12800, 32, 256, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1, 1, 50, 50), (2500, 2500, 50, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((1, 50, 384), (19200, 384, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((1, 50, 384), (19200, 384, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((1, 1, 1, 50), (50, 50, 50, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
