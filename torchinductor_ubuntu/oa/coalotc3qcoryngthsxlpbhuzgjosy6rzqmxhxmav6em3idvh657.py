# AOT ID: ['1_inference']
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/7w/c7wzexheewt66elqhbot2ml5vncbo2ixopmep2ri6raectlkhska.py
# Topologically Sorted Source Nodes: [pow_1, mean, add, rsqrt, mul_1, mul_2], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   mean => mean
#   mul_1 => mul_1
#   mul_2 => mul_2
#   pow_1 => pow_1
#   rsqrt => rsqrt
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg1_1, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, %mul_1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 231936}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel):
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
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp13, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/nf/cnfehoh4k6l3otcz6shdp3chj7l7akuk6bepi5ielbhql6xqmt2v.py
# Topologically Sorted Source Nodes: [to, sub, mul, add_3, iadd_1, softmax], Original ATen: [aten._to_copy, aten.rsub, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   add_3 => add_4
#   iadd_1 => add_5
#   mul => mul
#   softmax => div_2
#   sub => sub
#   to => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_1, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %convert_element_type), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, -3.4028234663852886e+38), kwargs = {})
#   %add_4 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_4, %mul), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %add_4), kwargs = {})
#   %prepare_softmax_online_default_5 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_5, -1), kwargs = {})
#   %sub_tensor_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_10), kwargs = {})
#   %exp_default_5 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_5,), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_5, %getitem_11), kwargs = {})
triton_per_fused__softmax__to_copy_add_mul_rsub_1 = async_compile.triton('triton_per_fused__softmax__to_copy_add_mul_rsub_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_mul_rsub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_mul_rsub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 50)
    x1 = xindex // 50
    x3 = xindex
    tmp30 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr2 + (r0_2 + 50*x3), r0_mask & xmask, other=0.0)
    tmp0 = r0_2 + ((-1)*x0)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.full([1, 1], 16, tl.int64)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 + tmp1
    tmp7 = tl_math.abs(r0_2 + ((-1)*x0))
    tmp8 = tl.full([1, 1], 8, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp7.to(tl.float32)
    tmp11 = 0.125
    tmp12 = tmp10 * tmp11
    tmp13 = tl_math.log(tmp12)
    tmp14 = 0.36067376022224085
    tmp15 = tmp13 * tmp14
    tmp16 = 8.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.int64)
    tmp19 = tmp18 + tmp8
    tmp20 = tl.full([1, 1], 15, tl.int64)
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tl.where(tmp9, tmp7, tmp21)
    tmp23 = tmp6 + tmp22
    tmp24 = tl.full([XBLOCK, R0_BLOCK], 32, tl.int32)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp23 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp23)
    tl.device_assert(((0 <= tmp27) & (tmp27 < 32)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp27 < 32")
    tmp29 = tl.load(in_ptr0 + (x1 + 8*tmp27), r0_mask & xmask, eviction_policy='evict_last')
    tmp31 = tmp30.to(tl.float32)
    tmp32 = 1.0
    tmp33 = tmp32 - tmp31
    tmp34 = -3.4028234663852886e+38
    tmp35 = tmp33 * tmp34
    tmp36 = tmp29 + tmp35
    tmp38 = tmp37 + tmp36
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
    tmp43 = tl.where(r0_mask & xmask, tmp41, float("-inf"))
    tmp44 = triton_helpers.max2(tmp43, 1)[:, None]
    tmp45 = tmp39 - tmp44
    tmp46 = tl_math.exp(tmp45)
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, R0_BLOCK])
    tmp49 = tl.where(r0_mask & xmask, tmp47, 0)
    tmp50 = tl.sum(tmp49, 1)[:, None]
    tmp51 = tmp38 - tmp44
    tmp52 = tl_math.exp(tmp51)
    tmp53 = (tmp52 / tmp50)
    tl.store(out_ptr0 + (r0_2 + 50*x0 + 2528*x1), tmp36, r0_mask & xmask)
    tl.store(out_ptr3 + (r0_2 + 50*x0 + 2528*x1), tmp53, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/xj/cxjwy2vgtyvlt4bwcpcnicgn7rsvuvwlvtx3snep4xjzhbkaefyr.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 153600}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/ey/ceygvd7szk7zsh6cog3w5gl7jcq75tgyhcdz6kwz6fypifonhofg.py
# Topologically Sorted Source Nodes: [add_4, pow_2, mean_1, add_5, rsqrt_1, mul_5, mul_6], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_6
#   add_5 => add_7
#   mean_1 => mean_1
#   mul_5 => mul_5
#   mul_6 => mul_6
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
# Graph fragment:
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %view_19), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_6, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %rsqrt_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg8_1, %mul_5), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_3 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 308736}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_3(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/qv/cqv73jpmir4wl2tpdk5j4kxvfny6dphsxovcaw7clhlmhw72mrzo.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_21,), kwargs = {})
triton_poi_fused_relu_4 = async_compile.triton('triton_poi_fused_relu_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 307200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_4(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/gl/cglv5bxfiqegm7escpb66ltlly7f5vrp37mz7kgtvbp3s3o5xzkz.py
# Topologically Sorted Source Nodes: [add_4, add_6, pow_3, mean_2, add_7, rsqrt_2, mul_7, mul_8], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_6
#   add_6 => add_8
#   add_7 => add_9
#   mean_2 => mean_2
#   mul_7 => mul_7
#   mul_8 => mul_8
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
# Graph fragment:
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %view_19), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_23), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_8, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %rsqrt_2), kwargs = {})
#   %mul_8 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg11_1, %mul_7), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_5 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 385536}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/7m/c7mz7vr6bkrwjj5imx4qpkjzm5b35yar4qnwwkujddm3d4utpios.py
# Topologically Sorted Source Nodes: [iadd_2, softmax_1], Original ATen: [aten.add, aten._softmax]
# Source node to ATen node mapping:
#   iadd_2 => add_10
#   softmax_1 => div_3
# Graph fragment:
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_35, %add_4), kwargs = {})
#   %prepare_softmax_online_default_4 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_10, -1), kwargs = {})
#   %sub_tensor_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_10, %getitem_8), kwargs = {})
#   %exp_default_4 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_4,), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_4, %getitem_9), kwargs = {})
triton_per_fused__softmax_add_6 = async_compile.triton('triton_per_fused__softmax_add_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 320000}}
)
@triton.jit
def triton_per_fused__softmax_add_6(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r0_2 + 50*x0 + 2528*x1), r0_mask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp3 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp2 - tmp8
    tmp16 = tl_math.exp(tmp15)
    tmp17 = (tmp16 / tmp14)
    tl.store(out_ptr2 + (r0_2 + 50*x0 + 2528*x1), tmp17, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/hs/chsluzpp5gwza4va36sp3jkrlujnevoivtl6peuuf2yuelbnxizj.py
# Topologically Sorted Source Nodes: [add_4, add_6, add_8, pow_4, mean_3, add_9, rsqrt_3, mul_9, mul_10], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_6
#   add_6 => add_8
#   add_8 => add_11
#   add_9 => add_12
#   mean_3 => mean_3
#   mul_10 => mul_10
#   mul_9 => mul_9
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
# Graph fragment:
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %view_19), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_23), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_43), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_11, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %rsqrt_3), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg16_1, %mul_9), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 462336}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/3w/c3wkk7zwxdz6wwmes6mclt77unzxuxc2a4kquhob5euloq4pr5l4.py
# Topologically Sorted Source Nodes: [add_4, add_6, add_8, add_10, pow_5, mean_4, add_11, rsqrt_4, mul_11, mul_12], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_10 => add_13
#   add_11 => add_14
#   add_4 => add_6
#   add_6 => add_8
#   add_8 => add_11
#   mean_4 => mean_4
#   mul_11 => mul_11
#   mul_12 => mul_12
#   pow_5 => pow_5
#   rsqrt_4 => rsqrt_4
# Graph fragment:
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg1_1, %view_19), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_23), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_43), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_47), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_13, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %rsqrt_4), kwargs = {})
#   %mul_12 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg19_1, %mul_11), kwargs = {})
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/ha/chaq7meclr4gd4ab3ik5gsvdislt5vdzonivuycqglk7dvx3d45q.py
# Topologically Sorted Source Nodes: [add_12, add_14, add_16, add_18, pow_9, mean_8, add_19, rsqrt_8, mul_19, mul_20], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_12 => add_16
#   add_14 => add_18
#   add_16 => add_21
#   add_18 => add_23
#   add_19 => add_24
#   mean_8 => mean_8
#   mul_19 => mul_19
#   mul_20 => mul_20
#   pow_9 => pow_9
#   rsqrt_8 => rsqrt_8
# Graph fragment:
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_67), kwargs = {})
#   %add_18 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_71), kwargs = {})
#   %add_21 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %view_91), kwargs = {})
#   %add_23 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_95), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_23, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [-1], True), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_8, 1e-06), kwargs = {})
#   %rsqrt_8 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_24,), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %rsqrt_8), kwargs = {})
#   %mul_20 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg35_1, %mul_19), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 692736}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/jg/cjgujg2s3lvpsqkrlzuzbcldofxrz3zop22m6skn526gjzfe3ddl.py
# Topologically Sorted Source Nodes: [iadd_6, softmax_5], Original ATen: [aten.add, aten._softmax]
# Source node to ATen node mapping:
#   iadd_6 => add_30
#   softmax_5 => div_7
# Graph fragment:
#   %add_30 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_131, %add_4), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_30, -1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_30, %getitem), kwargs = {})
#   %exp_default : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
triton_per_fused__softmax_add_10 = async_compile.triton('triton_per_fused__softmax_add_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 320000}}
)
@triton.jit
def triton_per_fused__softmax_add_10(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_out_ptr0 + (r0_2 + 50*x0 + 2528*x1), r0_mask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp3 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp2 - tmp8
    tmp16 = tl_math.exp(tmp15)
    tmp17 = (tmp16 / tmp14)
    tl.store(in_out_ptr0 + (r0_2 + 50*x0 + 2528*x1), tmp17, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/fw/cfw6noecyf2qv4nyhtb3ike76rdvhvnngxozxqkpbuih7jkdhjir.py
# Topologically Sorted Source Nodes: [add_20, add_22, add_24, pow_12, mean_11, add_25, rsqrt_11, mul_25, mul_26], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_20 => add_26
#   add_22 => add_28
#   add_24 => add_31
#   add_25 => add_32
#   mean_11 => mean_11
#   mul_25 => mul_25
#   mul_26 => mul_26
#   pow_12 => pow_12
#   rsqrt_11 => rsqrt_11
# Graph fragment:
#   %add_26 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %view_115), kwargs = {})
#   %add_28 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %view_119), kwargs = {})
#   %add_31 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %view_139), kwargs = {})
#   %pow_12 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_31, 2), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_12, [-1], True), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_11, 1e-06), kwargs = {})
#   %rsqrt_11 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_32,), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_31, %rsqrt_11), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg48_1, %mul_25), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 615936}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel):
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
    tmp12 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
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
    tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp6, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp19, r0_mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 50), (50, 1))
    assert_size_stride(arg1_1, (1, 50, 384), (19200, 384, 1))
    assert_size_stride(arg2_1, (384, ), (1, ))
    assert_size_stride(arg3_1, (256, 384), (384, 1))
    assert_size_stride(arg4_1, (256, 384), (384, 1))
    assert_size_stride(arg5_1, (256, 384), (384, 1))
    assert_size_stride(arg6_1, (32, 8), (8, 1))
    assert_size_stride(arg7_1, (384, 256), (256, 1))
    assert_size_stride(arg8_1, (384, ), (1, ))
    assert_size_stride(arg9_1, (512, 384), (384, 1))
    assert_size_stride(arg10_1, (384, 512), (512, 1))
    assert_size_stride(arg11_1, (384, ), (1, ))
    assert_size_stride(arg12_1, (256, 384), (384, 1))
    assert_size_stride(arg13_1, (256, 384), (384, 1))
    assert_size_stride(arg14_1, (256, 384), (384, 1))
    assert_size_stride(arg15_1, (384, 256), (256, 1))
    assert_size_stride(arg16_1, (384, ), (1, ))
    assert_size_stride(arg17_1, (512, 384), (384, 1))
    assert_size_stride(arg18_1, (384, 512), (512, 1))
    assert_size_stride(arg19_1, (384, ), (1, ))
    assert_size_stride(arg20_1, (256, 384), (384, 1))
    assert_size_stride(arg21_1, (256, 384), (384, 1))
    assert_size_stride(arg22_1, (256, 384), (384, 1))
    assert_size_stride(arg23_1, (384, 256), (256, 1))
    assert_size_stride(arg24_1, (384, ), (1, ))
    assert_size_stride(arg25_1, (512, 384), (384, 1))
    assert_size_stride(arg26_1, (384, 512), (512, 1))
    assert_size_stride(arg27_1, (384, ), (1, ))
    assert_size_stride(arg28_1, (256, 384), (384, 1))
    assert_size_stride(arg29_1, (256, 384), (384, 1))
    assert_size_stride(arg30_1, (256, 384), (384, 1))
    assert_size_stride(arg31_1, (384, 256), (256, 1))
    assert_size_stride(arg32_1, (384, ), (1, ))
    assert_size_stride(arg33_1, (512, 384), (384, 1))
    assert_size_stride(arg34_1, (384, 512), (512, 1))
    assert_size_stride(arg35_1, (384, ), (1, ))
    assert_size_stride(arg36_1, (256, 384), (384, 1))
    assert_size_stride(arg37_1, (256, 384), (384, 1))
    assert_size_stride(arg38_1, (256, 384), (384, 1))
    assert_size_stride(arg39_1, (384, 256), (256, 1))
    assert_size_stride(arg40_1, (384, ), (1, ))
    assert_size_stride(arg41_1, (512, 384), (384, 1))
    assert_size_stride(arg42_1, (384, 512), (512, 1))
    assert_size_stride(arg43_1, (384, ), (1, ))
    assert_size_stride(arg44_1, (256, 384), (384, 1))
    assert_size_stride(arg45_1, (256, 384), (384, 1))
    assert_size_stride(arg46_1, (256, 384), (384, 1))
    assert_size_stride(arg47_1, (384, 256), (256, 1))
    assert_size_stride(arg48_1, (384, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pow_1, mean, add, rsqrt, mul_1, mul_2], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_0.run(arg1_1, arg2_1, buf1, 50, 384, stream=stream0)
        del arg2_1
        buf2 = empty_strided_cuda((50, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (50, 384), (384, 1), 0), reinterpret_tensor(arg3_1, (384, 256), (1, 384), 0), out=buf2)
        del arg3_1
        buf3 = empty_strided_cuda((50, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (50, 384), (384, 1), 0), reinterpret_tensor(arg4_1, (384, 256), (1, 384), 0), out=buf3)
        del arg4_1
        buf4 = empty_strided_cuda((8, 50, 50), (2500, 50, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf3, (8, 32, 50), (32, 1, 256), 0), out=buf4)
        buf5 = empty_strided_cuda((1, 8, 50, 50), (20224, 2528, 50, 1), torch.float32)
        buf9 = empty_strided_cuda((1, 8, 50, 50), (20224, 2528, 50, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to, sub, mul, add_3, iadd_1, softmax], Original ATen: [aten._to_copy, aten.rsub, aten.mul, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_mul_rsub_1.run(arg6_1, arg0_1, buf4, buf5, buf9, 400, 50, stream=stream0)
        del arg0_1
        del arg6_1
        buf8 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (50, 384), (384, 1), 0), reinterpret_tensor(arg5_1, (384, 256), (1, 384), 0), out=buf8)
        del arg5_1
        buf10 = reinterpret_tensor(buf2, (8, 50, 32), (1600, 32, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf8, (8, 50, 32), (32, 256, 1), 0), out=buf10)
        buf11 = reinterpret_tensor(buf8, (1, 50, 8, 32), (12800, 256, 32, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf10, buf11, 12800, stream=stream0)
        buf12 = reinterpret_tensor(buf1, (50, 384), (384, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (50, 256), (256, 1), 0), reinterpret_tensor(arg7_1, (256, 384), (1, 256), 0), out=buf12)
        del arg7_1
        buf14 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_4, pow_2, mean_1, add_5, rsqrt_1, mul_5, mul_6], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_3.run(arg1_1, buf12, arg8_1, buf14, 50, 384, stream=stream0)
        del arg8_1
        buf15 = empty_strided_cuda((50, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (50, 384), (384, 1), 0), reinterpret_tensor(arg9_1, (384, 512), (1, 384), 0), out=buf15)
        del arg9_1
        buf16 = reinterpret_tensor(buf15, (1, 50, 512), (25600, 512, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_4.run(buf16, 25600, stream=stream0)
        buf17 = reinterpret_tensor(buf14, (50, 384), (384, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (50, 512), (512, 1), 0), reinterpret_tensor(arg10_1, (512, 384), (1, 512), 0), out=buf17)
        del arg10_1
        buf19 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_4, add_6, pow_3, mean_2, add_7, rsqrt_2, mul_7, mul_8], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_5.run(arg1_1, buf12, buf17, arg11_1, buf19, 50, 384, stream=stream0)
        del arg11_1
        buf20 = reinterpret_tensor(buf11, (50, 256), (256, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (50, 384), (384, 1), 0), reinterpret_tensor(arg12_1, (384, 256), (1, 384), 0), out=buf20)
        del arg12_1
        buf21 = reinterpret_tensor(buf10, (50, 256), (256, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (50, 384), (384, 1), 0), reinterpret_tensor(arg13_1, (384, 256), (1, 384), 0), out=buf21)
        del arg13_1
        buf22 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf21, (8, 32, 50), (32, 1, 256), 0), out=buf22)
        buf26 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [iadd_2, softmax_1], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_6.run(buf22, buf5, buf26, 400, 50, stream=stream0)
        buf25 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (50, 384), (384, 1), 0), reinterpret_tensor(arg14_1, (384, 256), (1, 384), 0), out=buf25)
        del arg14_1
        buf27 = reinterpret_tensor(buf20, (8, 50, 32), (1600, 32, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf26, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf25, (8, 50, 32), (32, 256, 1), 0), out=buf27)
        buf28 = reinterpret_tensor(buf25, (1, 50, 8, 32), (12800, 256, 32, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf27, buf28, 12800, stream=stream0)
        buf29 = reinterpret_tensor(buf19, (50, 384), (384, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf28, (50, 256), (256, 1), 0), reinterpret_tensor(arg15_1, (256, 384), (1, 256), 0), out=buf29)
        del arg15_1
        buf31 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_4, add_6, add_8, pow_4, mean_3, add_9, rsqrt_3, mul_9, mul_10], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_7.run(arg1_1, buf12, buf17, buf29, arg16_1, buf31, 50, 384, stream=stream0)
        del arg16_1
        buf32 = reinterpret_tensor(buf16, (50, 512), (512, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (50, 384), (384, 1), 0), reinterpret_tensor(arg17_1, (384, 512), (1, 384), 0), out=buf32)
        del arg17_1
        buf33 = reinterpret_tensor(buf32, (1, 50, 512), (25600, 512, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [relu_1], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_4.run(buf33, 25600, stream=stream0)
        buf34 = reinterpret_tensor(buf31, (50, 384), (384, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (50, 512), (512, 1), 0), reinterpret_tensor(arg18_1, (512, 384), (1, 512), 0), out=buf34)
        del arg18_1
        buf35 = reinterpret_tensor(buf12, (1, 50, 384), (19200, 384, 1), 0); del buf12  # reuse
        buf37 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_4, add_6, add_8, add_10, pow_5, mean_4, add_11, rsqrt_4, mul_11, mul_12], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_8.run(buf35, arg1_1, buf17, buf29, buf34, arg19_1, buf37, 50, 384, stream=stream0)
        del arg19_1
        del arg1_1
        buf38 = reinterpret_tensor(buf28, (50, 256), (256, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (50, 384), (384, 1), 0), reinterpret_tensor(arg20_1, (384, 256), (1, 384), 0), out=buf38)
        del arg20_1
        buf39 = reinterpret_tensor(buf27, (50, 256), (256, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (50, 384), (384, 1), 0), reinterpret_tensor(arg21_1, (384, 256), (1, 384), 0), out=buf39)
        del arg21_1
        buf40 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf38, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf39, (8, 32, 50), (32, 1, 256), 0), out=buf40)
        buf44 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [iadd_3, softmax_2], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_6.run(buf40, buf5, buf44, 400, 50, stream=stream0)
        buf43 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (50, 384), (384, 1), 0), reinterpret_tensor(arg22_1, (384, 256), (1, 384), 0), out=buf43)
        del arg22_1
        buf45 = reinterpret_tensor(buf38, (8, 50, 32), (1600, 32, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf44, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf43, (8, 50, 32), (32, 256, 1), 0), out=buf45)
        buf46 = reinterpret_tensor(buf43, (1, 50, 8, 32), (12800, 256, 32, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf45, buf46, 12800, stream=stream0)
        buf47 = reinterpret_tensor(buf37, (50, 384), (384, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (50, 256), (256, 1), 0), reinterpret_tensor(arg23_1, (256, 384), (1, 256), 0), out=buf47)
        del arg23_1
        buf49 = reinterpret_tensor(buf34, (1, 50, 384), (19200, 384, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [add_12, pow_6, mean_5, add_13, rsqrt_5, mul_13, mul_14], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_3.run(buf35, buf47, arg24_1, buf49, 50, 384, stream=stream0)
        del arg24_1
        buf50 = reinterpret_tensor(buf33, (50, 512), (512, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (50, 384), (384, 1), 0), reinterpret_tensor(arg25_1, (384, 512), (1, 384), 0), out=buf50)
        del arg25_1
        buf51 = reinterpret_tensor(buf50, (1, 50, 512), (25600, 512, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [relu_2], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_4.run(buf51, 25600, stream=stream0)
        buf52 = reinterpret_tensor(buf49, (50, 384), (384, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf51, (50, 512), (512, 1), 0), reinterpret_tensor(arg26_1, (512, 384), (1, 512), 0), out=buf52)
        del arg26_1
        buf54 = reinterpret_tensor(buf29, (1, 50, 384), (19200, 384, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [add_12, add_14, pow_7, mean_6, add_15, rsqrt_6, mul_15, mul_16], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_5.run(buf35, buf47, buf52, arg27_1, buf54, 50, 384, stream=stream0)
        del arg27_1
        buf55 = reinterpret_tensor(buf46, (50, 256), (256, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (50, 384), (384, 1), 0), reinterpret_tensor(arg28_1, (384, 256), (1, 384), 0), out=buf55)
        del arg28_1
        buf56 = reinterpret_tensor(buf45, (50, 256), (256, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (50, 384), (384, 1), 0), reinterpret_tensor(arg29_1, (384, 256), (1, 384), 0), out=buf56)
        del arg29_1
        buf57 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf55, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf56, (8, 32, 50), (32, 1, 256), 0), out=buf57)
        buf61 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [iadd_4, softmax_3], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_6.run(buf57, buf5, buf61, 400, 50, stream=stream0)
        buf60 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (50, 384), (384, 1), 0), reinterpret_tensor(arg30_1, (384, 256), (1, 384), 0), out=buf60)
        del arg30_1
        buf62 = reinterpret_tensor(buf55, (8, 50, 32), (1600, 32, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf61, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf60, (8, 50, 32), (32, 256, 1), 0), out=buf62)
        buf63 = reinterpret_tensor(buf60, (1, 50, 8, 32), (12800, 256, 32, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [contiguous_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf62, buf63, 12800, stream=stream0)
        buf64 = reinterpret_tensor(buf54, (50, 384), (384, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (50, 256), (256, 1), 0), reinterpret_tensor(arg31_1, (256, 384), (1, 256), 0), out=buf64)
        del arg31_1
        buf66 = reinterpret_tensor(buf17, (1, 50, 384), (19200, 384, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [add_12, add_14, add_16, pow_8, mean_7, add_17, rsqrt_7, mul_17, mul_18], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_7.run(buf35, buf47, buf52, buf64, arg32_1, buf66, 50, 384, stream=stream0)
        del arg32_1
        buf67 = reinterpret_tensor(buf51, (50, 512), (512, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (50, 384), (384, 1), 0), reinterpret_tensor(arg33_1, (384, 512), (1, 384), 0), out=buf67)
        del arg33_1
        buf68 = reinterpret_tensor(buf67, (1, 50, 512), (25600, 512, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [relu_3], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_4.run(buf68, 25600, stream=stream0)
        buf69 = reinterpret_tensor(buf66, (50, 384), (384, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (50, 512), (512, 1), 0), reinterpret_tensor(arg34_1, (512, 384), (1, 512), 0), out=buf69)
        del arg34_1
        buf70 = buf35; del buf35  # reuse
        buf72 = empty_strided_cuda((1, 50, 384), (19200, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_12, add_14, add_16, add_18, pow_9, mean_8, add_19, rsqrt_8, mul_19, mul_20], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf70, buf47, buf52, buf64, buf69, arg35_1, buf72, 50, 384, stream=stream0)
        del arg35_1
        del buf47
        buf73 = reinterpret_tensor(buf63, (50, 256), (256, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (50, 384), (384, 1), 0), reinterpret_tensor(arg36_1, (384, 256), (1, 384), 0), out=buf73)
        del arg36_1
        buf74 = reinterpret_tensor(buf62, (50, 256), (256, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (50, 384), (384, 1), 0), reinterpret_tensor(arg37_1, (384, 256), (1, 384), 0), out=buf74)
        del arg37_1
        buf75 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf74, (8, 32, 50), (32, 1, 256), 0), out=buf75)
        buf79 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [iadd_5, softmax_4], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_6.run(buf75, buf5, buf79, 400, 50, stream=stream0)
        buf78 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (50, 384), (384, 1), 0), reinterpret_tensor(arg38_1, (384, 256), (1, 384), 0), out=buf78)
        del arg38_1
        buf80 = reinterpret_tensor(buf73, (8, 50, 32), (1600, 32, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf78, (8, 50, 32), (32, 256, 1), 0), out=buf80)
        del buf79
        buf81 = reinterpret_tensor(buf78, (1, 50, 8, 32), (12800, 256, 32, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf80, buf81, 12800, stream=stream0)
        buf82 = reinterpret_tensor(buf72, (50, 384), (384, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (50, 256), (256, 1), 0), reinterpret_tensor(arg39_1, (256, 384), (1, 256), 0), out=buf82)
        del arg39_1
        buf84 = reinterpret_tensor(buf69, (1, 50, 384), (19200, 384, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [add_20, pow_10, mean_9, add_21, rsqrt_9, mul_21, mul_22], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_3.run(buf70, buf82, arg40_1, buf84, 50, 384, stream=stream0)
        del arg40_1
        buf85 = reinterpret_tensor(buf68, (50, 512), (512, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (50, 384), (384, 1), 0), reinterpret_tensor(arg41_1, (384, 512), (1, 384), 0), out=buf85)
        del arg41_1
        buf86 = reinterpret_tensor(buf85, (1, 50, 512), (25600, 512, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [relu_4], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_4.run(buf86, 25600, stream=stream0)
        buf87 = reinterpret_tensor(buf84, (50, 384), (384, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (50, 512), (512, 1), 0), reinterpret_tensor(arg42_1, (512, 384), (1, 512), 0), out=buf87)
        del arg42_1
        del buf86
        buf89 = reinterpret_tensor(buf64, (1, 50, 384), (19200, 384, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [add_20, add_22, pow_11, mean_10, add_23, rsqrt_10, mul_23, mul_24], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_5.run(buf70, buf82, buf87, arg43_1, buf89, 50, 384, stream=stream0)
        del arg43_1
        buf90 = reinterpret_tensor(buf81, (50, 256), (256, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (50, 384), (384, 1), 0), reinterpret_tensor(arg44_1, (384, 256), (1, 384), 0), out=buf90)
        del arg44_1
        buf91 = reinterpret_tensor(buf80, (50, 256), (256, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (50, 384), (384, 1), 0), reinterpret_tensor(arg45_1, (384, 256), (1, 384), 0), out=buf91)
        del arg45_1
        buf92 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (8, 50, 32), (32, 256, 1), 0), reinterpret_tensor(buf91, (8, 32, 50), (32, 1, 256), 0), out=buf92)
        buf96 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [iadd_6, softmax_5], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_add_10.run(buf96, buf92, 400, 50, stream=stream0)
        del buf92
        buf95 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (50, 384), (384, 1), 0), reinterpret_tensor(arg46_1, (384, 256), (1, 384), 0), out=buf95)
        del arg46_1
        buf97 = reinterpret_tensor(buf90, (8, 50, 32), (1600, 32, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf96, (8, 50, 50), (2528, 50, 1), 0), reinterpret_tensor(buf95, (8, 50, 32), (32, 256, 1), 0), out=buf97)
        del buf96
        buf98 = reinterpret_tensor(buf95, (1, 50, 8, 32), (12800, 256, 32, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf97, buf98, 12800, stream=stream0)
        del buf97
        buf99 = reinterpret_tensor(buf89, (50, 384), (384, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf98, (50, 256), (256, 1), 0), reinterpret_tensor(arg47_1, (256, 384), (1, 256), 0), out=buf99)
        del arg47_1
        del buf98
        buf100 = buf70; del buf70  # reuse
        buf102 = reinterpret_tensor(buf52, (1, 50, 384), (19200, 384, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [add_20, add_22, add_24, pow_12, mean_11, add_25, rsqrt_11, mul_25, mul_26], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf100, buf82, buf87, buf99, arg48_1, buf102, 50, 384, stream=stream0)
        del arg48_1
        del buf82
        del buf87
        del buf99
    return (buf102, buf100, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 50), (50, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 50, 384), (19200, 384, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
