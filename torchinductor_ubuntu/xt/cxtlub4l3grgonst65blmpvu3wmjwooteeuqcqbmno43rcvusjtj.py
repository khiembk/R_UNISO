# AOT ID: ['9_inference']
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/ug/cuggpsc3ohxg4teqelsr3xqew2hbnyultfmksaqdqmuzrupdzg26.py
# Topologically Sorted Source Nodes: [embedding, pow_1, mean, add, rsqrt, mul_1, mul_2], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   embedding => embedding
#   mean => mean
#   mul_1 => mul_1
#   mul_2 => mul_2
#   pow_1 => pow_1
#   rsqrt => rsqrt
# Graph fragment:
#   %embedding : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%embedding, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, %mul_1), kwargs = {})
triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 32128, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 32128)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 32128")
        tmp6 = tl.load(in_ptr1 + (r0_1 + 512*tmp4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp6 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp11 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full([XBLOCK, R0_BLOCK], 32128, tl.int32)
        tmp13 = tmp0 + tmp12
        tmp14 = tmp0 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp0)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 32128)) | ~(xmask), "index out of bounds: 0 <= tmp15 < 32128")
        tmp17 = tl.load(in_ptr1 + (r0_1 + 512*tmp15), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = 512.0
        tmp19 = (tmp9 / tmp18)
        tmp20 = 1e-06
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp24 = tmp11 * tmp23
        tl.store(out_ptr1 + (r0_1 + 512*x0), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/q3/cq3kxhk6ezre54h5ziwl6hsecpapb2olfiljuaorambagbt7lmwk.py
# Topologically Sorted Source Nodes: [to, sub, mul, add_3], Original ATen: [aten._to_copy, aten.rsub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add_3 => add_4
#   mul => mul
#   sub => sub
#   to => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_1, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %convert_element_type), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, -3.4028234663852886e+38), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_4, %mul), kwargs = {})
triton_poi_fused__to_copy_add_mul_rsub_1 = async_compile.triton('triton_poi_fused__to_copy_add_mul_rsub_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_rsub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_rsub_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 64)
    x2 = xindex // 512
    x0 = (xindex % 8)
    x4 = xindex
    tmp30 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1 + ((-1)*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.full([1], 16, tl.int64)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 + tmp1
    tmp7 = tl_math.abs(x1 + ((-1)*x2))
    tmp8 = tl.full([1], 8, tl.int64)
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
    tmp20 = tl.full([1], 15, tl.int64)
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tl.where(tmp9, tmp7, tmp21)
    tmp23 = tmp6 + tmp22
    tmp24 = tl.full([XBLOCK], 32, tl.int32)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp23 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp23)
    tl.device_assert((0 <= tmp27) & (tmp27 < 32), "index out of bounds: 0 <= tmp27 < 32")
    tmp29 = tl.load(in_ptr0 + (x0 + 8*tmp27), None)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = 1.0
    tmp33 = tmp32 - tmp31
    tmp34 = -3.4028234663852886e+38
    tmp35 = tmp33 * tmp34
    tmp36 = tmp29 + tmp35
    tl.store(out_ptr0 + (x4), tmp36, None)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/la/claemv32fqlvujzqgcxvvvdmn3ith7s5dli5nr6leneidikarcqt.py
# Topologically Sorted Source Nodes: [iadd_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   iadd_1 => add_5
# Graph fragment:
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_12, %add_4), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_5, -1), kwargs = {})
triton_per_fused_add_2 = async_compile.triton('triton_per_fused_add_2', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 131072, 'x': 8192, 'r0_': 131072}}
)
@triton.jit
def triton_per_fused_add_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/bn/cbnwmtvai6eyv2vnbx5a5dqsz4avapf4xfblx6js3upepstdpxwj.py
# Topologically Sorted Source Nodes: [iadd_1, softmax], Original ATen: [aten.add, aten._softmax]
# Source node to ATen node mapping:
#   iadd_1 => add_5
#   softmax => div_2
# Graph fragment:
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_12, %add_4), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_2), kwargs = {})
#   %exp_default_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %getitem_3), kwargs = {})
triton_poi_fused__softmax_add_3 = async_compile.triton('triton_poi_fused__softmax_add_3', '''
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
    inductor_meta={'grid_type': 'Grid3D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'z': 131072, 'y': 4096, 'x': 393216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, znumel, ynumel, xnumel, ZBLOCK : tl.constexpr, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/hz/chzibmqdgboz2tiebq3rvfqta4prppwgs46qsejwzskpfnocpaeq.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/ys/cysqe6nnnabirq4oi2rshbgx2umf7p4ionmnjgsdtuonxuvzwq2k.py
# Topologically Sorted Source Nodes: [embedding, add_4, pow_2, mean_1, add_5, rsqrt_1, mul_5, mul_6], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_6
#   add_5 => add_7
#   embedding => embedding
#   mean_1 => mean_1
#   mul_5 => mul_5
#   mul_6 => mul_6
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
# Graph fragment:
#   %embedding : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_20), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_6, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %rsqrt_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg9_1, %mul_5), kwargs = {})
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_5 = async_compile.triton('triton_per_fused_add_embedding_mean_mul_pow_rsqrt_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_embedding_mean_mul_pow_rsqrt_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel):
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 512*x0), None)
    tmp13 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.full([R0_BLOCK], 32128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32128), "index out of bounds: 0 <= tmp4 < 32128")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 512*tmp4), None)
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
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp20, None)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/cj/ccjjohixofzjxrvs2rpz66b7wfr2kt3ieux6uiwzpgm7ahuvg7up.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_22,), kwargs = {})
triton_poi_fused_relu_6 = async_compile.triton('triton_poi_fused_relu_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1572864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_6(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/5t/c5tqlz75cvds6xe33hwpvurzrkrvas5x57rc6jm7lhgiqksr77c4.py
# Topologically Sorted Source Nodes: [embedding, add_4, add_6, pow_3, mean_2, add_7, rsqrt_2, mul_7, mul_8], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_6
#   add_6 => add_8
#   add_7 => add_9
#   embedding => embedding
#   mean_2 => mean_2
#   mul_7 => mul_7
#   mul_8 => mul_8
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
# Graph fragment:
#   %embedding : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_20), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_24), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_8, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %rsqrt_2), kwargs = {})
#   %mul_8 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg12_1, %mul_7), kwargs = {})
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel):
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None)
    tmp9 = tl.load(in_ptr2 + (r0_1 + 512*x0), None)
    tmp15 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.full([R0_BLOCK], 32128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32128), "index out of bounds: 0 <= tmp4 < 32128")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 512*tmp4), None)
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [R0_BLOCK])
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp16 = 512.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp10 * tmp20
    tmp22 = tmp15 * tmp21
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp10, None)
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp22, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1 = args
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
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 64, 512), (32768, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, pow_1, mean, add, rsqrt, mul_1, mul_2], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0.run(arg0_1, arg1_1, arg3_1, buf1, 64, 512, stream=stream0)
        del arg3_1
        buf2 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (64, 512), (512, 1), 0), reinterpret_tensor(arg4_1, (512, 512), (1, 512), 0), out=buf2)
        del arg4_1
        buf3 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (64, 512), (512, 1), 0), reinterpret_tensor(arg5_1, (512, 512), (1, 512), 0), out=buf3)
        del arg5_1
        buf4 = empty_strided_cuda((8, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (8, 64, 64), (64, 512, 1), 0), reinterpret_tensor(buf3, (8, 64, 64), (64, 1, 512), 0), out=buf4)
        buf5 = reinterpret_tensor(buf3, (1, 8, 64, 64), (8, 1, 512, 8), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [to, sub, mul, add_3], Original ATen: [aten._to_copy, aten.rsub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mul_rsub_1.run(arg7_1, arg2_1, buf5, 32768, stream=stream0)
        del arg2_1
        del arg7_1
        buf6 = empty_strided_cuda((1, 8, 64, 1), (512, 64, 1, 512), torch.float32)
        buf7 = empty_strided_cuda((1, 8, 64, 1), (512, 64, 1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [iadd_1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_2.run(buf4, buf5, buf6, buf7, 8, 64, 64, stream=stream0)
        buf8 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (64, 512), (512, 1), 0), reinterpret_tensor(arg6_1, (512, 512), (1, 512), 0), out=buf8)
        del arg6_1
        buf9 = reinterpret_tensor(buf4, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [iadd_1, softmax], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_3.run(buf9, buf5, buf6, buf7, 8, 64, 64, stream=stream0)
        buf10 = reinterpret_tensor(buf1, (8, 64, 64), (4096, 64, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (8, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf8, (8, 64, 64), (64, 512, 1), 0), out=buf10)
        buf11 = reinterpret_tensor(buf9, (1, 64, 8, 64), (32768, 512, 64, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf10, buf11, 32768, stream=stream0)
        buf12 = reinterpret_tensor(buf10, (64, 512), (512, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (64, 512), (512, 1), 0), reinterpret_tensor(arg8_1, (512, 512), (1, 512), 0), out=buf12)
        del arg8_1
        buf14 = reinterpret_tensor(buf11, (1, 64, 512), (32768, 512, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [embedding, add_4, pow_2, mean_1, add_5, rsqrt_1, mul_5, mul_6], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_5.run(arg0_1, arg1_1, buf12, arg9_1, buf14, 64, 512, stream=stream0)
        del arg9_1
        buf15 = empty_strided_cuda((64, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (64, 512), (512, 1), 0), reinterpret_tensor(arg10_1, (512, 2048), (1, 512), 0), out=buf15)
        del arg10_1
        buf16 = reinterpret_tensor(buf15, (1, 64, 2048), (131072, 2048, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_6.run(buf16, 131072, stream=stream0)
        buf17 = reinterpret_tensor(buf14, (64, 512), (512, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (64, 2048), (2048, 1), 0), reinterpret_tensor(arg11_1, (2048, 512), (1, 2048), 0), out=buf17)
        del arg11_1
        del buf16
        buf18 = reinterpret_tensor(buf12, (1, 64, 512), (32768, 512, 1), 0); del buf12  # reuse
        buf20 = reinterpret_tensor(buf8, (1, 64, 512), (32768, 512, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [embedding, add_4, add_6, pow_3, mean_2, add_7, rsqrt_2, mul_7, mul_8], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_7.run(buf18, arg0_1, arg1_1, buf17, arg12_1, buf20, 64, 512, stream=stream0)
        del arg0_1
        del arg12_1
        del arg1_1
        buf21 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (64, 512), (512, 1), 0), reinterpret_tensor(arg13_1, (512, 512), (1, 512), 0), out=buf21)
        del arg13_1
        buf22 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (64, 512), (512, 1), 0), reinterpret_tensor(arg14_1, (512, 512), (1, 512), 0), out=buf22)
        del arg14_1
        buf23 = empty_strided_cuda((8, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (8, 64, 64), (64, 512, 1), 0), reinterpret_tensor(buf22, (8, 64, 64), (64, 1, 512), 0), out=buf23)
        del buf21
        buf24 = buf7; del buf7  # reuse
        buf25 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [iadd_2], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_2.run(buf23, buf5, buf24, buf25, 8, 64, 64, stream=stream0)
        buf26 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (64, 512), (512, 1), 0), reinterpret_tensor(arg15_1, (512, 512), (1, 512), 0), out=buf26)
        del arg15_1
        buf27 = reinterpret_tensor(buf23, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [iadd_2, softmax_1], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_3.run(buf27, buf5, buf24, buf25, 8, 64, 64, stream=stream0)
        del buf24
        del buf25
        buf28 = reinterpret_tensor(buf20, (8, 64, 64), (4096, 64, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf27, (8, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf26, (8, 64, 64), (64, 512, 1), 0), out=buf28)
        del buf26
        buf29 = reinterpret_tensor(buf27, (1, 64, 8, 64), (32768, 512, 64, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf28, buf29, 32768, stream=stream0)
        del buf28
    return (reinterpret_tensor(buf29, (1, 64, 512), (32768, 512, 1), 0), buf18, buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 64), (64, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((32128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 64), (64, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
