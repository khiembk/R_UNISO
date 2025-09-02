# AOT ID: ['14_inference']
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/kz/ckzqi7x2at4m3jxs6zdc46t47t6lw6nmlhnn2rg6szev4wipvslb.py
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
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1296
    r0_numel = 384
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
        tmp6 = tl.load(in_ptr1 + (r0_1 + 384*tmp4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp17 = tl.load(in_ptr1 + (r0_1 + 384*tmp15), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = 384.0
        tmp19 = (tmp9 / tmp18)
        tmp20 = 1e-06
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp24 = tmp11 * tmp23
        tl.store(out_ptr1 + (r0_1 + 384*x0), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/3d/c3daqynt2hriz3zhpmrtvr4pbbx72jc57bhsxsofwy7il53jyadn.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_scalar_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_default_20, 1.0), kwargs = {})
#   %clone_default_15 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_default_20,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3981312}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 331776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 324)
    x2 = ((xindex // 10368) % 8)
    x3 = xindex // 82944
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 256*x1 + 82944*x3), None)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/hh/chh4ejsizsnp6bru5hz2bxcf62jsquzrqpa6xoyynnkb7ghsg6b7.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_scalar_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_default_23, 1.0), kwargs = {})
#   %clone_default_16 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_default_21,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1327104, 'x': 2654208}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 324
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 82944*y1), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + 324*y3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/mq/cmqv6e24k4wvcxo4wvqimr25vnjajkg7vnjx43ryyeav3qiqwwsx.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_default_17 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_default_23,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3981312}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 331776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 324)
    x2 = ((xindex // 10368) % 8)
    x3 = xindex // 82944
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 256*x1 + 82944*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/ci/ccisb2yra76ab2haychl3vkjloggui7jornx4qodogdjo7v6y64e.py
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
#   %add_4 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_4, %mul), kwargs = {})
#   %add_tensor_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_32, %add_4), kwargs = {})
#   %eq_scalar_5 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_tensor_5, -inf), kwargs = {})
#   %logical_not_default_10 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_scalar_5,), kwargs = {})
#   %any_dim_5 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_default_10, -1, True), kwargs = {})
#   %logical_not_default_11 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dim_5,), kwargs = {})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 8, 324, 324], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %prepare_softmax_online_default_5 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_tensor_5, -1), kwargs = {})
#   %sub_tensor_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor_5, %getitem_10), kwargs = {})
#   %exp_default_11 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_11,), kwargs = {})
#   %div_tensor_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_11, %getitem_11), kwargs = {})
#   %where_self_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_11, %full_default_6, %div_tensor_5), kwargs = {})
triton_red_fused__to_copy_add_mul_rsub_4 = async_compile.triton('triton_red_fused__to_copy_add_mul_rsub_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_rsub_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_rsub_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 10368
    r0_numel = 324
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 324)
    x1 = ((xindex // 324) % 8)
    x2 = xindex // 2592
    x4 = xindex // 324
    x6 = xindex
    _tmp45 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    x5 = (xindex % 2592)
    _tmp48_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp48_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp30 = tl.load(in_ptr1 + (r0_3 + 324*x2), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr2 + (r0_3 + 324*x6), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = r0_3 + ((-1)*x0)
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 > tmp1
        tmp3 = tmp2.to(tl.int64)
        tmp4 = tl.full([1, 1], 16, tl.int64)
        tmp5 = tmp3 * tmp4
        tmp6 = tmp5 + tmp1
        tmp7 = tl_math.abs(r0_3 + ((-1)*x0))
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
        tmp39 = float("-inf")
        tmp40 = tmp38 == tmp39
        tmp41 = tmp40 == 0
        tmp42 = tmp41.to(tl.int64)
        tmp43 = (tmp42 != 0)
        tmp44 = tl.broadcast_to(tmp43, [XBLOCK, R0_BLOCK])
        tmp46 = _tmp45 | tmp44
        _tmp45 = tl.where(r0_mask & xmask, tmp46, _tmp45)
        tmp47 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])

        _tmp48_max_next, _tmp48_sum_next = triton_helpers.online_softmax_combine(
            _tmp48_max, _tmp48_sum, tmp47, False
        )

        _tmp48_max = tl.where(r0_mask & xmask, _tmp48_max_next, _tmp48_max)
        _tmp48_sum = tl.where(r0_mask & xmask, _tmp48_sum_next, _tmp48_sum)
        tl.store(out_ptr0 + (r0_3 + 324*x0 + 104992*x4), tmp36, r0_mask & xmask)
    tmp45 = triton_helpers.any(_tmp45.to(tl.int8), 1)[:, None].to(tl.int1)

    tmp50, tmp51 = triton_helpers.online_softmax_reduce(
        _tmp48_max, _tmp48_sum, 1, False)
    tmp50 = tmp50[:, None]
    tmp51 = tmp51[:, None]
    tmp48 = tmp50
    tmp49 = tmp51
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp53 = tl.load(in_ptr2 + (r0_3 + 324*x6), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp54 = tl.load(out_ptr0 + (r0_3 + 324*x0 + 104992*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp52 = tmp45 == 0
        tmp55 = tmp53 + tmp54
        tmp56 = tmp55 - tmp48
        tmp57 = tl_math.exp(tmp56)
        tmp58 = (tmp57 / tmp49)
        tmp59 = 0.0
        tmp60 = tl.where(tmp52, tmp59, tmp58)
        tl.store(out_ptr4 + (r0_3 + 324*x0 + 104992*x4), tmp60, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/ap/captsip4a2dohe6uc7gpbp4psmmzhzwos7qchyt3hwznod4hnzmf.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3981312}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 331776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 8)
    x2 = ((xindex // 256) % 324)
    x3 = xindex // 82944
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 10368*x1 + 82944*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/an/can7zgqqxrvsop53mvdhkbj3mxtnv5rgw27tseez26h6subniohd.py
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
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_6 = async_compile.triton('triton_per_fused_add_embedding_mean_mul_pow_rsqrt_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_embedding_mean_mul_pow_rsqrt_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel):
    xnumel = 1296
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([R0_BLOCK], 32128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32128), "index out of bounds: 0 <= tmp4 < 32128")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 384*tmp4), r0_mask, other=0.0)
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
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp21, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/ti/ctidassbzoj2o5qay7l24wt6aptxxhh7vrdcg7mk77ylb2agso5b.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_22,), kwargs = {})
triton_poi_fused_relu_7 = async_compile.triton('triton_poi_fused_relu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7962624}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_7(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 663552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/b6/cb6tg7dsyq3l34opnkmos5seobgwozosc5edp5i54d257osybvrn.py
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
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_8 = async_compile.triton('triton_per_fused_add_embedding_mean_mul_pow_rsqrt_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_embedding_mean_mul_pow_rsqrt_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
    xnumel = 1296
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([R0_BLOCK], 32128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32128), "index out of bounds: 0 <= tmp4 < 32128")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 384*tmp4), r0_mask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [R0_BLOCK])
    tmp14 = tl.where(r0_mask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 384.0
    tmp18 = (tmp15 / tmp17)
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp10 * tmp21
    tmp23 = tmp16 * tmp22
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp23, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/oe/coefcjzwqeaazrpai4joq343fvrsgrhhb67vzijrbyjgc63uwhrl.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %add_tensor_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_26, %add_4), kwargs = {})
#   %eq_scalar_4 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_tensor_4, -inf), kwargs = {})
#   %logical_not_default_8 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_scalar_4,), kwargs = {})
#   %any_dim_4 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_default_8, -1, True), kwargs = {})
#   %logical_not_default_9 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dim_4,), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 8, 324, 324], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %prepare_softmax_online_default_4 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_tensor_4, -1), kwargs = {})
#   %sub_tensor_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor_4, %getitem_8), kwargs = {})
#   %exp_default_10 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_10,), kwargs = {})
#   %div_tensor_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_10, %getitem_9), kwargs = {})
#   %where_self_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_9, %full_default_5, %div_tensor_4), kwargs = {})
triton_red_fused_9 = async_compile.triton('triton_red_fused_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 53747712}}
)
@triton.jit
def triton_red_fused_9(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 10368
    r0_numel = 324
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x5 = xindex
    x0 = (xindex % 324)
    x4 = xindex // 324
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    x2 = xindex // 2592
    x6 = (xindex % 2592)
    _tmp12_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp12_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 324*x5), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 324*x0 + 104992*x4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = float("-inf")
        tmp4 = tmp2 == tmp3
        tmp5 = tmp4 == 0
        tmp6 = tmp5.to(tl.int64)
        tmp7 = (tmp6 != 0)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 | tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
        tmp11 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])

        _tmp12_max_next, _tmp12_sum_next = triton_helpers.online_softmax_combine(
            _tmp12_max, _tmp12_sum, tmp11, False
        )

        _tmp12_max = tl.where(r0_mask & xmask, _tmp12_max_next, _tmp12_max)
        _tmp12_sum = tl.where(r0_mask & xmask, _tmp12_sum_next, _tmp12_sum)
    tmp9 = triton_helpers.any(_tmp9.to(tl.int8), 1)[:, None].to(tl.int1)

    tmp14, tmp15 = triton_helpers.online_softmax_reduce(
        _tmp12_max, _tmp12_sum, 1, False)
    tmp14 = tmp14[:, None]
    tmp15 = tmp15[:, None]
    tmp12 = tmp14
    tmp13 = tmp15
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp17 = tl.load(in_ptr0 + (r0_3 + 324*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr1 + (r0_3 + 324*x0 + 104992*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tmp9 == 0
        tmp19 = tmp17 + tmp18
        tmp20 = tmp19 - tmp12
        tmp21 = tl_math.exp(tmp20)
        tmp22 = (tmp21 / tmp13)
        tmp23 = 0.0
        tmp24 = tl.where(tmp16, tmp23, tmp22)
        tl.store(out_ptr3 + (r0_3 + 324*x0 + 104992*x4), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/ms/cmskui5zdkdehzpvjklswh53pbihuhbhsdy24v5pozr2pr2nqxtz.py
# Topologically Sorted Source Nodes: [embedding, add_4, add_6, add_8, pow_4, mean_3, add_9, rsqrt_3, mul_9, mul_10], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_6
#   add_6 => add_8
#   add_8 => add_11
#   add_9 => add_12
#   embedding => embedding
#   mean_3 => mean_3
#   mul_10 => mul_10
#   mul_9 => mul_9
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
# Graph fragment:
#   %embedding : [num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_20), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_24), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_44), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_11, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %rsqrt_3), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg17_1, %mul_9), kwargs = {})
triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10 = async_compile.triton('triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
    xnumel = 1296
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0_1 + 384*x0), r0_mask, other=0.0)
    tmp18 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([R0_BLOCK], 32128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 32128), "index out of bounds: 0 <= tmp4 < 32128")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 384*tmp4), r0_mask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = 384.0
    tmp20 = (tmp17 / tmp19)
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp12 * tmp23
    tmp25 = tmp18 * tmp24
    tl.store(in_out_ptr0 + (r0_1 + 384*x0), tmp12, r0_mask)
    tl.store(out_ptr1 + (r0_1 + 384*x0), tmp25, r0_mask)
''', device_str='cuda')


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/5e/c5ehjulwbeg2sfaskpc6uvd4ta46jvxw5r25zbmr2zo3mpbc6r2d.py
# Topologically Sorted Source Nodes: [add_10, pow_5, mean_4, add_11, rsqrt_4, mul_11, mul_12], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_10 => add_13
#   add_11 => add_14
#   mean_4 => mean_4
#   mul_11 => mul_11
#   mul_12 => mul_12
#   pow_5 => pow_5
#   rsqrt_4 => rsqrt_4
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_48), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_13, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %rsqrt_4), kwargs = {})
#   %mul_12 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg20_1, %mul_11), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 7964160}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_11(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel):
    xnumel = 1296
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/fc/cfcdffjb2fu7qgz4ys6zi6pi5nilz37xgjs2wfbpobw7tndyojcb.py
# Topologically Sorted Source Nodes: [add_10, add_12, pow_6, mean_5, add_13, rsqrt_5, mul_13, mul_14], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_10 => add_13
#   add_12 => add_16
#   add_13 => add_17
#   mean_5 => mean_5
#   mul_13 => mul_13
#   mul_14 => mul_14
#   pow_6 => pow_6
#   rsqrt_5 => rsqrt_5
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_48), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_68), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_16, 2), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_6, [-1], True), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, 1e-06), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %rsqrt_5), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg25_1, %mul_13), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 9954816}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel):
    xnumel = 1296
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/py/cpyhifkbpewlrtbshcs5zryxdfj56w422zdsghigikosv7s7nd2c.py
# Topologically Sorted Source Nodes: [add_10, add_12, add_14, pow_7, mean_6, add_15, rsqrt_6, mul_15, mul_16], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_10 => add_13
#   add_12 => add_16
#   add_14 => add_18
#   add_15 => add_19
#   mean_6 => mean_6
#   mul_15 => mul_15
#   mul_16 => mul_16
#   pow_7 => pow_7
#   rsqrt_6 => rsqrt_6
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_48), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_68), kwargs = {})
#   %add_18 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_72), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_18, 2), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [-1], True), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_6, 1e-06), kwargs = {})
#   %rsqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %rsqrt_6), kwargs = {})
#   %mul_16 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg28_1, %mul_15), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 11945472}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
    xnumel = 1296
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/bq/cbqwkqkyn2k5cgaw3tdwldy743dvt7nlss2n46rjldjmewbwfaz5.py
# Topologically Sorted Source Nodes: [add_10, add_12, add_14, add_16, pow_8, mean_7, add_17, rsqrt_7, mul_17, mul_18], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_10 => add_13
#   add_12 => add_16
#   add_14 => add_18
#   add_16 => add_21
#   add_17 => add_22
#   mean_7 => mean_7
#   mul_17 => mul_17
#   mul_18 => mul_18
#   pow_8 => pow_8
#   rsqrt_7 => rsqrt_7
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_48), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_68), kwargs = {})
#   %add_18 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_72), kwargs = {})
#   %add_21 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %view_92), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_21, 2), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [-1], True), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_7, 1e-06), kwargs = {})
#   %rsqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %rsqrt_7), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg33_1, %mul_17), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 17917440}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
    xnumel = 1296
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/3s/c3s5rpotgkrxj53byxl6iiai4e64ppfdhf7r5r7xy2iceaf5vzca.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %add_tensor : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_default_2, %add_4), kwargs = {})
#   %eq_scalar : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_tensor, -inf), kwargs = {})
#   %logical_not_default : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_scalar,), kwargs = {})
#   %any_dim : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_default, -1, True), kwargs = {})
#   %logical_not_default_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dim,), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 8, 324, 324], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_tensor, -1), kwargs = {})
#   %sub_tensor_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_tensor, %getitem), kwargs = {})
#   %exp_default_6 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_6,), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_6, %getitem_1), kwargs = {})
#   %where_self : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_1, %full_default_1, %div_tensor), kwargs = {})
triton_red_fused_15 = async_compile.triton('triton_red_fused_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 53747712}}
)
@triton.jit
def triton_red_fused_15(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 10368
    r0_numel = 324
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x5 = xindex
    x0 = (xindex % 324)
    x4 = xindex // 324
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    x2 = xindex // 2592
    x6 = (xindex % 2592)
    _tmp12_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp12_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 324*x5), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_out_ptr0 + (r0_3 + 324*x0 + 104992*x4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = float("-inf")
        tmp4 = tmp2 == tmp3
        tmp5 = tmp4 == 0
        tmp6 = tmp5.to(tl.int64)
        tmp7 = (tmp6 != 0)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 | tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
        tmp11 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])

        _tmp12_max_next, _tmp12_sum_next = triton_helpers.online_softmax_combine(
            _tmp12_max, _tmp12_sum, tmp11, False
        )

        _tmp12_max = tl.where(r0_mask & xmask, _tmp12_max_next, _tmp12_max)
        _tmp12_sum = tl.where(r0_mask & xmask, _tmp12_sum_next, _tmp12_sum)
    tmp9 = triton_helpers.any(_tmp9.to(tl.int8), 1)[:, None].to(tl.int1)

    tmp14, tmp15 = triton_helpers.online_softmax_reduce(
        _tmp12_max, _tmp12_sum, 1, False)
    tmp14 = tmp14[:, None]
    tmp15 = tmp15[:, None]
    tmp12 = tmp14
    tmp13 = tmp15
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp17 = tl.load(in_ptr0 + (r0_3 + 324*x5), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_out_ptr0 + (r0_3 + 324*x0 + 104992*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tmp9 == 0
        tmp19 = tmp17 + tmp18
        tmp20 = tmp19 - tmp12
        tmp21 = tl_math.exp(tmp20)
        tmp22 = (tmp21 / tmp13)
        tmp23 = 0.0
        tmp24 = tl.where(tmp16, tmp23, tmp22)
        tl.store(in_out_ptr0 + (r0_3 + 324*x0 + 104992*x4), tmp24, r0_mask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 324), (324, 1))
    assert_size_stride(arg1_1, (32128, 384), (384, 1))
    assert_size_stride(arg2_1, (4, 324), (324, 1))
    assert_size_stride(arg3_1, (384, ), (1, ))
    assert_size_stride(arg4_1, (256, 384), (384, 1))
    assert_size_stride(arg5_1, (256, 384), (384, 1))
    assert_size_stride(arg6_1, (256, 384), (384, 1))
    assert_size_stride(arg7_1, (32, 8), (8, 1))
    assert_size_stride(arg8_1, (384, 256), (256, 1))
    assert_size_stride(arg9_1, (384, ), (1, ))
    assert_size_stride(arg10_1, (512, 384), (384, 1))
    assert_size_stride(arg11_1, (384, 512), (512, 1))
    assert_size_stride(arg12_1, (384, ), (1, ))
    assert_size_stride(arg13_1, (256, 384), (384, 1))
    assert_size_stride(arg14_1, (256, 384), (384, 1))
    assert_size_stride(arg15_1, (256, 384), (384, 1))
    assert_size_stride(arg16_1, (384, 256), (256, 1))
    assert_size_stride(arg17_1, (384, ), (1, ))
    assert_size_stride(arg18_1, (512, 384), (384, 1))
    assert_size_stride(arg19_1, (384, 512), (512, 1))
    assert_size_stride(arg20_1, (384, ), (1, ))
    assert_size_stride(arg21_1, (256, 384), (384, 1))
    assert_size_stride(arg22_1, (256, 384), (384, 1))
    assert_size_stride(arg23_1, (256, 384), (384, 1))
    assert_size_stride(arg24_1, (384, 256), (256, 1))
    assert_size_stride(arg25_1, (384, ), (1, ))
    assert_size_stride(arg26_1, (512, 384), (384, 1))
    assert_size_stride(arg27_1, (384, 512), (512, 1))
    assert_size_stride(arg28_1, (384, ), (1, ))
    assert_size_stride(arg29_1, (256, 384), (384, 1))
    assert_size_stride(arg30_1, (256, 384), (384, 1))
    assert_size_stride(arg31_1, (256, 384), (384, 1))
    assert_size_stride(arg32_1, (384, 256), (256, 1))
    assert_size_stride(arg33_1, (384, ), (1, ))
    assert_size_stride(arg34_1, (512, 384), (384, 1))
    assert_size_stride(arg35_1, (384, 512), (512, 1))
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
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((4, 324, 384), (124416, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, pow_1, mean, add, rsqrt, mul_1, mul_2], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_embedding_mean_mul_pow_rsqrt_0.run(arg0_1, arg1_1, arg3_1, buf1, 1296, 384, stream=stream0)
        del arg3_1
        buf2 = empty_strided_cuda((1296, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (1296, 384), (384, 1), 0), reinterpret_tensor(arg4_1, (384, 256), (1, 384), 0), out=buf2)
        del arg4_1
        buf4 = empty_strided_cuda((4, 8, 324, 32), (82944, 10368, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(buf2, buf4, 331776, stream=stream0)
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (1296, 384), (384, 1), 0), reinterpret_tensor(arg5_1, (384, 256), (1, 384), 0), out=buf3)
        del arg5_1
        buf11 = empty_strided_cuda((1296, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (1296, 384), (384, 1), 0), reinterpret_tensor(arg6_1, (384, 256), (1, 384), 0), out=buf11)
        del arg6_1
        buf5 = empty_strided_cuda((4, 8, 32, 324), (82944, 10368, 324, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf3, buf5, 1024, 324, stream=stream0)
        buf13 = reinterpret_tensor(buf3, (4, 8, 324, 32), (82944, 10368, 32, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(buf11, buf13, 331776, stream=stream0)
        buf6 = empty_strided_cuda((32, 324, 324), (104976, 324, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf4, (32, 324, 32), (10368, 32, 1), 0), reinterpret_tensor(buf5, (32, 32, 324), (10368, 324, 1), 0), out=buf6)
        buf7 = empty_strided_cuda((4, 8, 324, 324), (839936, 104992, 324, 1), torch.float32)
        buf12 = empty_strided_cuda((4, 8, 324, 324), (839936, 104992, 324, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to, sub, mul, add_3], Original ATen: [aten._to_copy, aten.rsub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mul_rsub_4.run(arg7_1, arg2_1, buf6, buf7, buf12, 10368, 324, stream=stream0)
        del arg2_1
        del arg7_1
        buf14 = reinterpret_tensor(buf5, (32, 324, 32), (10368, 32, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf12, (32, 324, 324), (104992, 324, 1), 0), reinterpret_tensor(buf13, (32, 324, 32), (10368, 32, 1), 0), out=buf14)
        buf15 = reinterpret_tensor(buf13, (4, 324, 8, 32), (82944, 256, 32, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf14, buf15, 331776, stream=stream0)
        buf16 = reinterpret_tensor(buf1, (1296, 384), (384, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (1296, 256), (256, 1), 0), reinterpret_tensor(arg8_1, (256, 384), (1, 256), 0), out=buf16)
        del arg8_1
        buf18 = empty_strided_cuda((4, 324, 384), (124416, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, add_4, pow_2, mean_1, add_5, rsqrt_1, mul_5, mul_6], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_6.run(arg0_1, arg1_1, buf16, arg9_1, buf18, 1296, 384, stream=stream0)
        del arg9_1
        buf19 = empty_strided_cuda((1296, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (1296, 384), (384, 1), 0), reinterpret_tensor(arg10_1, (384, 512), (1, 384), 0), out=buf19)
        del arg10_1
        buf20 = reinterpret_tensor(buf19, (4, 324, 512), (165888, 512, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_7.run(buf20, 663552, stream=stream0)
        buf21 = reinterpret_tensor(buf18, (1296, 384), (384, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (1296, 512), (512, 1), 0), reinterpret_tensor(arg11_1, (512, 384), (1, 512), 0), out=buf21)
        del arg11_1
        buf23 = empty_strided_cuda((4, 324, 384), (124416, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, add_4, add_6, pow_3, mean_2, add_7, rsqrt_2, mul_7, mul_8], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_8.run(arg0_1, arg1_1, buf16, buf21, arg12_1, buf23, 1296, 384, stream=stream0)
        del arg12_1
        buf24 = reinterpret_tensor(buf15, (1296, 256), (256, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (1296, 384), (384, 1), 0), reinterpret_tensor(arg13_1, (384, 256), (1, 384), 0), out=buf24)
        del arg13_1
        buf26 = reinterpret_tensor(buf14, (4, 8, 324, 32), (82944, 10368, 32, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(buf24, buf26, 331776, stream=stream0)
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (1296, 384), (384, 1), 0), reinterpret_tensor(arg14_1, (384, 256), (1, 384), 0), out=buf25)
        del arg14_1
        buf32 = reinterpret_tensor(buf4, (1296, 256), (256, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (1296, 384), (384, 1), 0), reinterpret_tensor(arg15_1, (384, 256), (1, 384), 0), out=buf32)
        del arg15_1
        buf27 = reinterpret_tensor(buf11, (4, 8, 32, 324), (82944, 10368, 324, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf25, buf27, 1024, 324, stream=stream0)
        buf34 = reinterpret_tensor(buf25, (4, 8, 324, 32), (82944, 10368, 32, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(buf32, buf34, 331776, stream=stream0)
        buf28 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf26, (32, 324, 32), (10368, 32, 1), 0), reinterpret_tensor(buf27, (32, 32, 324), (10368, 324, 1), 0), out=buf28)
        buf33 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_red_fused_9.run(buf28, buf7, buf33, 10368, 324, stream=stream0)
        buf35 = reinterpret_tensor(buf27, (32, 324, 32), (10368, 32, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf33, (32, 324, 324), (104992, 324, 1), 0), reinterpret_tensor(buf34, (32, 324, 32), (10368, 32, 1), 0), out=buf35)
        buf36 = reinterpret_tensor(buf34, (4, 324, 8, 32), (82944, 256, 32, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf35, buf36, 331776, stream=stream0)
        buf37 = reinterpret_tensor(buf23, (1296, 384), (384, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (1296, 256), (256, 1), 0), reinterpret_tensor(arg16_1, (256, 384), (1, 256), 0), out=buf37)
        del arg16_1
        buf38 = reinterpret_tensor(buf16, (4, 324, 384), (124416, 384, 1), 0); del buf16  # reuse
        buf40 = empty_strided_cuda((4, 324, 384), (124416, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, add_4, add_6, add_8, pow_4, mean_3, add_9, rsqrt_3, mul_9, mul_10], Original ATen: [aten.embedding, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_mean_mul_pow_rsqrt_10.run(buf38, arg0_1, arg1_1, buf21, buf37, arg17_1, buf40, 1296, 384, stream=stream0)
        del arg0_1
        del arg17_1
        del arg1_1
        buf41 = reinterpret_tensor(buf20, (1296, 512), (512, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (1296, 384), (384, 1), 0), reinterpret_tensor(arg18_1, (384, 512), (1, 384), 0), out=buf41)
        del arg18_1
        buf42 = reinterpret_tensor(buf41, (4, 324, 512), (165888, 512, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [relu_1], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_7.run(buf42, 663552, stream=stream0)
        buf43 = reinterpret_tensor(buf40, (1296, 384), (384, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (1296, 512), (512, 1), 0), reinterpret_tensor(arg19_1, (512, 384), (1, 512), 0), out=buf43)
        del arg19_1
        buf45 = reinterpret_tensor(buf37, (4, 324, 384), (124416, 384, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [add_10, pow_5, mean_4, add_11, rsqrt_4, mul_11, mul_12], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf38, buf43, arg20_1, buf45, 1296, 384, stream=stream0)
        del arg20_1
        buf46 = reinterpret_tensor(buf36, (1296, 256), (256, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (1296, 384), (384, 1), 0), reinterpret_tensor(arg21_1, (384, 256), (1, 384), 0), out=buf46)
        del arg21_1
        buf48 = reinterpret_tensor(buf35, (4, 8, 324, 32), (82944, 10368, 32, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(buf46, buf48, 331776, stream=stream0)
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (1296, 384), (384, 1), 0), reinterpret_tensor(arg22_1, (384, 256), (1, 384), 0), out=buf47)
        del arg22_1
        buf54 = reinterpret_tensor(buf26, (1296, 256), (256, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (1296, 384), (384, 1), 0), reinterpret_tensor(arg23_1, (384, 256), (1, 384), 0), out=buf54)
        del arg23_1
        buf49 = reinterpret_tensor(buf32, (4, 8, 32, 324), (82944, 10368, 324, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf47, buf49, 1024, 324, stream=stream0)
        buf56 = reinterpret_tensor(buf47, (4, 8, 324, 32), (82944, 10368, 32, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(buf54, buf56, 331776, stream=stream0)
        buf50 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf48, (32, 324, 32), (10368, 32, 1), 0), reinterpret_tensor(buf49, (32, 32, 324), (10368, 324, 1), 0), out=buf50)
        buf55 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_red_fused_9.run(buf50, buf7, buf55, 10368, 324, stream=stream0)
        buf57 = reinterpret_tensor(buf49, (32, 324, 32), (10368, 32, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf55, (32, 324, 324), (104992, 324, 1), 0), reinterpret_tensor(buf56, (32, 324, 32), (10368, 32, 1), 0), out=buf57)
        buf58 = reinterpret_tensor(buf56, (4, 324, 8, 32), (82944, 256, 32, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf57, buf58, 331776, stream=stream0)
        buf59 = reinterpret_tensor(buf45, (1296, 384), (384, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf58, (1296, 256), (256, 1), 0), reinterpret_tensor(arg24_1, (256, 384), (1, 256), 0), out=buf59)
        del arg24_1
        buf61 = reinterpret_tensor(buf21, (4, 324, 384), (124416, 384, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [add_10, add_12, pow_6, mean_5, add_13, rsqrt_5, mul_13, mul_14], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf38, buf43, buf59, arg25_1, buf61, 1296, 384, stream=stream0)
        del arg25_1
        buf62 = reinterpret_tensor(buf42, (1296, 512), (512, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (1296, 384), (384, 1), 0), reinterpret_tensor(arg26_1, (384, 512), (1, 384), 0), out=buf62)
        del arg26_1
        buf63 = reinterpret_tensor(buf62, (4, 324, 512), (165888, 512, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [relu_2], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_7.run(buf63, 663552, stream=stream0)
        buf64 = reinterpret_tensor(buf61, (1296, 384), (384, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (1296, 512), (512, 1), 0), reinterpret_tensor(arg27_1, (512, 384), (1, 512), 0), out=buf64)
        del arg27_1
        buf66 = empty_strided_cuda((4, 324, 384), (124416, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_10, add_12, add_14, pow_7, mean_6, add_15, rsqrt_6, mul_15, mul_16], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf38, buf43, buf59, buf64, arg28_1, buf66, 1296, 384, stream=stream0)
        del arg28_1
        buf67 = reinterpret_tensor(buf58, (1296, 256), (256, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (1296, 384), (384, 1), 0), reinterpret_tensor(arg29_1, (384, 256), (1, 384), 0), out=buf67)
        del arg29_1
        buf69 = reinterpret_tensor(buf57, (4, 8, 324, 32), (82944, 10368, 32, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(buf67, buf69, 331776, stream=stream0)
        buf68 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (1296, 384), (384, 1), 0), reinterpret_tensor(arg30_1, (384, 256), (1, 384), 0), out=buf68)
        del arg30_1
        buf75 = reinterpret_tensor(buf48, (1296, 256), (256, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (1296, 384), (384, 1), 0), reinterpret_tensor(arg31_1, (384, 256), (1, 384), 0), out=buf75)
        del arg31_1
        buf70 = reinterpret_tensor(buf54, (4, 8, 32, 324), (82944, 10368, 324, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf68, buf70, 1024, 324, stream=stream0)
        buf77 = reinterpret_tensor(buf68, (4, 8, 324, 32), (82944, 10368, 32, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(buf75, buf77, 331776, stream=stream0)
        buf71 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf69, (32, 324, 32), (10368, 32, 1), 0), reinterpret_tensor(buf70, (32, 32, 324), (10368, 324, 1), 0), out=buf71)
        buf76 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_red_fused_9.run(buf71, buf7, buf76, 10368, 324, stream=stream0)
        buf78 = reinterpret_tensor(buf70, (32, 324, 32), (10368, 32, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf76, (32, 324, 324), (104992, 324, 1), 0), reinterpret_tensor(buf77, (32, 324, 32), (10368, 32, 1), 0), out=buf78)
        buf79 = reinterpret_tensor(buf77, (4, 324, 8, 32), (82944, 256, 32, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [contiguous_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf78, buf79, 331776, stream=stream0)
        buf80 = reinterpret_tensor(buf66, (1296, 384), (384, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (1296, 256), (256, 1), 0), reinterpret_tensor(arg32_1, (256, 384), (1, 256), 0), out=buf80)
        del arg32_1
        buf81 = buf38; del buf38  # reuse
        buf83 = empty_strided_cuda((4, 324, 384), (124416, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_10, add_12, add_14, add_16, pow_8, mean_7, add_17, rsqrt_7, mul_17, mul_18], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_14.run(buf81, buf43, buf59, buf64, buf80, arg33_1, buf83, 1296, 384, stream=stream0)
        del arg33_1
        buf84 = reinterpret_tensor(buf63, (1296, 512), (512, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (1296, 384), (384, 1), 0), reinterpret_tensor(arg34_1, (384, 512), (1, 384), 0), out=buf84)
        del arg34_1
        buf85 = reinterpret_tensor(buf84, (4, 324, 512), (165888, 512, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [relu_3], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_7.run(buf85, 663552, stream=stream0)
        buf86 = reinterpret_tensor(buf83, (1296, 384), (384, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (1296, 512), (512, 1), 0), reinterpret_tensor(arg35_1, (512, 384), (1, 512), 0), out=buf86)
        del arg35_1
        buf88 = reinterpret_tensor(buf80, (4, 324, 384), (124416, 384, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [add_18, pow_9, mean_8, add_19, rsqrt_8, mul_19, mul_20], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_11.run(buf81, buf86, arg36_1, buf88, 1296, 384, stream=stream0)
        del arg36_1
        buf89 = reinterpret_tensor(buf79, (1296, 256), (256, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (1296, 384), (384, 1), 0), reinterpret_tensor(arg37_1, (384, 256), (1, 384), 0), out=buf89)
        del arg37_1
        buf91 = reinterpret_tensor(buf78, (4, 8, 324, 32), (82944, 10368, 32, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(buf89, buf91, 331776, stream=stream0)
        buf90 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (1296, 384), (384, 1), 0), reinterpret_tensor(arg38_1, (384, 256), (1, 384), 0), out=buf90)
        del arg38_1
        buf97 = reinterpret_tensor(buf69, (1296, 256), (256, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (1296, 384), (384, 1), 0), reinterpret_tensor(arg39_1, (384, 256), (1, 384), 0), out=buf97)
        del arg39_1
        buf92 = reinterpret_tensor(buf75, (4, 8, 32, 324), (82944, 10368, 324, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf90, buf92, 1024, 324, stream=stream0)
        buf99 = reinterpret_tensor(buf90, (4, 8, 324, 32), (82944, 10368, 32, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(buf97, buf99, 331776, stream=stream0)
        buf93 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf91, (32, 324, 32), (10368, 32, 1), 0), reinterpret_tensor(buf92, (32, 32, 324), (10368, 324, 1), 0), out=buf93)
        buf98 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_red_fused_9.run(buf93, buf7, buf98, 10368, 324, stream=stream0)
        buf100 = reinterpret_tensor(buf92, (32, 324, 32), (10368, 32, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf98, (32, 324, 324), (104992, 324, 1), 0), reinterpret_tensor(buf99, (32, 324, 32), (10368, 32, 1), 0), out=buf100)
        del buf98
        buf101 = reinterpret_tensor(buf99, (4, 324, 8, 32), (82944, 256, 32, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf100, buf101, 331776, stream=stream0)
        buf102 = reinterpret_tensor(buf88, (1296, 384), (384, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (1296, 256), (256, 1), 0), reinterpret_tensor(arg40_1, (256, 384), (1, 256), 0), out=buf102)
        del arg40_1
        buf104 = reinterpret_tensor(buf64, (4, 324, 384), (124416, 384, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [add_18, add_20, pow_10, mean_9, add_21, rsqrt_9, mul_21, mul_22], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_12.run(buf81, buf86, buf102, arg41_1, buf104, 1296, 384, stream=stream0)
        del arg41_1
        buf105 = reinterpret_tensor(buf85, (1296, 512), (512, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (1296, 384), (384, 1), 0), reinterpret_tensor(arg42_1, (384, 512), (1, 384), 0), out=buf105)
        del arg42_1
        buf106 = reinterpret_tensor(buf105, (4, 324, 512), (165888, 512, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [relu_4], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_7.run(buf106, 663552, stream=stream0)
        buf107 = reinterpret_tensor(buf104, (1296, 384), (384, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (1296, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 384), (1, 512), 0), out=buf107)
        del arg43_1
        del buf106
        buf109 = reinterpret_tensor(buf59, (4, 324, 384), (124416, 384, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [add_18, add_20, add_22, pow_11, mean_10, add_23, rsqrt_10, mul_23, mul_24], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_13.run(buf81, buf86, buf102, buf107, arg44_1, buf109, 1296, 384, stream=stream0)
        del arg44_1
        buf110 = reinterpret_tensor(buf101, (1296, 256), (256, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (1296, 384), (384, 1), 0), reinterpret_tensor(arg45_1, (384, 256), (1, 384), 0), out=buf110)
        del arg45_1
        buf112 = reinterpret_tensor(buf100, (4, 8, 324, 32), (82944, 10368, 32, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(buf110, buf112, 331776, stream=stream0)
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (1296, 384), (384, 1), 0), reinterpret_tensor(arg46_1, (384, 256), (1, 384), 0), out=buf111)
        del arg46_1
        buf118 = reinterpret_tensor(buf91, (1296, 256), (256, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (1296, 384), (384, 1), 0), reinterpret_tensor(arg47_1, (384, 256), (1, 384), 0), out=buf118)
        del arg47_1
        buf113 = reinterpret_tensor(buf97, (4, 8, 32, 324), (82944, 10368, 324, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf111, buf113, 1024, 324, stream=stream0)
        buf120 = reinterpret_tensor(buf111, (4, 8, 324, 32), (82944, 10368, 32, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(buf118, buf120, 331776, stream=stream0)
        del buf118
        buf114 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf112, (32, 324, 32), (10368, 32, 1), 0), reinterpret_tensor(buf113, (32, 32, 324), (10368, 324, 1), 0), out=buf114)
        del buf112
        buf119 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_red_fused_15.run(buf119, buf114, 10368, 324, stream=stream0)
        del buf114
        buf121 = reinterpret_tensor(buf113, (32, 324, 32), (10368, 32, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.bmm(reinterpret_tensor(buf119, (32, 324, 324), (104992, 324, 1), 0), reinterpret_tensor(buf120, (32, 324, 32), (10368, 32, 1), 0), out=buf121)
        del buf119
        buf122 = reinterpret_tensor(buf120, (4, 324, 8, 32), (82944, 256, 32, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf121, buf122, 331776, stream=stream0)
        del buf121
        buf123 = reinterpret_tensor(buf109, (1296, 384), (384, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (1296, 256), (256, 1), 0), reinterpret_tensor(arg48_1, (256, 384), (1, 256), 0), out=buf123)
        del arg48_1
        del buf122
        buf124 = buf81; del buf81  # reuse
        buf126 = reinterpret_tensor(buf43, (4, 324, 384), (124416, 384, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [add_18, add_20, add_22, add_24, pow_12, mean_11, add_25, rsqrt_11, mul_25, mul_26], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_14.run(buf124, buf86, buf102, buf107, buf123, arg49_1, buf126, 1296, 384, stream=stream0)
        del arg49_1
        del buf102
        del buf107
        del buf123
        del buf86
    return (buf126, buf124, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 324), (324, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((32128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 324), (324, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((384, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((384, 512), (512, 1), device='cuda:0', dtype=torch.float32)
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
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
