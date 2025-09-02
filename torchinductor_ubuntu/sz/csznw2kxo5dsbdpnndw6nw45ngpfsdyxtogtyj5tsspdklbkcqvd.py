# AOT ID: ['11_inference']
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/e4/ce4l2gmxok4krvzaiusxfjyzvjbk7fp2dw5qgzpzpxkcvgceg5ak.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_1,), kwargs = {})
triton_poi_fused_relu_0 = async_compile.triton('triton_poi_fused_relu_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1572864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_0(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/wk/cwkr7vvybohpohtnyds6w6rhham6nm2c4poasspesigelzetdmd4.py
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
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %view_3), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, %mul), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_1 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 526336}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/la/claemv32fqlvujzqgcxvvvdmn3ith7s5dli5nr6leneidikarcqt.py
# Topologically Sorted Source Nodes: [iadd], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   iadd => add_2
# Graph fragment:
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_15, %arg8_1), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_2, -1), kwargs = {})
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
# Topologically Sorted Source Nodes: [iadd, softmax], Original ATen: [aten.add, aten._softmax]
# Source node to ATen node mapping:
#   iadd => add_2
#   softmax => div
# Graph fragment:
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_15, %arg8_1), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_2), kwargs = {})
#   %exp_default_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %getitem_3), kwargs = {})
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
#   contiguous => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/y7/cy72xxe5idjvtsdcyyyppsvsjp5otdnulfikoduzl66coybssbxf.py
# Topologically Sorted Source Nodes: [add, add_2, pow_2, mean_1, add_3, rsqrt_1, mul_2, mul_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_2 => add_3
#   add_3 => add_4
#   mean_1 => mean_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %view_3), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_23), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_3, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg10_1, %mul_2), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=68, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '759C4B554222D749F0083B7FB1A70411C26A1536920BC9DEC9136159FEBEF13E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 657408}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel):
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


# kernel path: /mnt/disk1/khiemtt/universal-offline-bbo/torchinductor_ubuntu/wd/cwdhcmrlnsbjtobzmwwysl4wtplzsnpz5uqtgductf72pfm5ppxz.py
# Topologically Sorted Source Nodes: [add, add_2, add_4, pow_3, mean_2, add_5, rsqrt_2, mul_4, mul_5], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_2 => add_3
#   add_4 => add_5
#   add_5 => add_6
#   mean_2 => mean_2
#   mul_4 => mul_4
#   mul_5 => mul_5
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %view_3), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_23), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_27), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_5, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %rsqrt_2), kwargs = {})
#   %mul_5 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg13_1, %mul_4), kwargs = {})
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
#   add_2 => add_3
#   add_4 => add_5
#   add_6 => add_8
#   add_7 => add_9
#   mean_3 => mean_3
#   mul_6 => mul_6
#   mul_7 => mul_7
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %view_3), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_23), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_27), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_47), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_8, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %rsqrt_3), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg18_1, %mul_6), kwargs = {})
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 64, 512), (32768, 512, 1))
    assert_size_stride(arg1_1, (2048, 512), (512, 1))
    assert_size_stride(arg2_1, (512, 2048), (2048, 1))
    assert_size_stride(arg3_1, (1, 64, 512), (32768, 512, 1))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, 512), (512, 1))
    assert_size_stride(arg6_1, (512, 512), (512, 1))
    assert_size_stride(arg7_1, (512, 512), (512, 1))
    assert_size_stride(arg8_1, (1, 8, 64, 64), (8, 1, 512, 8))
    assert_size_stride(arg9_1, (512, 512), (512, 1))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (2048, 512), (512, 1))
    assert_size_stride(arg12_1, (512, 2048), (2048, 1))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, 512), (512, 1))
    assert_size_stride(arg15_1, (512, 512), (512, 1))
    assert_size_stride(arg16_1, (512, 512), (512, 1))
    assert_size_stride(arg17_1, (512, 512), (512, 1))
    assert_size_stride(arg18_1, (512, ), (1, ))
    assert_size_stride(arg19_1, (2048, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(arg0_1, (64, 512), (512, 1), 0), reinterpret_tensor(arg1_1, (512, 2048), (1, 512), 0), out=buf0)
        del arg0_1
        del arg1_1
        buf1 = reinterpret_tensor(buf0, (1, 64, 2048), (131072, 2048, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_0.run(buf1, 131072, stream=stream0)
        buf2 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (64, 2048), (2048, 1), 0), reinterpret_tensor(arg2_1, (2048, 512), (1, 2048), 0), out=buf2)
        del arg2_1
        buf4 = empty_strided_cuda((1, 64, 512), (32768, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, pow_1, mean, add_1, rsqrt, mul, mul_1], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_1.run(arg3_1, buf2, arg4_1, buf4, 64, 512, stream=stream0)
        del arg4_1
        buf5 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (64, 512), (512, 1), 0), reinterpret_tensor(arg5_1, (512, 512), (1, 512), 0), out=buf5)
        del arg5_1
        buf6 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (64, 512), (512, 1), 0), reinterpret_tensor(arg6_1, (512, 512), (1, 512), 0), out=buf6)
        del arg6_1
        buf7 = empty_strided_cuda((8, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf5, (8, 64, 64), (64, 512, 1), 0), reinterpret_tensor(buf6, (8, 64, 64), (64, 1, 512), 0), out=buf7)
        buf8 = empty_strided_cuda((1, 8, 64, 1), (512, 64, 1, 512), torch.float32)
        buf9 = empty_strided_cuda((1, 8, 64, 1), (512, 64, 1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [iadd], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_2.run(buf7, arg8_1, buf8, buf9, 8, 64, 64, stream=stream0)
        buf10 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (64, 512), (512, 1), 0), reinterpret_tensor(arg7_1, (512, 512), (1, 512), 0), out=buf10)
        del arg7_1
        buf11 = reinterpret_tensor(buf7, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [iadd, softmax], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_3.run(buf11, arg8_1, buf8, buf9, 8, 64, 64, stream=stream0)
        buf12 = reinterpret_tensor(buf4, (8, 64, 64), (4096, 64, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (8, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf10, (8, 64, 64), (64, 512, 1), 0), out=buf12)
        buf13 = reinterpret_tensor(buf11, (1, 64, 8, 64), (32768, 512, 64, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf12, buf13, 32768, stream=stream0)
        buf14 = reinterpret_tensor(buf12, (64, 512), (512, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (64, 512), (512, 1), 0), reinterpret_tensor(arg9_1, (512, 512), (1, 512), 0), out=buf14)
        del arg9_1
        buf16 = reinterpret_tensor(buf13, (1, 64, 512), (32768, 512, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [add, add_2, pow_2, mean_1, add_3, rsqrt_1, mul_2, mul_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_5.run(arg3_1, buf2, buf14, arg10_1, buf16, 64, 512, stream=stream0)
        del arg10_1
        buf17 = reinterpret_tensor(buf1, (64, 2048), (2048, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (64, 512), (512, 1), 0), reinterpret_tensor(arg11_1, (512, 2048), (1, 512), 0), out=buf17)
        del arg11_1
        buf18 = reinterpret_tensor(buf17, (1, 64, 2048), (131072, 2048, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [relu_1], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_0.run(buf18, 131072, stream=stream0)
        buf19 = reinterpret_tensor(buf16, (64, 512), (512, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (64, 2048), (2048, 1), 0), reinterpret_tensor(arg12_1, (2048, 512), (1, 2048), 0), out=buf19)
        del arg12_1
        buf21 = reinterpret_tensor(buf10, (1, 64, 512), (32768, 512, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [add, add_2, add_4, pow_3, mean_2, add_5, rsqrt_2, mul_4, mul_5], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_6.run(arg3_1, buf2, buf14, buf19, arg13_1, buf21, 64, 512, stream=stream0)
        del arg13_1
        buf22 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (64, 512), (512, 1), 0), reinterpret_tensor(arg14_1, (512, 512), (1, 512), 0), out=buf22)
        del arg14_1
        buf23 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (64, 512), (512, 1), 0), reinterpret_tensor(arg15_1, (512, 512), (1, 512), 0), out=buf23)
        del arg15_1
        buf24 = empty_strided_cuda((8, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf22, (8, 64, 64), (64, 512, 1), 0), reinterpret_tensor(buf23, (8, 64, 64), (64, 1, 512), 0), out=buf24)
        del buf22
        buf25 = buf9; del buf9  # reuse
        buf26 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [iadd_1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_2.run(buf24, arg8_1, buf25, buf26, 8, 64, 64, stream=stream0)
        buf27 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (64, 512), (512, 1), 0), reinterpret_tensor(arg16_1, (512, 512), (1, 512), 0), out=buf27)
        del arg16_1
        buf28 = reinterpret_tensor(buf24, (1, 8, 64, 64), (32768, 4096, 64, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [iadd_1, softmax_1], Original ATen: [aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_3.run(buf28, arg8_1, buf25, buf26, 8, 64, 64, stream=stream0)
        del arg8_1
        del buf25
        del buf26
        buf29 = reinterpret_tensor(buf21, (8, 64, 64), (4096, 64, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (8, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf27, (8, 64, 64), (64, 512, 1), 0), out=buf29)
        del buf27
        buf30 = reinterpret_tensor(buf28, (1, 64, 8, 64), (32768, 512, 64, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf29, buf30, 32768, stream=stream0)
        buf31 = reinterpret_tensor(buf29, (64, 512), (512, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (64, 512), (512, 1), 0), reinterpret_tensor(arg17_1, (512, 512), (1, 512), 0), out=buf31)
        del arg17_1
        buf32 = reinterpret_tensor(buf2, (1, 64, 512), (32768, 512, 1), 0); del buf2  # reuse
        buf34 = reinterpret_tensor(buf30, (1, 64, 512), (32768, 512, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [add, add_2, add_4, add_6, pow_4, mean_3, add_7, rsqrt_3, mul_6, mul_7], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_7.run(buf32, arg3_1, buf14, buf19, buf31, arg18_1, buf34, 64, 512, stream=stream0)
        del arg18_1
        del arg3_1
        del buf14
        del buf19
        del buf31
        buf35 = reinterpret_tensor(buf18, (64, 2048), (2048, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (64, 512), (512, 1), 0), reinterpret_tensor(arg19_1, (512, 2048), (1, 512), 0), out=buf35)
        del arg19_1
        del buf34
        buf36 = reinterpret_tensor(buf35, (1, 64, 2048), (131072, 2048, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [relu_2], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_0.run(buf36, 131072, stream=stream0)
    return (buf36, buf32, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 64, 512), (32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1, 64, 512), (32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((1, 8, 64, 64), (8, 1, 512, 8), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
