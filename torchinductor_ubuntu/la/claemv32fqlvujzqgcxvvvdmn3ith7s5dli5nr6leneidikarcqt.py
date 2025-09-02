
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
