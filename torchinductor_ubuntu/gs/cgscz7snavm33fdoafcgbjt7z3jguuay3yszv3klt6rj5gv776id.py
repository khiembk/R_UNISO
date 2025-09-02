
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
