
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
