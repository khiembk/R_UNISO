
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
