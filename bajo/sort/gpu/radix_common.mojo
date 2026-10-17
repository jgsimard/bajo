from std.atomic import Atomic, Ordering
from std.bit import count_trailing_zeros
from std.math import ceildiv, max
from std.memory import stack_allocation, bitcast
from std.sys.info import bit_width_of
from max.gpu import thread_idx, block_idx, lane_id, grid_dim, WARP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from .utils import DoubleBuffer

comptime ordering = Ordering.RELAXED


@always_inline
def init_radix_workspace[
    NUM_PASSES: Int, BLOCK_SIZE: Int, count_dtype: DType = .int32
](
    global_hist: Pointer[UInt32, MutAnyOrigin],
    pass_hist: Pointer[UInt32, MutAnyOrigin],
    index: Pointer[UInt32, MutAnyOrigin],
    pass_hist_size: Scalar[count_dtype],
):
    var idx = block_idx.x * BLOCK_SIZE + thread_idx.x
    if idx < Int(pass_hist_size):
        pass_hist[unsafe_offset=idx] = 0
    if idx < NUM_PASSES * 256:
        global_hist[unsafe_offset=idx] = 0
    if idx < NUM_PASSES:
        index[unsafe_offset=idx] = 0


def global_histogram[
    keys_dtype: DType,
    *,
    BLOCK_SIZE: Int,
    RADIX: Int,
    VEC_WIDTH: SIMDLength,
    ITEMS_PER_THREAD: Int,
](
    sort: Pointer[Scalar[keys_dtype], MutAnyOrigin],
    global_hist: Pointer[UInt32, MutAnyOrigin],
    size: Int32,
):
    var size_int = Int(size)
    comptime BYTES_PER_KEY = bit_width_of[keys_dtype]() / 8
    comptime PART_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD
    comptime THREADS_PER_PARTITION = 64
    comptime NUM_PARTITIONS = BLOCK_SIZE / THREADS_PER_PARTITION
    comptime PADDED_RADIX = RADIX * NUM_PARTITIONS
    comptime assert BLOCK_SIZE > 0 and BLOCK_SIZE % THREADS_PER_PARTITION == 0
    comptime assert ITEMS_PER_THREAD > 0 and PART_SIZE % VEC_WIDTH == 0

    var tid = thread_idx.x
    var bid = block_idx.x

    var s_hists = stack_allocation[
        PADDED_RADIX * BYTES_PER_KEY, UInt32, address_space=AddressSpace.SHARED
    ]()

    for i in range(tid, PADDED_RADIX * BYTES_PER_KEY, BLOCK_SIZE):
        s_hists[unsafe_offset=i] = 0
    barrier()

    # 64 threads : 1 histogram in shared memory
    var wave_offset = tid / THREADS_PER_PARTITION * RADIX

    def _accumulate_hist[width: SIMDLength](i: Int) {imm}:
        var _t = sort.unsafe_load[width=width](i)
        var t = bitcast[.uint8, width * BYTES_PER_KEY](_t)

        comptime for v in range(width):
            comptime for b in range(BYTES_PER_KEY):
                var byte_val = Int(t[v * BYTES_PER_KEY + b])
                var s_idx = b * PADDED_RADIX + wave_offset + byte_val
                _ = Atomic.fetch_add[ordering=ordering](
                    s_hists.unsafe_offset(s_idx), 1
                )

    var block_start = bid * PART_SIZE
    if block_start + PART_SIZE <= size_int:
        for i in range(
            block_start + tid * VEC_WIDTH,
            block_start + PART_SIZE,
            BLOCK_SIZE * VEC_WIDTH,
        ):
            _accumulate_hist[VEC_WIDTH](i)
    else:
        for i in range(block_start + tid, size_int, BLOCK_SIZE):
            _accumulate_hist[1](i)
    barrier()

    # Reduce
    for i in range(tid, RADIX, BLOCK_SIZE):
        comptime for b in range(BYTES_PER_KEY):
            var sum: UInt32 = 0
            comptime for p in range(NUM_PARTITIONS):
                sum += s_hists[unsafe_offset=b * PADDED_RADIX + p * RADIX + i]
            var g_idx = (b * RADIX) + i
            _ = Atomic.fetch_add[ordering=ordering](
                global_hist.unsafe_offset(g_idx), sum
            )


@always_inline
def scan_local_histogram[
    histogram_origin: MutOrigin, //, BITS_PER_PASS: Int
](
    s_local_histogram: Pointer[
        UInt32, histogram_origin, address_space=AddressSpace.SHARED
    ],
):
    comptime RADIX = 2**BITS_PER_PASS
    comptime LANE_LOG = count_trailing_zeros(WARP_SIZE)
    var tid = thread_idx.x
    var lid = lane_id()
    var idx = tid << LANE_LOG
    var val: UInt32 = 0
    if tid < (RADIX >> LANE_LOG):
        val = s_local_histogram[unsafe_offset=idx]
    var exc = warp.prefix_sum[exclusive=True](val)
    if tid < (RADIX >> LANE_LOG):
        s_local_histogram[unsafe_offset=idx] = exc
    barrier()

    if tid < RADIX and lid > 0:
        s_local_histogram[unsafe_offset=tid] += s_local_histogram[
            unsafe_offset=tid - lid
        ]
    barrier()


@always_inline
def scatter_radix_tile[
    values_origin: MutOrigin,
    //,
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    BITS_PER_PASS: Int,
    BLOCK_SIZE: Int,
    KEYS_PER_THREAD: Int,
    HAVE_PAYLOAD: Bool,
](
    keys_alternate: Pointer[Scalar[keys_dtype], MutAnyOrigin],
    vals_current_opt: Optional[Pointer[Scalar[vals_dtype], MutAnyOrigin]],
    vals_alternate_opt: Optional[Pointer[Scalar[vals_dtype], MutAnyOrigin]],
    s_keys: Pointer[Scalar[keys_dtype], _, address_space=AddressSpace.SHARED],
    s_values: Pointer[
        Scalar[vals_dtype], values_origin, address_space=AddressSpace.SHARED
    ],
    s_local_histogram: Pointer[UInt32, _, address_space=AddressSpace.SHARED],
    offsets: Array[UInt32, KEYS_PER_THREAD],
    partition_index: Int,
    size: Int32,
    radix_shift: UInt32,
):
    var size_int = Int(size)
    comptime RADIX_MASK = Scalar[keys_dtype](2**BITS_PER_PASS - 1)
    comptime BIN_PART_SIZE = BLOCK_SIZE * KEYS_PER_THREAD
    var tid = thread_idx.x
    var gdim = grid_dim.x
    var BIN_PART_START = partition_index * BIN_PART_SIZE
    var t_base = (
        lane_id()
        + (tid / WARP_SIZE) * WARP_SIZE * KEYS_PER_THREAD
        + BIN_PART_START
    )
    # Scatter to Device Memory
    var part_size = (
        BIN_PART_SIZE if partition_index
        < gdim - 1 else size_int - BIN_PART_START
    )
    comptime if HAVE_PAYLOAD:
        debug_assert["safe", _use_compiler_assume=True](Bool(vals_current_opt))
        debug_assert["safe", _use_compiler_assume=True](
            Bool(vals_alternate_opt)
        )
        var vals_current = vals_current_opt.unsafe_value()
        var vals_alternate = vals_alternate_opt.unsafe_value()

        var vals = Array[Scalar[vals_dtype], KEYS_PER_THREAD](
            uninitialized=True
        )
        var digits = Array[UInt8, KEYS_PER_THREAD](uninitialized=True)

        comptime for i in range(KEYS_PER_THREAD):
            var t_idx = tid + i * BLOCK_SIZE
            if t_idx < part_size:
                var key = s_keys[unsafe_offset=t_idx]
                var d = (key >> Scalar[keys_dtype](radix_shift)) & RADIX_MASK
                digits[i] = d.cast[.uint8]()
                keys_alternate[
                    unsafe_offset=s_local_histogram[unsafe_offset=Int(d)]
                    + UInt32(t_idx)
                ] = key
        barrier()

        var t_payload = t_base
        comptime for i in range(KEYS_PER_THREAD):
            if t_payload < size_int:
                vals[i] = vals_current[unsafe_offset=t_payload]
            else:
                vals[i] = Scalar[vals_dtype].MAX
            t_payload += WARP_SIZE

        comptime for i in range(KEYS_PER_THREAD):
            s_values[unsafe_offset=Int(offsets[i])] = vals[i]
        barrier()

        comptime for i in range(KEYS_PER_THREAD):
            var t_idx = tid + i * BLOCK_SIZE
            if t_idx < part_size:
                var d = Int(digits[i])
                vals_alternate[
                    unsafe_offset=s_local_histogram[unsafe_offset=d]
                    + UInt32(t_idx)
                ] = s_values[unsafe_offset=t_idx]
    else:
        for i in range(tid, part_size, BLOCK_SIZE):
            var key = s_keys[unsafe_offset=i]
            var digit = Int(
                (key >> Scalar[keys_dtype](radix_shift)) & RADIX_MASK
            )
            var dst = s_local_histogram[unsafe_offset=digit] + UInt32(i)
            keys_alternate[unsafe_offset=dst] = key


comptime RadixScanArgs = TypeList.of[
    Pointer[UInt32, MutAnyOrigin],
    Pointer[UInt32, MutAnyOrigin],
    Int32,
]()
comptime RadixBinningArgs[keys_dtype: DType, vals_dtype: DType] = TypeList.of[
    Pointer[Scalar[keys_dtype], MutAnyOrigin],
    Pointer[Scalar[keys_dtype], MutAnyOrigin],
    Optional[Pointer[Scalar[vals_dtype], MutAnyOrigin]],
    Optional[Pointer[Scalar[vals_dtype], MutAnyOrigin]],
    Pointer[UInt32, MutAnyOrigin],
    Pointer[UInt32, MutAnyOrigin],
    Int32,
    UInt32,
]()

comptime RadixScanFn = def(* args: * RadixScanArgs) thin
comptime RadixBinningFn[keys_dtype: DType, vals_dtype: DType] = def(
    * args: * RadixBinningArgs[keys_dtype, vals_dtype]
) thin


def enqueue_radix_passes[
    keys_origin: MutOrigin,
    keys_alternate_origin: MutOrigin,
    values_origin: MutOrigin,
    values_alternate_origin: MutOrigin,
    global_hist_origin: MutOrigin,
    pass_hist_origin: MutOrigin,
    index_origin: MutOrigin,
    //,
    keys_dtype: DType,
    vals_dtype: DType,
    *,
    BLOCK_SIZE: Int,
    KEYS_PER_THREAD: Int,
    HAVE_PAYLOAD: Bool,
    HISTOGRAM_WORDS: Int,
    G_HIST_TPB: Int,
    G_HIST_ITEMS_PER_THREAD: Int,
    VEC_WIDTH: SIMDLength,
    scan_fn: RadixScanFn,
    binning_fn: RadixBinningFn[keys_dtype, vals_dtype],
](
    ctx: DeviceContext,
    keys: Pointer[Scalar[keys_dtype], keys_origin],
    keys_alternate: Pointer[Scalar[keys_dtype], keys_alternate_origin],
    vals_current: Optional[Pointer[Scalar[vals_dtype], values_origin]],
    vals_alternate: Optional[
        Pointer[Scalar[vals_dtype], values_alternate_origin]
    ],
    global_hist: Pointer[UInt32, global_hist_origin],
    pass_hist: Pointer[UInt32, pass_hist_origin],
    index: Pointer[UInt32, index_origin],
    size: Int,
) raises:
    """Enqueue validated inputs without changing the selected look-back protocol.
    """
    comptime NUM_PASSES = bit_width_of[keys_dtype]() / 8
    comptime assert NUM_PASSES % 2 == 0
    comptime assert HISTOGRAM_WORDS == 1 or HISTOGRAM_WORDS == 2
    comptime count_dtype: DType = .int64 if HISTOGRAM_WORDS == 2 else .int32
    var binning_blocks = ceildiv(size, BLOCK_SIZE * KEYS_PER_THREAD)
    var hist_blocks = binning_blocks + 1
    var pass_hist_size = hist_blocks * 256 * NUM_PASSES * HISTOGRAM_WORDS
    ctx.enqueue_function[init_radix_workspace[NUM_PASSES, 256, count_dtype]](
        global_hist,
        pass_hist,
        index,
        Scalar[count_dtype](pass_hist_size),
        grid_dim=ceildiv(max(pass_hist_size, NUM_PASSES * 256), 256),
        block_dim=256,
    )
    ctx.enqueue_function[
        global_histogram[
            keys_dtype,
            BLOCK_SIZE=G_HIST_TPB,
            RADIX=256,
            VEC_WIDTH=VEC_WIDTH,
            ITEMS_PER_THREAD=G_HIST_ITEMS_PER_THREAD,
        ]
    ](
        keys,
        global_hist,
        Int32(size),
        grid_dim=ceildiv(size, G_HIST_TPB * G_HIST_ITEMS_PER_THREAD),
        block_dim=G_HIST_TPB,
    )
    ctx.enqueue_function[declared_arg_types=RadixScanArgs, func=scan_fn](
        global_hist,
        pass_hist,
        Int32(hist_blocks),
        grid_dim=NUM_PASSES,
        block_dim=256,
    )
    var db_keys = DoubleBuffer(keys, keys_alternate)
    # Payload buffers have different origins. Their launch pointers use one
    # fixed type while the borrowed buffers remain alive through this sequence.
    var current_values = Optional[Pointer[Scalar[vals_dtype], MutAnyOrigin]]()
    var alternate_values = Optional[Pointer[Scalar[vals_dtype], MutAnyOrigin]]()
    comptime if HAVE_PAYLOAD:
        current_values = Optional(
            vals_current.unsafe_value().unsafe_origin_cast[MutAnyOrigin]()
        )
        alternate_values = Optional(
            vals_alternate.unsafe_value().unsafe_origin_cast[MutAnyOrigin]()
        )
    comptime for pass_idx in range(NUM_PASSES):
        ctx.enqueue_function[
            declared_arg_types=RadixBinningArgs[keys_dtype, vals_dtype],
            func=binning_fn,
        ](
            db_keys.current,
            db_keys.alternate,
            current_values,
            alternate_values,
            pass_hist.unsafe_offset(
                pass_idx * hist_blocks * 256 * HISTOGRAM_WORDS
            ),
            index,
            Int32(size),
            UInt32(pass_idx * 8),
            grid_dim=binning_blocks,
            block_dim=BLOCK_SIZE,
        )
        db_keys.swap()
        comptime if HAVE_PAYLOAD:
            current_values, alternate_values = alternate_values, current_values
