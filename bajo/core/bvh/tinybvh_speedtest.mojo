# Mojo implementation of tinybvh


comptime INST_IDX_BITS = 10
comptime SCRWIDTH = 480
comptime SCRHEIGHT = 320


comptime _DEFAULT = 1
comptime _BVH = 2
comptime _SWEEP = 3
comptime _VERBOSE = 4
comptime _DOUBLE = 5
comptime _SOA = 6
comptime _GPU2 = 7
comptime _BVH4 = 8
comptime _CPU4 = 9
comptime _ALT4 = 10
comptime _CPU4A = 11
comptime _CPU8 = 12
comptime _OPT8 = 13
comptime _GPU4 = 14
comptime _BVH8 = 15
comptime _CWBVH = 16


def TestPrimaryRays(layout: Int) -> Float32:
    return -1.0


def main() raises:
    print("tinybvh_speedtest hello")
