"""GPU light-record offsets shared by upload, RGB and spectral sampling.

Offsets are Float32 elements within each record. The packed field buffer begins
with a dense selection-probability prefix, followed by STRIDE-sized records.
"""


struct GPU_RT_LIGHT:
    comptime STRIDE = 22
    comptime P0_X = 0
    comptime P0_Y = 1
    comptime P0_Z = 2
    comptime P1_X = 3
    comptime P1_Y = 4
    comptime P1_Z = 5
    comptime P2_X = 6
    comptime P2_Y = 7
    comptime P2_Z = 8
    comptime RADIUS = 9
    comptime E_X = 10
    comptime E_Y = 11
    comptime E_Z = 12
    # The selection hot path reads the duplicated dense probability prefix.
    comptime ALIAS_PROBABILITY = 13
    comptime UV0 = 14
    comptime PRIMITIVE = 20
    comptime FLAGS = 21
