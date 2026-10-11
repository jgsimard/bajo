"""Optical constants for the RGB conductor model used by PBRT ingestion."""

from bajo.rt.render_types import Color


def _named_optical_constant(name: ImmStringSpan) raises -> Color:
    # RGB approximation at 610, 550, and 460 nm, linearly interpolated from
    # PBRT v4's Al_eta/Al_k wavelength tables (Apache-2.0):
    # https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/spectrum.cpp
    # These are optical constants, not RGB reflectance values.
    __match name:
        case "metal-Al-eta":
            return Color(1.24750867, 0.95856902, 0.64627187)
        case "metal-Al-k":
            return Color(7.36845593, 6.68696769, 5.59014846)
        case _:
            raise Error("unsupported PBRT optical spectrum: " + String(name))
