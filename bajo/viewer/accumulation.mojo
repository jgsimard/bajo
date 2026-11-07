"""Owned progressive pixels and generation tracking for the native viewer."""

from std.sys.info import size_of

from bajo.core.color import Color
from bajo.rt.image_output import color_to_byte
from .worker import CompletedFrame


struct FrameAccumulator:
    var generation: UInt64
    var samples: Int
    var batches: Int
    var linear: List[Color]
    var rgba: List[UInt8]

    def __init__(out self):
        self.generation = 0
        self.samples = 0
        self.batches = 0
        self.linear = List[Color]()
        self.rgba = List[UInt8]()

    def reset(mut self):
        self.generation += 1
        self.samples = 0
        self.batches = 0
        self.linear.clear()
        self.rgba.clear()

    def take_frame(mut self, mut frame: CompletedFrame) -> Bool:
        """Accept a frame, adopting its owned preview buffer when present."""
        if frame.generation != self.generation:
            return False
        if len(frame.rgba) == 0:
            return self.add(frame)
        self.linear.clear()
        self.rgba = frame.rgba^
        frame.rgba = List[UInt8]()
        self.samples = frame.preview_samples
        self.batches = frame.preview_batches
        return True

    def add(mut self, frame: CompletedFrame) -> Bool:
        """Accept only the current camera; weight batch means by sample count.
        """
        if frame.generation != self.generation:
            return False
        if len(frame.rgba) != 0:
            self.linear.clear()
            self.rgba.resize(unsafe_uninit_length=len(frame.rgba))
            Span(self.rgba).copy_from(frame.rgba)
            self.samples = frame.preview_samples
            self.batches = frame.preview_batches
            return True
        if len(self.linear) != len(frame.pixels):
            self.linear = List[Color](length=len(frame.pixels), fill=Color(0))
        self.samples += frame.samples
        self.batches += 1
        self.rgba.resize(unsafe_uninit_length=len(self.linear) * 4)
        var batch_samples = Float32(frame.samples)
        var total_samples = Float32(self.samples)
        comptime assert size_of[Color]() == 3 * size_of[Float32]()
        var source = frame.pixels.unsafe_ptr().unsafe_bitcast[Float32]()
        var linear = self.linear.unsafe_ptr().unsafe_bitcast[Float32]()
        var rgba = self.rgba.unsafe_ptr()
        var pixel = 0
        # Six remaining pixels bound both 16-float loads. Only the first
        # twelve lanes are stored, so each group of four pixels accumulates
        # exactly once. The scalar tail handles the last two to five pixels.
        while len(self.linear) - pixel >= 6:
            var offset = pixel * 3
            var values = (
                linear.unsafe_offset(offset).unsafe_load[
                    width=16, alignment=1
                ]()
                + source.unsafe_offset(offset).unsafe_load[
                    width=16, alignment=1
                ]()
                * batch_samples
            )
            linear.unsafe_offset(offset).unsafe_store[width=8, alignment=1](
                values.slice[8]()
            )
            linear.unsafe_offset(offset + 8).unsafe_store[width=4, alignment=1](
                values.slice[4, offset=8]()
            )
            var rgb = color_to_byte(values / total_samples)
            var packed = rgb.shuffle[
                0, 1, 2, 16, 3, 4, 5, 16, 6, 7, 8, 16, 9, 10, 11, 16
            ](SIMD[.uint8, 16](255))
            rgba.unsafe_offset(pixel * 4).unsafe_store[width=16, alignment=1](
                packed
            )
            pixel += 4
        for index in range(pixel, len(self.linear)):
            self.linear[index] += frame.pixels[index] * batch_samples
            var color = self.linear[index] / total_samples
            self.rgba[4 * index] = color_to_byte(color.x)
            self.rgba[4 * index + 1] = color_to_byte(color.y)
            self.rgba[4 * index + 2] = color_to_byte(color.z)
            self.rgba[4 * index + 3] = 255
        return True
