"""Owned progressive pixels and generation tracking for the native viewer."""

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

    def add(mut self, frame: CompletedFrame) -> Bool:
        """Accept only the current camera; weight batch means by sample count.
        """
        if frame.generation != self.generation:
            return False
        if len(self.linear) != len(frame.pixels):
            self.linear = List[Color](length=len(frame.pixels), fill=Color(0))
        for index in range(len(self.linear)):
            self.linear[index] += frame.pixels[index] * Float32(frame.samples)
        self.samples += frame.samples
        self.batches += 1
        self.rgba.resize(len(self.linear) * 4, 0)
        for index in range(len(self.linear)):
            var color = self.linear[index] / Float32(self.samples)
            self.rgba[4 * index] = color_to_byte(color.x)
            self.rgba[4 * index + 1] = color_to_byte(color.y)
            self.rgba[4 * index + 2] = color_to_byte(color.z)
            self.rgba[4 * index + 3] = 255
        return True
