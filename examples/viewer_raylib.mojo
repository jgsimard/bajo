"""Native raylib window/input/rasterization and Bajo CPU/GPU tracing."""

from bajo.viewer.native import parse_options, run_viewer


def main() raises:
    run_viewer(parse_options())
