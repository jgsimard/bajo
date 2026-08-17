trait GpuScene:
    """Prepared, device-resident RT scene contract.

    Concrete implementations stay geometry- and layout-specialized so this
    abstraction adds no runtime dispatch to traversal or shading kernels.
    """

    comptime is_prepared_gpu_scene: Bool
