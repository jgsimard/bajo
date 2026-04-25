from std.algorithm import vectorize, parallelize
from std.benchmark import run, Unit, keep
from std.math import clamp
from std.memory import UnsafePointer
from std.sys import simd_width_of
from std.utils import IndexList, StaticTuple

from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout.math import max as lmax
from tensor import InputTensor, OutputTensor, foreach


comptime system_fn[dtype: DType, layout: Layout] = def(
    dy: LayoutTensor[dtype, layout, MutAnyOrigin],
    y: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    t: Scalar[dtype],
) thin -> None


struct Euler[
    dtype: DType where dtype.is_floating_point(),
    layout: Layout,
    //,
    system: system_fn[dtype, layout],
]:
    comptime LT = LayoutTensor[Self.dtype, Self.layout, MutAnyOrigin]
    var y: Self.LT
    var t: Scalar[Self.dtype]
    var dt: Scalar[Self.dtype]
    var dy: Self.LT

    def __init__(
        out self,
        u0: Self.LT,
        t_start: Scalar[Self.dtype],
        dt: Scalar[Self.dtype],
    ):
        self.y = u0
        self.t = t_start
        self.dt = dt
        self.dy = Self.LT.stack_allocation()

    def step(mut self):
        Self.system(self.dy, self.y, self.t)

        # y = y + dt * dy
        self.y += self.dy * self.dt
        self.t += self.dt


# # parametrized trait would solve this
# trait Tableau:
#     comptime dtype: DType
#     comptime N_STAGES: Int
#     comptime IS_FSAL: Bool
#     comptime N_STAGES_FSAL = Self.N_STAGES - (1 if Self.IS_FSAL else 0)
#     comptime _a: StaticTuple[Scalar[Self.dtype], Self.N_STAGES * Self.N_STAGES]
#     comptime b: StaticTuple[Scalar[Self.dtype], Self.N_STAGES]
#     comptime c: StaticTuple[Scalar[Self.dtype], Self.N_STAGES]
#     comptime e: StaticTuple[Scalar[Self.dtype], Self.N_STAGES]

#     @staticmethod
#     def a(i: Int, j: Int) -> Scalar[Self.dtype]:
#         return Self._a[i * Self.N_STAGES + j]


# struct Tsit5Tableau(Tableau):
#     comptime dtype = dtype
#     comptime N_STAGES: Int = 7
#     comptime IS_FSAL: Bool = True
#     # comptime N_STAGES_FSAL = Self.N_STAGES - (1 if Self.IS_FSAL else 0)
#     comptime _a = StaticTuple[Scalar[Self.dtype], 49](
#         0, 0, 0, 0, 0, 0, 0, # Row 0
#         0.161, 0, 0, 0, 0, 0, 0, # Row 1
#         -0.008480655492356989, 0.335480655492357, 0, 0, 0, 0, 0, # Row 2
#         2.8971530571054935, -6.359448489975075, 4.3622954328695815, 0, 0, 0, 0, # Row 3
#         5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525, 0, 0, 0, # Row 4
#         5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383, 0, 0, # Row 5
#         0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0 # Row 6
#     )
#     comptime b = StaticTuple[Scalar[Self.dtype], 7](
#         0.09646076681806523, 0.01, 0.4798896504144996,
#         1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0
#     )
#     comptime c = StaticTuple[Scalar[Self.dtype], 7](0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0)
#     # comptime c : InlineArray[Scalar[Self.dtype], 7] =  [0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0]
#     comptime e = StaticTuple[Scalar[Self.dtype], 7](
#         -0.00178001105222577714, -0.0008164344596567469, 0.007880878010261995,
#         -0.1447110071732629, 0.5823571654525552, -0.45808210592918697, 1.0/66.0
#     )


struct Tsit5[
    dtype: DType,
    layout: Layout,
    //,
    system: system_fn[dtype, layout],
    adaptive: Bool,
]:
    # fmt: off
    comptime _a = StaticTuple[Self.S, 49](
        0, 0, 0, 0, 0, 0, 0, # Row 0
        0.161, 0, 0, 0, 0, 0, 0, # Row 1
        -0.008480655492356989, 0.335480655492357, 0, 0, 0, 0, 0, # Row 2
        2.8971530571054935, -6.359448489975075, 4.3622954328695815, 0, 0, 0, 0, # Row 3
        5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525, 0, 0, 0, # Row 4
        5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383, 0, 0, # Row 5
        0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0 # Row 6
    )
    comptime b = StaticTuple[Self.S, 7](
        0.09646076681806523, 0.01, 0.4798896504144996,
        1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0
    )
    comptime c = StaticTuple[Self.S, 7](0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0)
    # comptime c : InlineArray[Self.S, 7] =  [0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0]
    comptime e = StaticTuple[Self.S, 7](
        -0.00178001105222577714, -0.0008164344596567469, 0.007880878010261995,
        -0.1447110071732629, 0.5823571654525552, -0.45808210592918697, 1.0/66.0
    )
    # fmt: on
    comptime N_STAGES = 7
    comptime IS_FSAL = True
    comptime N_STAGES_FSAL = Self.N_STAGES - (1 if Self.IS_FSAL else 0)

    comptime LT = LayoutTensor[Self.dtype, Self.layout, MutAnyOrigin]
    comptime S = Scalar[Self.dtype]

    var y: Self.LT
    var t: Self.S
    var dt: Self.S
    var u_modified: Bool

    var qold: Self.S
    var abstol: Self.S
    var reltol: Self.S

    var ks: InlineArray[Self.LT, Self.N_STAGES]
    var tmp: Self.LT
    var y_next: Self.LT

    def __init__(
        out self,
        u0: Self.LT,
        t_start: Scalar[Self.dtype],
        dt: Scalar[Self.dtype],
        qold: Scalar[Self.dtype] = 1e-4,
        abstol: Scalar[Self.dtype] = 1e-6,
        reltol: Scalar[Self.dtype] = 1e-3,
    ):
        self.y = u0
        self.t = t_start
        self.dt = dt
        self.u_modified = True
        self.qold = qold
        self.abstol = abstol
        self.reltol = reltol

        self.ks = InlineArray[Self.LT, Self.N_STAGES](uninitialized=True)
        size = u0.size()

        for i in range(Self.N_STAGES):
            self.ks[i] = Self.LT(alloc[Self.S](size))

        self.tmp = Self.LT(alloc[Self.S](size))
        self.y_next = Self.LT(alloc[Self.S](size))

    def __del__(deinit self):
        for i in range(Self.N_STAGES):
            self.ks[i].ptr.free()
        self.tmp.ptr.free()
        self.y_next.ptr.free()

    @staticmethod
    def a(i: Int, j: Int) -> Scalar[Self.dtype]:
        return Self._a[i * Self.N_STAGES + j]

    def step(mut self):
        if self.u_modified:
            Self.system(self.ks[0], self.y, self.t)
            self.u_modified = False

        else:
            comptime if Self.IS_FSAL:
                # FSAL requires a deep copy of the last derivative into the first slot
                self.ks[0].copy_from(self.ks[Self.N_STAGES_FSAL])
            else:
                Self.system(self.ks[0], self.y, self.t)

        comptime if self.adaptive:
            accepted = False
            while not accepted:
                h = self.dt
                self._compute_stages(h)

                # Compute y_next using weights b (sum of stages 1 to S-1)
                self.y_next.copy_from(self.y)
                weight_sum = self.ks[0] * Self.b[0]

                comptime for i in range(1, Self.N_STAGES_FSAL):
                    weight_sum += self.ks[i] * Self.b[i]
                self.y_next += weight_sum * h

                comptime if Self.IS_FSAL:
                    # Calculate final stage derivative at the proposed y_next
                    Self.system(
                        self.ks[Self.N_STAGES_FSAL], self.y_next, self.t + h
                    )

                # e_est = self._estimate_error(h)
                e_est = self._estimate_error(h)

                comptime beta1 = 7.0 / 50.0
                comptime beta2 = 2.0 / 25.0
                if e_est <= 1.0:
                    self.y.copy_from(self.y_next)
                    self.t += h
                    # PI-Adaptive step size control
                    q = (
                        pow(e_est, beta1) / pow(self.qold, beta2) if e_est
                        > 0 else 0.1
                    )
                    self.dt = h * clamp(0.9 / q, 0.1, 5.0)
                    self.qold = max(e_est, 1e-4)
                    accepted = True
                else:
                    # Step rejected: reduce dt and retry
                    self.dt = h * max(0.9 / pow(e_est, beta1), 0.1)
        else:
            # Fixed step logic
            h = self.dt
            self._compute_stages(h)

            self.y_next.copy_from(self.y)
            weight_sum = self.ks[0] * Self.b[0]

            comptime for i in range(1, Self.N_STAGES_FSAL):
                weight_sum += self.ks[i] * Self.b[i]

            self.y += weight_sum * h
            self.t += h

            comptime if Self.IS_FSAL:
                Self.system(self.ks[Self.N_STAGES_FSAL], self.y, self.t)

    def _compute_stages(mut self, h: Scalar[Self.dtype]):
        for i in range(1, Self.N_STAGES_FSAL):
            self.tmp.copy_from(self.y)

            for j in range(i):
                self.tmp += self.ks[j] * Self.a(i, j) * h

            Self.system(self.ks[i], self.tmp, self.t + Self.c[i] * h)

    def _estimate_error(self, h: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        comptime SIMD_WIDTH = simd_width_of[Self.dtype]()

        e_est: SIMD[Self.dtype, 1] = 0.0

        def compute[w: Int](i: Int) {read self, mut e_est, read h}:
            err_v = SIMD[Self.dtype, w](0.0)

            comptime for s in range(Self.N_STAGES):
                comptime e_coeff = Self.e[s]

                comptime if e_coeff != 0:
                    err_v += self.ks[s].ptr.load[width=w](i) * e_coeff

            ev = err_v * h
            y = self.y.ptr.load[width=w](i)
            yn = self.y_next.ptr.load[width=w](i)
            sc = self.abstol + max(abs(y), abs(yn)) * self.reltol
            e_est = max(e_est, (abs(ev) / sc).reduce_max())

        vectorize[SIMD_WIDTH](self.y.flatten().size(), compute)

        return e_est


struct ODEProblem[
    dtype: DType where dtype.is_floating_point(),
    layout: Layout,
    //,
    system: system_fn[dtype, layout],
](Copyable):
    comptime LT = LayoutTensor[Self.dtype, Self.layout, MutAnyOrigin]
    var u0: Self.LT
    var tspan: Tuple[Scalar[Self.dtype], Scalar[Self.dtype]]
    var dt: Scalar[Self.dtype]

    def __init__(
        out self,
        u0: Self.LT,
        t_start: Scalar[Self.dtype],
        t_end: Scalar[Self.dtype],
        dt: Scalar[Self.dtype],
    ):
        self.u0 = u0
        self.tspan = (t_start, t_end)
        self.dt = dt


def solve[
    dtype: DType where dtype.is_floating_point(),
    layout: Layout,
    system: system_fn[dtype, layout],
](prob: ODEProblem[system], dt: Scalar[dtype]) -> LayoutTensor[
    dtype, layout, MutAnyOrigin
]:
    integrator = solver[system](prob.u0, prob.tspan[0], dt)
    t_end = prob.tspan[1]

    while integrator.t < t_end:
        # Check if the next step would overshoot the end
        # We use a small epsilon or simply clamp dt
        remaining = t_end - integrator.t
        if integrator.dt > remaining:
            integrator.dt = remaining
        integrator.step()

    # print(integrator.t)

    return integrator.y


comptime float_type = DType.float64
comptime lorenz_layout = Layout.row_major(3)

comptime Tsit5Adaptative = Tsit5[system=_, adaptive=True]
comptime Tsit5Fixed = Tsit5[system=_, adaptive=False]

comptime solver = Tsit5Adaptative
# comptime solver = Tsit5Fixed


def lorenz(
    dy: LayoutTensor[float_type, lorenz_layout, MutAnyOrigin],
    y: LayoutTensor[float_type, lorenz_layout, ImmutAnyOrigin],
    t: Float64,
):
    dy[0] = 10.0 * (y[1] - y[0])
    dy[1] = y[0] * (28.0 - y[2]) - y[1]
    dy[2] = y[0] * y[1] - (8.0 / 3.0) * y[2]


comptime GS_N = 8
comptime gs_layout = Layout.row_major(GS_N, GS_N, 2)


def gray_scott_system(
    dy: LayoutTensor[float_type, gs_layout, MutAnyOrigin],
    y: LayoutTensor[float_type, gs_layout, ImmutAnyOrigin],
    t: Scalar[float_type],
):
    """
    Gray-Scott Reaction-Diffusion.
    y[i, j, 0] is substance U.
    y[i, j, 1] is substance V.
    """
    # Parameters
    comptime Du = 0.16
    comptime Dv = 0.08
    comptime F = 0.060
    comptime k = 0.062

    # Retrieval of spatial dimensions (Species dim is index 2)
    dim_i = y.shape[0]()
    dim_j = y.shape[1]()

    for i in range(dim_i):
        for j in range(dim_j):
            # Finite difference indices (Periodic Boundary)
            ip1 = (i + 1) % GS_N
            im1 = (i - 1 + GS_N) % GS_N
            jp1 = (j + 1) % GS_N
            jm1 = (j - 1 + GS_N) % GS_N

            u = y[i, j, 0]
            lap_u = (
                y[ip1, j, 0]
                + y[im1, j, 0]
                + y[i, jp1, 0]
                + y[i, jm1, 0]
                - 4.0 * u
            )

            v = y[i, j, 1]
            lap_v = (
                y[ip1, j, 1]
                + y[im1, j, 1]
                + y[i, jp1, 1]
                + y[i, jm1, 1]
                - 4.0 * v
            )

            # Reaction-Diffusion logic
            uv2 = u * v * v
            dy[i, j, 0] = Du * lap_u - uv2 + F * (1.0 - u)
            dy[i, j, 1] = Dv * lap_v + uv2 - (F + k) * v


def setup_gray_scott_problem() -> ODEProblem[gray_scott_system]:
    size = GS_N * GS_N * 2
    ptr = alloc[Scalar[float_type]](size)

    # Initialize layout as (Height, Width, Species)
    u0 = LayoutTensor[float_type, gs_layout, MutAnyOrigin](
        ptr, RuntimeLayout[gs_layout].row_major(IndexList[3](GS_N, GS_N, 2))
    )
    dt = 0.1

    # Initialize U to 1.0, V to 0.0 everywhere
    for i in range(GS_N):
        for j in range(GS_N):
            u0[i, j, 0] = 1.0
            u0[i, j, 1] = 0.0

    # Add seed square in the center
    mid = GS_N // 2
    r = 3
    for i in range(mid - r, mid + r):
        for j in range(mid - r, mid + r):
            u0[i, j, 0] = 0.5  # perturb U
            u0[i, j, 1] = 0.25  # add V

    return ODEProblem[gray_scott_system](u0, 0.0, 10.0, dt)


def setup_lorenz_problem() -> ODEProblem[lorenz]:
    u0 = LayoutTensor[float_type, lorenz_layout, MutAnyOrigin](
        alloc[Scalar[float_type]](3)
    )
    dt = 0.01
    u0[0] = 1.0
    u0[1] = 0.0
    u0[2] = 0.0
    return ODEProblem[lorenz](u0, 0.0, 50.0, dt)


def basic_bench[
    dtype: DType where dtype.is_floating_point(),
    layout: Layout,
    system: system_fn[dtype, layout],
    //,
    setup_func: def() thin -> ODEProblem[system],
]() raises:
    def bench_fn() raises:
        prob = setup_func()
        res = solve[dtype, layout, system](prob, dt=prob.dt)
        keep(res)

    time_us = run[func1=bench_fn](max_iters=100).mean(Unit.us)
    print("t =", round(time_us, 1), "us")


def main() raises:
    print("Lorenz")
    lorenz_prob = setup_lorenz_problem()
    result = solve(lorenz_prob, dt=lorenz_prob.dt)
    print("Final State:", result)
    basic_bench[system=lorenz, setup_func=setup_lorenz_problem]()

    print("Gray-Scott")
    gs_prob = setup_gray_scott_problem()
    result_gs = solve(gs_prob, dt=gs_prob.dt)
    print("Final State at center:", result_gs[GS_N // 2, GS_N // 2, 0])
    basic_bench[system=gray_scott_system, setup_gray_scott_problem]()
