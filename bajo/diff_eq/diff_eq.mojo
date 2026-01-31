from utils import IndexList, StaticTuple
from layout import Layout, LayoutTensor, RuntimeLayout
from layout.math import max as lmax
from memory import UnsafePointer
from tensor import InputTensor, OutputTensor, foreach
from benchmark import run, Unit, keep
from sys import simd_width_of
from algorithm import vectorize


comptime system_fn[dtype: DType, layout: Layout] = fn(
    mut LayoutTensor[dtype, layout, MutAnyOrigin],
    LayoutTensor[dtype, layout, MutAnyOrigin],
    Scalar[dtype],
) -> None


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

    fn __init__(
        out self,
        u0: Self.LT,
        t_start: Scalar[Self.dtype],
        dt: Scalar[Self.dtype],
    ):
        self.y = u0
        self.t = t_start
        self.dt = dt
        self.dy = Self.LT.stack_allocation()

    fn step(mut self):
        Self.system(self.dy, self.y, self.t)

        # y = y + dt * dy
        self.y += self.dy * self.dt
        self.t += self.dt


trait Tableau:
    @staticmethod
    fn stages() -> Int:
        ...

    @staticmethod
    fn is_fsal() -> Bool:
        ...

    @staticmethod
    fn order() -> Int:
        ...

    @staticmethod
    fn beta1[dtype: DType]() -> Scalar[dtype]:
        ...

    @staticmethod
    fn beta2[dtype: DType]() -> Scalar[dtype]:
        ...

    @staticmethod
    fn c[dtype: DType](i: Int) -> Scalar[dtype]:
        ...

    @staticmethod
    fn b[dtype: DType, i: Int]() -> Scalar[dtype]:
        ...

    @staticmethod
    fn e[dtype: DType](i: Int) -> Scalar[dtype]:
        ...

    @staticmethod
    fn a[dtype: DType](i: Int, j: Int) -> Scalar[dtype]:
        ...


struct ExplicitIntegrator[
    dtype: DType,
    layout: Layout,
    //,
    system: system_fn[dtype, layout],
    tableau: Tableau,
    adaptive: Bool,
]:
    comptime LT = LayoutTensor[Self.dtype, Self.layout, MutAnyOrigin]
    comptime Tab = Self.tableau

    var y: Self.LT
    var t: Scalar[Self.dtype]
    var dt: Scalar[Self.dtype]
    var u_modified: Bool

    var qold: Scalar[Self.dtype]
    var abstol: Scalar[Self.dtype]
    var reltol: Scalar[Self.dtype]

    var ks: InlineArray[Self.LT, Self.tableau.stages()]
    var tmp: Self.LT
    var y_next: Self.LT
    var error_vec: Self.LT

    fn __init__(
        out self,
        u0: Self.LT,
        t_start: Scalar[Self.dtype],
        dt: Scalar[Self.dtype],
        qold: Scalar[Self.dtype] = 1e-4,
        abstol: Scalar[Self.dtype] = 1e-6,
        reltol: Scalar[Self.dtype] = 1e-3,
    ):
        comptime FLOAT = Scalar[Self.dtype]
        comptime N_STAGES = Self.tableau.stages()

        self.y = u0
        self.t = t_start
        self.dt = dt
        self.u_modified = True
        self.qold = qold
        self.abstol = abstol
        self.reltol = reltol

        self.ks = InlineArray[Self.LT, N_STAGES](uninitialized=True)
        var size = u0.size()
        # var layout = u0.layout

        @parameter
        if Self.layout.shape.all_known():
            for i in range(N_STAGES):
                self.ks[i] = Self.LT(alloc[FLOAT](size))

            self.tmp = Self.LT(alloc[FLOAT](size))
            self.y_next = Self.LT(alloc[FLOAT](size))
            self.error_vec = Self.LT(alloc[FLOAT](size))

        else:
            for i in range(N_STAGES):
                self.ks[i] = Self.LT(alloc[FLOAT](size))

            self.tmp = Self.LT(alloc[FLOAT](size))
            self.y_next = Self.LT(alloc[FLOAT](size))
            self.error_vec = Self.LT(alloc[FLOAT](size))

    fn __del__(deinit self):
        for i in range(Self.tableau.stages()):
            self.ks[i].ptr.free()
        self.tmp.ptr.free()
        self.y_next.ptr.free()
        self.error_vec.ptr.free()

    fn step(mut self):
        comptime FLOAT = Scalar[Self.dtype]
        comptime N_STAGES = Self.tableau.stages()
        comptime N_STAGES_FSAL = N_STAGES - (1 if Self.tableau.is_fsal() else 0)

        if self.u_modified:
            Self.system(self.ks[0], self.y, self.t)
            self.u_modified = False

        else:

            @parameter
            if Self.tableau.is_fsal():
                # FSAL requires a deep copy of the last derivative into the first slot
                self.ks[0].copy_from(self.ks[N_STAGES_FSAL])
            else:
                Self.system(self.ks[0], self.y, self.t)

        @parameter
        if self.adaptive:
            var accepted = False
            while not accepted:
                var h = self.dt
                self._compute_stages(h)

                # Compute y_next using weights b (sum of stages 1 to S-1)
                self.y_next.copy_from(self.y)
                var weight_sum = self.ks[0] * Self.tableau.b[Self.dtype, 0]()

                @parameter
                for i in range(1, N_STAGES_FSAL):
                    weight_sum += self.ks[i] * Self.tableau.b[Self.dtype, i]()
                self.y_next += weight_sum * h

                @parameter
                if Self.tableau.is_fsal():
                    # Calculate final stage derivative at the proposed y_next
                    Self.system(self.ks[N_STAGES_FSAL], self.y_next, self.t + h)

                # var e_est = self._estimate_error(h)
                var e_est = self._estimate_error(h)

                comptime beta1 = Self.tableau.beta1[Self.dtype]()
                comptime beta2 = Self.tableau.beta2[Self.dtype]()
                if e_est <= 1.0:
                    self.y.copy_from(self.y_next)
                    self.t += h
                    # PI-Adaptive step size control
                    var q = (
                        pow(e_est, beta1) / pow(self.qold, beta2) if e_est
                        > 0 else 0.1
                    )
                    self.dt = h * min(FLOAT(5.0), max(FLOAT(0.1), 0.9 / q))
                    self.qold = max(e_est, 1e-4)
                    accepted = True
                else:
                    # Step rejected: reduce dt and retry
                    self.dt = h * max(FLOAT(0.1), 0.9 / pow(e_est, beta1))
        else:
            # Fixed step logic
            var h = self.dt
            self._compute_stages(h)

            self.y_next.copy_from(self.y)
            var weight_sum = self.ks[0] * Self.tableau.b[Self.dtype, 0]()

            @parameter
            for i in range(1, N_STAGES_FSAL):
                weight_sum += self.ks[i] * Self.tableau.b[Self.dtype, i]()

            self.y += weight_sum * h
            self.t += h

            @parameter
            if Self.tableau.is_fsal():
                Self.system(self.ks[N_STAGES_FSAL], self.y, self.t)

    fn _compute_stages(mut self, h: Scalar[Self.dtype]):
        comptime FLOAT = Scalar[Self.dtype]
        comptime N_STAGES = Self.tableau.stages()
        comptime N_STAGES_FSAL = N_STAGES - (1 if Self.tableau.is_fsal() else 0)

        var stage_sum = self.ks[0] * Self.tableau.a[Self.dtype](
            0, 0
        )  # dummy value to get LT
        for i in range(1, N_STAGES_FSAL):
            self.tmp.copy_from(self.y)

            _ = stage_sum.fill(0.0)
            for j in range(i):
                self.tmp += self.ks[j] * Self.tableau.a[Self.dtype](i, j) * h

            self.tmp += stage_sum * h
            Self.system(
                self.ks[i], self.tmp, self.t + Self.tableau.c[Self.dtype](i) * h
            )

    fn _estimate_error(mut self, h: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        comptime N_STAGES = Self.tableau.stages()
        comptime simd_width = simd_width_of[Self.dtype]()

        var e_est: SIMD[Self.dtype, 1] = 0.0

        fn compute[w: Int](i: Int) unified {mut}:
            var err_v = SIMD[Self.dtype, w](0.0)

            @parameter
            for s in range(N_STAGES):
                comptime e_coeff = Self.tableau.e[Self.dtype](s)
                if e_coeff != 0:
                    err_v += self.ks[s].ptr.load[width=w](i) * e_coeff

            var ev = err_v * h
            var y = self.y.ptr.load[width=w](i)
            var yn = self.y_next.ptr.load[width=w](i)
            var sc = self.abstol + max(abs(y), abs(yn)) * self.reltol
            e_est = max(e_est, (abs(ev) / sc).reduce_max())

        vectorize[simd_width](self.y.flatten().size(), compute)

        return e_est


struct EulerTableau(Tableau):
    @staticmethod
    fn stages() -> Int:
        return 1

    @staticmethod
    fn is_fsal() -> Bool:
        return False

    @staticmethod
    fn order() -> Int:
        return 1

    @staticmethod
    fn beta1[dtype: DType]() -> Scalar[dtype]:
        return 0.0

    @staticmethod
    fn beta2[dtype: DType]() -> Scalar[dtype]:
        return 0.0

    @staticmethod
    fn c[dtype: DType](i: Int) -> Scalar[dtype]:
        return 0.0

    @staticmethod
    fn b[dtype: DType, i: Int]() -> Scalar[dtype]:
        return 1.0

    @staticmethod
    fn e[dtype: DType](i: Int) -> Scalar[dtype]:
        return 0.0

    @staticmethod
    fn a[dtype: DType](i: Int, j: Int) -> Scalar[dtype]:
        return 0.0


struct Tsit5Tableau[tdtype: DType](Tableau):
    # fmt: off
    comptime _a = StaticTuple[Scalar[Self.tdtype], 49](
        0, 0, 0, 0, 0, 0, 0, # Row 0
        0.161, 0, 0, 0, 0, 0, 0, # Row 1
        -0.008480655492356989, 0.335480655492357, 0, 0, 0, 0, 0, # Row 2
        2.8971530571054935, -6.359448489975075, 4.3622954328695815, 0, 0, 0, 0, # Row 3
        5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525, 0, 0, 0, # Row 4
        5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383, 0, 0, # Row 5
        0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0 # Row 6
    )
    comptime _b = StaticTuple[Scalar[Self.tdtype], 7](
        0.09646076681806523, 0.01, 0.4798896504144996,
        1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0
    )
    comptime _c = StaticTuple[Scalar[Self.tdtype], 7](0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0)
    comptime _e = StaticTuple[Scalar[Self.tdtype], 7](
        -0.00178001105222577714, -0.0008164344596567469, 0.007880878010261995,
        -0.1447110071732629, 0.5823571654525552, -0.45808210592918697, 1.0/66.0
    )
    # fmt: on

    @staticmethod
    fn stages() -> Int:
        return 7

    @staticmethod
    fn is_fsal() -> Bool:
        return True

    @staticmethod
    fn order() -> Int:
        return 5

    @staticmethod
    fn beta1[dtype: DType]() -> Scalar[dtype]:
        return 7.0 / 50.0

    @staticmethod
    fn beta2[dtype: DType]() -> Scalar[dtype]:
        return 2.0 / 25.0

    @staticmethod
    fn c[dtype: DType](i: Int) -> Scalar[dtype]:
        return Self._c[i].cast[dtype]()

    @staticmethod
    fn b[dtype: DType, i: Int]() -> Scalar[dtype]:
        # In FSAL methods, the weights b are often the last row of the A matrix
        return Self._b[i].cast[dtype]()

    @staticmethod
    fn e[dtype: DType](i: Int) -> Scalar[dtype]:
        return Self._e[i].cast[dtype]()

    @staticmethod
    fn a[dtype: DType](i: Int, j: Int) -> Scalar[dtype]:
        return Self._a[i * Self.stages() + j].cast[dtype]()


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

    fn __init__(
        out self,
        u0: Self.LT,
        t_start: Scalar[Self.dtype],
        t_end: Scalar[Self.dtype],
        dt: Scalar[Self.dtype],
    ):
        self.u0 = u0
        self.tspan = (t_start, t_end)
        self.dt = dt


fn solve[
    dtype: DType where dtype.is_floating_point(),
    layout: Layout,
    system: system_fn[dtype, layout],
](prob: ODEProblem[system], dt: Scalar[dtype]) -> LayoutTensor[
    dtype, layout, MutAnyOrigin
]:
    var integrator = solver[system](prob.u0, prob.tspan[0], dt)
    var t_end = prob.tspan[1]

    while integrator.t < t_end:
        # Check if the next step would overshoot the end
        # We use a small epsilon or simply clamp dt
        var remaining = t_end - integrator.t
        if integrator.dt > remaining:
            integrator.dt = remaining
        integrator.step()

    # print(integrator.t)

    return integrator.y


comptime float_type = DType.float64
comptime lorenz_layout = Layout.row_major(3)

comptime Tsit5Adaptative = ExplicitIntegrator[
    tableau = Tsit5Tableau[float_type], adaptive=True
]

comptime Tsit5Fixed = ExplicitIntegrator[
    tableau = Tsit5Tableau[float_type], adaptive=False
]

comptime solver = Tsit5Adaptative
# comptime solver = Tsit5Fixed


fn lorenz(
    mut dy: LayoutTensor[float_type, lorenz_layout, MutAnyOrigin],
    y: LayoutTensor[float_type, lorenz_layout, MutAnyOrigin],
    t: Float64,
):
    dy[0] = 10.0 * (y[1] - y[0])
    dy[1] = y[0] * (28.0 - y[2]) - y[1]
    dy[2] = y[0] * y[1] - (8.0 / 3.0) * y[2]


comptime GS_N = 8
comptime gs_layout = Layout.row_major(GS_N, GS_N, 2)


fn gray_scott_system(
    mut dy: LayoutTensor[float_type, gs_layout, MutAnyOrigin],
    y: LayoutTensor[float_type, gs_layout, MutAnyOrigin],
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
    var dim_i = y.shape[0]()
    var dim_j = y.shape[1]()

    for i in range(dim_i):
        for j in range(dim_j):
            # Finite difference indices (Periodic Boundary)
            var ip1 = (i + 1) % GS_N
            var im1 = (i - 1 + GS_N) % GS_N
            var jp1 = (j + 1) % GS_N
            var jm1 = (j - 1 + GS_N) % GS_N

            var u = y[i, j, 0]
            var lap_u = (
                y[ip1, j, 0]
                + y[im1, j, 0]
                + y[i, jp1, 0]
                + y[i, jm1, 0]
                - 4.0 * u
            )

            var v = y[i, j, 1]
            var lap_v = (
                y[ip1, j, 1]
                + y[im1, j, 1]
                + y[i, jp1, 1]
                + y[i, jm1, 1]
                - 4.0 * v
            )

            # Reaction-Diffusion logic
            var uv2 = u * v * v
            dy[i, j, 0] = Du * lap_u - uv2 + F * (1.0 - u)
            dy[i, j, 1] = Dv * lap_v + uv2 - (F + k) * v


fn setup_gray_scott_problem() -> ODEProblem[gray_scott_system]:
    var size = GS_N * GS_N * 2
    var ptr = alloc[Scalar[float_type]](size)

    # Initialize layout as (Height, Width, Species)
    var u0 = LayoutTensor[float_type, gs_layout, MutAnyOrigin](
        ptr, RuntimeLayout[gs_layout].row_major(IndexList[3](GS_N, GS_N, 2))
    )
    var dt = 0.1

    # Initialize U to 1.0, V to 0.0 everywhere
    for i in range(GS_N):
        for j in range(GS_N):
            u0[i, j, 0] = 1.0
            u0[i, j, 1] = 0.0

    # Add seed square in the center
    var mid = GS_N // 2
    var r = 3
    for i in range(mid - r, mid + r):
        for j in range(mid - r, mid + r):
            u0[i, j, 0] = 0.5  # perturb U
            u0[i, j, 1] = 0.25  # add V

    return ODEProblem[gray_scott_system](u0, 0.0, 10.0, dt)


fn setup_lorenz_problem() -> ODEProblem[lorenz]:
    var u0 = LayoutTensor[float_type, lorenz_layout, MutAnyOrigin](
        alloc[Scalar[float_type]](3)
    )
    var dt = 0.01
    u0[0] = 1.0
    u0[1] = 0.0
    u0[2] = 0.0
    return ODEProblem[lorenz](u0, 0.0, 50.0, dt)


fn basic_bench[
    dtype: DType where dtype.is_floating_point(),
    layout: Layout,
    system: system_fn[dtype, layout],
    //,
    setup_func: fn() -> ODEProblem[system],
]() raises:
    fn bench_fn() raises:
        var prob = setup_func()
        var res = solve[dtype, layout, system](prob, dt=prob.dt)
        keep(res)

    var time_us = run[func1=bench_fn](max_iters=100).mean(Unit.us)
    print("t =", round(time_us, 1), "us")


fn main() raises:
    print("Lorenz")
    var lorenz_prob = setup_lorenz_problem()
    var result = solve(lorenz_prob, dt=lorenz_prob.dt)
    print("Final State:", result)
    basic_bench[setup_lorenz_problem]()

    print("Gray-Scott")
    var gs_prob = setup_gray_scott_problem()
    var result_gs = solve(gs_prob, dt=gs_prob.dt)
    print("Final State at center:", result_gs[GS_N // 2, GS_N // 2, 0])
    basic_bench[setup_gray_scott_problem]()
