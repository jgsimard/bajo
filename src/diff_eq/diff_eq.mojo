from utils.index import IndexList
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from memory import UnsafePointer
from utils import StaticTuple
from tensor import InputTensor, OutputTensor, foreach

comptime system_fn[dtype: DType, L: Layout] = fn (
    mut LayoutTensor[dtype, L, MutAnyOrigin],
    LayoutTensor[dtype, L, MutAnyOrigin],
    Scalar[dtype],
) -> None


struct RK4[
    dtype: DType where dtype.is_floating_point(),
    L: Layout,
    //,
    system: system_fn[dtype, L],
]:
    comptime LT = LayoutTensor[Self.dtype, Self.L, MutAnyOrigin]
    var y: Self.LT
    var t: Scalar[Self.dtype]
    var dt: Scalar[Self.dtype]
    var k1: Self.LT
    var k2: Self.LT
    var k3: Self.LT
    var k4: Self.LT
    var tmp: Self.LT

    fn __init__(
        out self,
        u0: Self.LT,
        t_start: Scalar[Self.dtype],
        dt: Scalar[Self.dtype],
    ):
        self.y = u0
        self.t = t_start
        self.dt = dt

        self.k1 = Self.LT.stack_allocation()
        self.k2 = Self.LT.stack_allocation()
        self.k3 = Self.LT.stack_allocation()
        self.k4 = Self.LT.stack_allocation()
        self.tmp = Self.LT.stack_allocation()

    @always_inline
    fn step(mut self):
        var h = self.dt
        var h_half = h * 0.5

        # k1 = f(y, t)
        Self.system(self.k1, self.y, self.t)

        # k2 = f(y + h/2*k1, t + h/2)
        self.tmp = self.y + (self.k1 * h_half)
        Self.system(self.k2, self.tmp, self.t + h_half)

        # k3 = f(y + h/2*k2, t + h/2)
        self.tmp = self.y + (self.k2 * h_half)
        Self.system(self.k3, self.tmp, self.t + h_half)

        # k4 = f(y + h*k3, t + h)
        self.tmp = self.y + (self.k3 * h)
        Self.system(self.k4, self.tmp, self.t + h)

        # y = y + (h/6) * (k1 + 2k2 + 2k3 + k4)
        self.y += (self.k1 + (self.k2 * 2.0) + (self.k3 * 2.0) + self.k4) * (
            h / 6.0
        )
        self.t += h


struct Euler[
    dtype: DType where dtype.is_floating_point(),
    L: Layout,
    //,
    system: system_fn[dtype, L],
]:
    comptime LT = LayoutTensor[Self.dtype, Self.L, MutAnyOrigin]
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

    @always_inline
    fn step(mut self):
        Self.system(self.dy, self.y, self.t)

        # y = y + dt * dy
        self.y += self.dy * self.dt
        self.t += self.dt


@fieldwise_init
struct Dopri5Tableau[dtype: DType]:
    # Time offsets
    comptime c = StaticTuple[Scalar[Self.dtype], 7](
        0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0
    )

    # Error coefficients (b_i - b*_i)
    comptime e = StaticTuple[Scalar[Self.dtype], 7](
        71.0 / 57600.0,
        0.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -1.0 / 40.0,
    )

    # Weights for final result (b_i)
    comptime b = StaticTuple[Scalar[Self.dtype], 7](
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    )

    # Butcher A matrix components
    comptime a21 = 1.0 / 5.0
    comptime a31 = 3.0 / 40.0
    comptime a32 = 9.0 / 40.0
    comptime a41 = 44.0 / 45.0
    comptime a42 = -56.0 / 15.0
    comptime a43 = 32.0 / 9.0
    comptime a51 = 19372.0 / 6561.0
    comptime a52 = -25360.0 / 2187.0
    comptime a53 = 64448.0 / 6561.0
    comptime a54 = -212.0 / 729.0
    comptime a61 = 9017.0 / 3168.0
    comptime a62 = -355.0 / 33.0
    comptime a63 = 46732.0 / 5247.0
    comptime a64 = 49.0 / 176.0
    comptime a65 = -5103.0 / 18656.0


struct DormandPrince[
    dtype: DType where dtype.is_floating_point(),
    L: Layout,
    //,
    system: system_fn[dtype, L],
    adaptive: Bool = False,
]:
    comptime LT = LayoutTensor[Self.dtype, Self.L, MutAnyOrigin]
    comptime tab = Dopri5Tableau[Self.dtype]()

    var y: Self.LT
    var t: Scalar[Self.dtype]
    var dt: Scalar[Self.dtype]
    var u_modified: Bool

    # Unique buffers to prevent aliasing
    var k1: Self.LT
    var k2: Self.LT
    var k3: Self.LT
    var k4: Self.LT
    var k5: Self.LT
    var k6: Self.LT
    var k7: Self.LT
    var tmp: Self.LT
    var y_next: Self.LT
    var error_vec: Self.LT

    var abstol: Scalar[Self.dtype]
    var reltol: Scalar[Self.dtype]

    fn __init__(
        out self,
        u0: Self.LT,
        t_start: Scalar[Self.dtype],
        dt: Scalar[Self.dtype],
    ):
        self.y = u0
        self.t = t_start
        self.dt = dt
        self.u_modified = True
        self.abstol = 1e-6
        self.reltol = 1e-3

        self.k1 = Self.LT.stack_allocation()
        self.k2 = Self.LT.stack_allocation()
        self.k3 = Self.LT.stack_allocation()
        self.k4 = Self.LT.stack_allocation()
        self.k5 = Self.LT.stack_allocation()
        self.k6 = Self.LT.stack_allocation()
        self.k7 = Self.LT.stack_allocation()
        self.tmp = Self.LT.stack_allocation()
        self.y_next = Self.LT.stack_allocation()
        self.error_vec = Self.LT.stack_allocation()

    @always_inline
    fn step(mut self):
        if self.u_modified:
            Self.system(self.k1, self.y, self.t)
            self.u_modified = False
        else:
            self.k1.copy_from(self.k7)  # Deep copy from FSAL

        @parameter
        if self.adaptive:
            var accepted = False
            while not accepted:
                var h = self.dt
                self._compute_stages(h)

                # y_next = y + h * sum(b_i * k_i)
                self.y_next.copy_from(self.y)
                self.y_next += (
                    (self.k1 * Self.tab.b[0])
                    + (self.k3 * Self.tab.b[2])
                    + (self.k4 * Self.tab.b[3])
                    + (self.k5 * Self.tab.b[4])
                    + (self.k6 * Self.tab.b[5])
                ) * h

                # FSAL stage
                Self.system(self.k7, self.y_next, self.t + h)

                # Error Estimation
                self.error_vec = (
                    (self.k1 * Self.tab.e[0])
                    + (self.k3 * Self.tab.e[2])
                    + (self.k4 * Self.tab.e[3])
                    + (self.k5 * Self.tab.e[4])
                    + (self.k6 * Self.tab.e[5])
                    + (self.k7 * Self.tab.e[6])
                ) * h

                var e_est: Scalar[Self.dtype] = 0.0
                for i in range(self.y.size()):
                    var sc = (
                        self.abstol
                        + max(
                            abs(self.y.load_scalar(i)),
                            abs(self.y_next.load_scalar(i)),
                        )
                        * self.reltol
                    )
                    e_est = max(e_est, abs(self.error_vec.load_scalar(i)) / sc)

                if e_est <= 1.0:
                    self.y.copy_from(self.y_next)
                    self.t += h
                    if e_est > 0:
                        self.dt = h * min(
                            Scalar[Self.dtype](5.0),
                            max(
                                Scalar[Self.dtype](0.1), 0.9 * pow(e_est, -0.2)
                            ),
                        )
                    accepted = True
                else:
                    self.dt = h * max(
                        Scalar[Self.dtype](0.1), 0.9 * pow(e_est, -0.2)
                    )
        else:
            var h = self.dt
            self._compute_stages(h)
            self.y += (
                (self.k1 * Self.tab.b[0])
                + (self.k3 * Self.tab.b[2])
                + (self.k4 * Self.tab.b[3])
                + (self.k5 * Self.tab.b[4])
                + (self.k6 * Self.tab.b[5])
            ) * h
            self.t += h
            Self.system(self.k7, self.y, self.t)

    @always_inline
    fn _compute_stages(mut self, h: Scalar[Self.dtype]):
        # Stage 2
        self.tmp.copy_from(self.y)
        self.tmp += (self.k1 * Self.tab.a21) * h
        Self.system(self.k2, self.tmp, self.t + Self.tab.c[1] * h)

        # Stage 3
        self.tmp.copy_from(self.y)
        self.tmp += ((self.k1 * Self.tab.a31) + (self.k2 * Self.tab.a32)) * h
        Self.system(self.k3, self.tmp, self.t + Self.tab.c[2] * h)

        # Stage 4
        self.tmp.copy_from(self.y)
        self.tmp += (
            (self.k1 * Self.tab.a41)
            + (self.k2 * Self.tab.a42)
            + (self.k3 * Self.tab.a43)
        ) * h
        Self.system(self.k4, self.tmp, self.t + Self.tab.c[3] * h)

        # Stage 5
        self.tmp.copy_from(self.y)
        self.tmp += (
            (self.k1 * Self.tab.a51)
            + (self.k2 * Self.tab.a52)
            + (self.k3 * Self.tab.a53)
            + (self.k4 * Self.tab.a54)
        ) * h
        Self.system(self.k5, self.tmp, self.t + Self.tab.c[4] * h)

        # Stage 6
        self.tmp.copy_from(self.y)
        self.tmp += (
            (self.k1 * Self.tab.a61)
            + (self.k2 * Self.tab.a62)
            + (self.k3 * Self.tab.a63)
            + (self.k4 * Self.tab.a64)
            + (self.k5 * Self.tab.a65)
        ) * h
        Self.system(self.k6, self.tmp, self.t + h)


@fieldwise_init
struct Tsit5Tableau[dtype: DType]:
    # Time offsets
    comptime c = StaticTuple[Scalar[Self.dtype], 7](
        0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0
    )
    # Error coefficients (Julia's btildes)
    comptime e = StaticTuple[Scalar[Self.dtype], 7](
        -0.00178001105222577714,
        -0.0008164344596567469,
        0.007880878010261995,
        -0.1447110071732629,
        0.5823571654525552,
        -0.45808210592918697,
        1.0 / 66.0,
    )
    # butcher aij coefficients (Solution weights are row 7)
    comptime a21 = 0.161
    comptime a31 = -0.008480655492356989
    comptime a32 = 0.335480655492357
    comptime a41 = 2.8971530571054935
    comptime a42 = -6.359448489975075
    comptime a43 = 4.3622954328695815
    comptime a51 = 5.325864828439257
    comptime a52 = -11.748883564062828
    comptime a53 = 7.4955393428898365
    comptime a54 = -0.09249506636175525
    comptime a61 = 5.86145544294642
    comptime a62 = -12.92096931784711
    comptime a63 = 8.159367898576159
    comptime a64 = -0.071584973281401
    comptime a65 = -0.028269050394068383
    comptime a71 = 0.09646076681806523
    comptime a72 = 0.01
    comptime a73 = 0.4798896504144996
    comptime a74 = 1.379008574103742
    comptime a75 = -3.290069515436081
    comptime a76 = 2.324710524099774


struct Tsit5[
    dtype: DType where dtype.is_floating_point(),
    L: Layout,
    //,
    system: system_fn[dtype, L],
    adaptive: Bool,
]:
    comptime LT = LayoutTensor[Self.dtype, Self.L, MutAnyOrigin]
    comptime tab = Tsit5Tableau[Self.dtype]()

    var y: Self.LT
    var t: Scalar[Self.dtype]
    var dt: Scalar[Self.dtype]
    var qold: Scalar[Self.dtype]
    var u_modified: Bool

    # Explicitly separate buffers to prevent stack pointer aliasing
    var k1: Self.LT
    var k2: Self.LT
    var k3: Self.LT
    var k4: Self.LT
    var k5: Self.LT
    var k6: Self.LT
    var k7: Self.LT
    var tmp: Self.LT
    var y_next: Self.LT
    var error_vec: Self.LT

    var abstol: Scalar[Self.dtype]
    var reltol: Scalar[Self.dtype]

    fn __init__(
        out self,
        u0: Self.LT,
        t_start: Scalar[Self.dtype],
        dt: Scalar[Self.dtype],
    ):
        self.y = u0
        self.t = t_start
        self.dt = dt
        self.qold = 1e-4
        self.u_modified = True
        self.abstol = 1e-6
        self.reltol = 1e-3

        # Individual allocations are safer in current Mojo for stack buffers
        self.k1 = Self.LT.stack_allocation()
        self.k2 = Self.LT.stack_allocation()
        self.k3 = Self.LT.stack_allocation()
        self.k4 = Self.LT.stack_allocation()
        self.k5 = Self.LT.stack_allocation()
        self.k6 = Self.LT.stack_allocation()
        self.k7 = Self.LT.stack_allocation()
        self.tmp = Self.LT.stack_allocation()
        self.y_next = Self.LT.stack_allocation()
        self.error_vec = Self.LT.stack_allocation()

    @always_inline
    fn step(mut self):
        if self.u_modified:
            Self.system(self.k1, self.y, self.t)
            self.u_modified = False
        else:
            self.k1.copy_from(self.k7)  # Deep copy from FSAL stage

        @parameter
        if self.adaptive:
            var accepted = False
            while not accepted:
                var h = self.dt
                self._compute_ks(h)

                # Compute 5th order result
                self.y_next.copy_from(self.y)
                self.y_next += (
                    (self.k1 * Self.tab.a71)
                    + (self.k2 * Self.tab.a72)
                    + (self.k3 * Self.tab.a73)
                    + (self.k4 * Self.tab.a74)
                    + (self.k5 * Self.tab.a75)
                    + (self.k6 * Self.tab.a76)
                ) * h

                # FSAL: Compute k7 at the proposed y_next
                Self.system(self.k7, self.y_next, self.t + h)

                # Error Estimation: diff = h * sum(e_i * k_i)
                self.error_vec = (
                    (self.k1 * Self.tab.e[0])
                    + (self.k2 * Self.tab.e[1])
                    + (self.k3 * Self.tab.e[2])
                    + (self.k4 * Self.tab.e[3])
                    + (self.k5 * Self.tab.e[4])
                    + (self.k6 * Self.tab.e[5])
                    + (self.k7 * Self.tab.e[6])
                ) * h

                var e_est: Scalar[Self.dtype] = 0.0
                for i in range(self.y.size()):
                    var sc = (
                        self.abstol
                        + max(
                            abs(self.y.load_scalar(i)),
                            abs(self.y_next.load_scalar(i)),
                        )
                        * self.reltol
                    )
                    e_est = max(e_est, abs(self.error_vec.load_scalar(i)) / sc)

                if e_est <= 1.0:  # Accepted
                    self.y.copy_from(self.y_next)
                    self.t += h
                    # PI-Controller
                    var q = (
                        pow(e_est, 0.14) / pow(self.qold, 0.08) if e_est
                        > 0 else 0.1
                    )
                    self.dt = h * min(
                        Scalar[Self.dtype](5.0),
                        max(Scalar[Self.dtype](0.1), 0.9 / q),
                    )
                    self.qold = max(e_est, 1e-4)
                    accepted = True
                else:  # Rejected
                    self.dt = h * max(
                        Scalar[Self.dtype](0.1), 0.9 / pow(e_est, 0.14)
                    )
        else:
            # Fixed Step
            var h = self.dt
            self._compute_ks(h)
            self.y += (
                (self.k1 * Self.tab.a71)
                + (self.k2 * Self.tab.a72)
                + (self.k3 * Self.tab.a73)
                + (self.k4 * Self.tab.a74)
                + (self.k5 * Self.tab.a75)
                + (self.k6 * Self.tab.a76)
            ) * h
            self.t += h
            Self.system(self.k7, self.y, self.t)

    @always_inline
    fn _compute_ks(mut self, h: Scalar[Self.dtype]):
        self.tmp.copy_from(self.y)
        self.tmp += (self.k1 * Self.tab.a21) * h
        Self.system(self.k2, self.tmp, self.t + Self.tab.c[1] * h)

        self.tmp.copy_from(self.y)
        self.tmp += ((self.k1 * Self.tab.a31) + (self.k2 * Self.tab.a32)) * h
        Self.system(self.k3, self.tmp, self.t + Self.tab.c[2] * h)

        self.tmp.copy_from(self.y)
        self.tmp += (
            (self.k1 * Self.tab.a41)
            + (self.k2 * Self.tab.a42)
            + (self.k3 * Self.tab.a43)
        ) * h
        Self.system(self.k4, self.tmp, self.t + Self.tab.c[3] * h)

        self.tmp.copy_from(self.y)
        self.tmp += (
            (self.k1 * Self.tab.a51)
            + (self.k2 * Self.tab.a52)
            + (self.k3 * Self.tab.a53)
            + (self.k4 * Self.tab.a54)
        ) * h
        Self.system(self.k5, self.tmp, self.t + Self.tab.c[4] * h)

        self.tmp.copy_from(self.y)
        self.tmp += (
            (self.k1 * Self.tab.a61)
            + (self.k2 * Self.tab.a62)
            + (self.k3 * Self.tab.a63)
            + (self.k4 * Self.tab.a64)
            + (self.k5 * Self.tab.a65)
        ) * h
        Self.system(self.k6, self.tmp, self.t + h)


@fieldwise_init
struct BS3Tableau[dtype: DType]:
    comptime c = StaticTuple[Scalar[Self.dtype], 4](
        0.0, 1.0 / 2.0, 3.0 / 4.0, 1.0
    )

    # Error coefficients (b_i - b*_i)
    comptime e = StaticTuple[Scalar[Self.dtype], 4](
        2.0 / 3.0 - 5.0 / 9.0,
        0.0 - 1.0 / 3.0,
        0.0 - 4.0 / 9.0,
        0.0 - 0.0,  # FSAL handles the rest
    )

    # Butcher A matrix components
    comptime a21 = 1.0 / 2.0
    comptime a32 = 3.0 / 4.0
    comptime a41 = 2.0 / 9.0
    comptime a42 = 1.0 / 3.0
    comptime a43 = 4.0 / 9.0


struct BS3[
    dtype: DType where dtype.is_floating_point(),
    L: Layout,
    //,
    system: system_fn[dtype, L],
    adaptive: Bool = True,
]:
    comptime LT = LayoutTensor[Self.dtype, Self.L, MutAnyOrigin]
    comptime tab = BS3Tableau[Self.dtype]()

    var y: Self.LT
    var t: Scalar[Self.dtype]
    var dt: Scalar[Self.dtype]
    var u_modified: Bool

    var k1: Self.LT
    var k2: Self.LT
    var k3: Self.LT
    var k4: Self.LT
    var tmp: Self.LT
    var y_next: Self.LT
    var error_vec: Self.LT

    var abstol: Scalar[Self.dtype]
    var reltol: Scalar[Self.dtype]

    fn __init__(
        out self,
        u0: Self.LT,
        t_start: Scalar[Self.dtype],
        dt: Scalar[Self.dtype],
    ):
        self.y = u0
        self.t = t_start
        self.dt = dt
        self.u_modified = True
        self.abstol = 1e-6
        self.reltol = 1e-3

        self.k1 = Self.LT.stack_allocation()
        self.k2 = Self.LT.stack_allocation()
        self.k3 = Self.LT.stack_allocation()
        self.k4 = Self.LT.stack_allocation()
        self.tmp = Self.LT.stack_allocation()
        self.y_next = Self.LT.stack_allocation()
        self.error_vec = Self.LT.stack_allocation()

    @always_inline
    fn step(mut self):
        if self.u_modified:
            Self.system(self.k1, self.y, self.t)
            self.u_modified = False
        else:
            self.k1.copy_from(self.k4)  # FSAL

        @parameter
        if self.adaptive:
            var accepted = False
            while not accepted:
                var h = self.dt
                self._compute_ks(h)

                # y_next = y + h * (2/9*k1 + 1/3*k2 + 4/9*k3)
                self.y_next.copy_from(self.y)
                self.y_next += (
                    (self.k1 * Self.tab.a41)
                    + (self.k2 * Self.tab.a42)
                    + (self.k3 * Self.tab.a43)
                ) * h

                # k4 = f(y_next, t + h)
                Self.system(self.k4, self.y_next, self.t + h)

                # Error = h * sum(e_i * k_i)
                # BS3 specific error weights (b - b*)
                self.error_vec = (
                    (self.k1 * (2.0 / 3.0 - 5.0 / 9.0))
                    + (self.k2 * (-1.0 / 3.0))
                    + (self.k3 * (-4.0 / 9.0))
                    + (self.k4 * (1.0 / 1.0))
                ) * h

                var e_est: Scalar[Self.dtype] = 0.0
                for i in range(self.y.size()):
                    var sc = (
                        self.abstol
                        + max(
                            abs(self.y.load_scalar(i)),
                            abs(self.y_next.load_scalar(i)),
                        )
                        * self.reltol
                    )
                    e_est = max(e_est, abs(self.error_vec.load_scalar(i)) / sc)

                if e_est <= 1.0:
                    self.y.copy_from(self.y_next)
                    self.t += h
                    if e_est > 0:
                        self.dt = h * min(
                            Scalar[Self.dtype](5.0), 0.8 * pow(e_est, -0.3333)
                        )
                    accepted = True
                else:
                    self.dt = h * max(
                        Scalar[Self.dtype](0.1), 0.8 * pow(e_est, -0.3333)
                    )
        else:
            var h = self.dt
            self._compute_ks(h)
            self.y += (
                (self.k1 * Self.tab.a41)
                + (self.k2 * Self.tab.a42)
                + (self.k3 * Self.tab.a43)
            ) * h
            self.t += h
            Self.system(self.k4, self.y, self.t)

    @always_inline
    fn _compute_ks(mut self, h: Scalar[Self.dtype]):
        # k2 = f(y + h/2*k1, t + h/2)
        self.tmp.copy_from(self.y)
        self.tmp += (self.k1 * Self.tab.a21) * h
        Self.system(self.k2, self.tmp, self.t + Self.tab.c[1] * h)

        # k3 = f(y + 3/4*h*k2, t + 3/4*h)
        self.tmp.copy_from(self.y)
        self.tmp += (self.k2 * Self.tab.a32) * h
        Self.system(self.k3, self.tmp, self.t + Self.tab.c[2] * h)


trait ExplicitTableau:
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
    fn b[dtype: DType](i: Int) -> Scalar[dtype]:
        ...

    @staticmethod
    fn e[dtype: DType](i: Int) -> Scalar[dtype]:
        ...

    @staticmethod
    fn a[dtype: DType](i: Int, j: Int) -> Scalar[dtype]:
        ...


struct ExplicitIntegrator[
    dtype: DType,
    L: Layout,
    //,
    system: system_fn[dtype, L],
    Tableau: ExplicitTableau,
    adaptive: Bool,
]:
    comptime LT = LayoutTensor[Self.dtype, Self.L, MutAnyOrigin]
    comptime Tab = Self.Tableau

    var y: Self.LT
    var t: Scalar[Self.dtype]
    var dt: Scalar[Self.dtype]
    var u_modified: Bool

    var qold: Scalar[Self.dtype]
    var abstol: Scalar[Self.dtype]
    var reltol: Scalar[Self.dtype]

    var ks: InlineArray[Self.LT, Self.Tab.stages()]
    var tmp: Self.LT
    var y_next: Self.LT
    var error_vec: Self.LT

    fn __init__(
        out self,
        u0: Self.LT,
        t_start: Scalar[Self.dtype],
        dt: Scalar[Self.dtype],
    ):
        self.y = u0
        self.t = t_start
        self.dt = dt
        self.u_modified = True
        self.qold = 1e-4
        self.abstol = 1e-6
        self.reltol = 1e-3

        self.ks = InlineArray[Self.LT, Self.Tab.stages()](uninitialized=True)

        # Compile-time allocation to ensure unique stack buffers for each stage
        @parameter
        for i in range(Self.Tab.stages()):
            self.ks[i] = Self.LT.stack_allocation()

        self.tmp = Self.LT.stack_allocation()
        self.y_next = Self.LT.stack_allocation()
        self.error_vec = Self.LT.stack_allocation()

    @always_inline
    fn step(mut self):
        if self.u_modified:
            Self.system(self.ks[0], self.y, self.t)
            self.u_modified = False
        else:

            @parameter
            if Self.Tab.is_fsal():
                # FSAL requires a deep copy of the last derivative into the first slot
                self.ks[0].copy_from(self.ks[Self.Tab.stages() - 1])
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
                var weight_sum = self.ks[0] * Self.Tab.b[Self.dtype](0)

                @parameter
                for i in range(
                    1, Self.Tab.stages() - (1 if Self.Tab.is_fsal() else 0)
                ):
                    weight_sum += self.ks[i] * Self.Tab.b[Self.dtype](i)
                self.y_next += weight_sum * h

                @parameter
                if Self.Tab.is_fsal():
                    # Calculate final stage derivative at the proposed y_next
                    Self.system(
                        self.ks[Self.Tab.stages() - 1], self.y_next, self.t + h
                    )

                var e_est = self._estimate_error(h)

                if e_est <= 1.0:
                    self.y.copy_from(self.y_next)
                    self.t += h
                    # PI-Adaptive step size control
                    var q = (
                        pow(e_est, Self.Tab.beta1[Self.dtype]())
                        / pow(self.qold, Self.Tab.beta2[Self.dtype]()) if e_est
                        > 0 else 0.1
                    )
                    self.dt = h * min(
                        Scalar[Self.dtype](5.0),
                        max(Scalar[Self.dtype](0.1), 0.9 / q),
                    )
                    self.qold = max(e_est, 1e-4)
                    accepted = True
                else:
                    # Step rejected: reduce dt and retry
                    self.dt = h * max(
                        Scalar[Self.dtype](0.1),
                        0.9 / pow(e_est, Self.Tab.beta1[Self.dtype]()),
                    )
        else:
            # Fixed step logic
            var h = self.dt
            self._compute_stages(h)

            self.y_next.copy_from(self.y)
            var weight_sum = self.ks[0] * Self.Tab.b[Self.dtype](0)

            @parameter
            for i in range(
                1, Self.Tab.stages() - (1 if Self.Tab.is_fsal() else 0)
            ):
                weight_sum += self.ks[i] * Self.Tab.b[Self.dtype](i)

            self.y += weight_sum * h
            self.t += h

            @parameter
            if Self.Tab.is_fsal():
                Self.system(self.ks[Self.Tab.stages() - 1], self.y, self.t)

    @always_inline
    fn _compute_stages(mut self, h: Scalar[Self.dtype]):
        @parameter
        for i in range(1, Self.Tab.stages() - (1 if Self.Tab.is_fsal() else 0)):
            self.tmp.copy_from(self.y)
            var stage_sum = self.ks[0] * Self.Tab.a[Self.dtype](i, 0)

            @parameter
            for j in range(1, i):
                stage_sum += self.ks[j] * Self.Tab.a[Self.dtype](i, j)
            self.tmp += stage_sum * h
            Self.system(
                self.ks[i], self.tmp, self.t + Self.Tab.c[Self.dtype](i) * h
            )

    @always_inline
    fn _estimate_error(self, h: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        var err_sum = self.ks[0] * Self.Tab.e[Self.dtype](0)

        @parameter
        for i in range(1, Self.Tab.stages()):
            err_sum += self.ks[i] * Self.Tab.e[Self.dtype](i)

        var e_est: Scalar[Self.dtype] = 0.0
        var error_vec = err_sum * h
        for i in range(self.y.size()):
            var sc = (
                self.abstol
                + max(
                    abs(self.y.load_scalar(i)), abs(self.y_next.load_scalar(i))
                )
                * self.reltol
            )
            e_est = max(e_est, abs(error_vec.load_scalar(i)) / sc)
        return e_est


struct EulerTableau(ExplicitTableau):
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
    fn b[dtype: DType](i: Int) -> Scalar[dtype]:
        return 1.0

    @staticmethod
    fn e[dtype: DType](i: Int) -> Scalar[dtype]:
        return 0.0

    @staticmethod
    fn a[dtype: DType](i: Int, j: Int) -> Scalar[dtype]:
        return 0.0


struct Tsit5TableauV2(ExplicitTableau):
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
        if i == 1:
            return 0.161
        elif i == 2:
            return 0.327
        elif i == 3:
            return 0.9
        elif i == 4:
            return 0.9800255409045097
        elif i == 5:
            return 1.0
        elif i == 6:
            return 1.0
        return 0.0

    @staticmethod
    fn b[dtype: DType](i: Int) -> Scalar[dtype]:
        # In FSAL methods, the weights b are often the last row of the A matrix
        if i == 0:
            return 0.09646076681806523
        elif i == 1:
            return 0.01
        elif i == 2:
            return 0.4798896504144996
        elif i == 3:
            return 1.379008574103742
        elif i == 4:
            return -3.290069515436081
        elif i == 5:
            return 2.324710524099774
        return 0.0

    @staticmethod
    fn e[dtype: DType](i: Int) -> Scalar[dtype]:
        if i == 0:
            return -0.00178001105222577714
        elif i == 1:
            return -0.0008164344596567469
        elif i == 2:
            return 0.007880878010261995
        elif i == 3:
            return -0.1447110071732629
        elif i == 4:
            return 0.5823571654525552
        elif i == 5:
            return -0.45808210592918697
        elif i == 6:
            return 1.0 / 66.0
        return 0.0

    @staticmethod
    fn a[dtype: DType](i: Int, j: Int) -> Scalar[dtype]:
        if i == 1:
            if j == 0:
                return 0.161
        elif i == 2:
            if j == 0:
                return -0.008480655492356989
            elif j == 1:
                return 0.335480655492357
        elif i == 3:
            if j == 0:
                return 2.8971530571054935
            elif j == 1:
                return -6.359448489975075
            elif j == 2:
                return 4.3622954328695815
        elif i == 4:
            if j == 0:
                return 5.325864828439257
            elif j == 1:
                return -11.748883564062828
            elif j == 2:
                return 7.4955393428898365
            elif j == 3:
                return -0.09249506636175525
        elif i == 5:
            if j == 0:
                return 5.86145544294642
            elif j == 1:
                return -12.92096931784711
            elif j == 2:
                return 8.159367898576159
            elif j == 3:
                return -0.071584973281401
            elif j == 4:
                return -0.028269050394068383
        elif i == 6:  # FSAL row, determines y_next
            if j == 0:
                return 0.09646076681806523
            elif j == 1:
                return 0.01
            elif j == 2:
                return 0.4798896504144996
            elif j == 3:
                return 1.379008574103742
            elif j == 4:
                return -3.290069515436081
            elif j == 5:
                return 2.324710524099774
        return 0.0


struct ODEProblem[
    dtype: DType where dtype.is_floating_point(),
    L: Layout,
    //,
    system: system_fn[dtype, L],
]:
    comptime LT = LayoutTensor[Self.dtype, Self.L, MutAnyOrigin]
    var u0: Self.LT
    var tspan: Tuple[Scalar[Self.dtype], Scalar[Self.dtype]]

    fn __init__(
        out self,
        u0: Self.LT,
        t_start: Scalar[Self.dtype],
        t_end: Scalar[Self.dtype],
    ):
        self.u0 = u0
        self.tspan = (t_start, t_end)


fn solve[
    dtype: DType where dtype.is_floating_point(),
    L: Layout,
    system: system_fn[dtype, L],
](prob: ODEProblem[system], dt: Scalar[dtype]) -> LayoutTensor[
    dtype, L, MutAnyOrigin
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

    print(integrator.t)

    return integrator.y


comptime floattype = DType.float64
comptime lorenz_layout = Layout.row_major(3)

# comptime solver = Euler
# comptime solver = ExplicitIntegrator[Tableau=EulerTableau, adaptive=False] # euler

# comptime solver = RK4

# comptime solver = Tsit5[adaptive=True]
comptime solver = ExplicitIntegrator[
    Tableau=Tsit5TableauV2, adaptive=True
]  # Tsit5

# comptime solver = DormandPrince


fn lorenz(
    mut dy: LayoutTensor[floattype, lorenz_layout, MutAnyOrigin],
    y: LayoutTensor[floattype, lorenz_layout, MutAnyOrigin],
    t: Float64,
):
    dy[0] = 10.0 * (y[1] - y[0])
    dy[1] = y[0] * (28.0 - y[2]) - y[1]
    dy[2] = y[0] * y[1] - (8.0 / 3.0) * y[2]


fn main():
    var u0 = LayoutTensor[
        floattype, lorenz_layout, MutAnyOrigin
    ].stack_allocation()
    u0[0] = 1.0
    u0[1] = 0.0
    u0[2] = 0.0

    var prob = ODEProblem[lorenz](u0, 0.0, 50.0)

    var result = solve(prob, dt=0.01)

    print("Final State x:", result[0])
    print("Final State y:", result[1])
    print("Final State z:", result[2])
