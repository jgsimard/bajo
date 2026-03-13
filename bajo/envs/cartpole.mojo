# see: https://github.com/PufferAI/PufferLib/tree/3.0/pufferlib/ocean/cartpole
from std.utils.numerics import isinf
from std.math import clamp, sin, cos, pi
from std.random import randn, randn_float64, random_float64
from std.testing import assert_false

from bajo.core.conversion import degrees_to_radians


comptime X_THRESHOLD: Float32 = 2.4
comptime THETA_THRESHOLD_RADIANS: Float32 = Float32(degrees_to_radians(12.0))
comptime MAX_STEPS = 200
comptime WIDTH = 600
comptime HEIGHT = 200
comptime SCALE = 100

comptime cart_mass: Float32 = 1.0
comptime pole_mass: Float32 = 0.1
comptime pole_length: Float32 = 0.5
comptime gravity: Float32 = 9.8
comptime force_mag: Float32 = 10.0
comptime tau: Float32 = 0.02
comptime render_mode = "human"
comptime report_interval = 1
comptime continuous = False


struct Log:
    var perf: Float32
    var episode_length: Float32
    var x_threshold_termination: Float32
    var pole_angle_termination: Float32
    var max_steps_termination: Float32
    var n: Float32
    var score: Float32


trait Env:
    def step(mut self) raises:
        ...

    def reset(mut self):
        ...


struct Cartpole[continuous: Bool, config: ConfigCartPole](Env):
    # State data
    var observations: InlineArray[Float32, 4]
    var actions: Float32
    var rewards: Float32
    var terminals: List[Bool]
    var truncations: List[Bool]

    # Logging
    var log: Log

    # Client* client;

    # Physics State
    var x: Float32
    var x_dot: Float32
    var theta: Float32
    var theta_dot: Float32
    var tick: Int
    var episode_return: Float32

    # # Environment Config
    # var cart_mass: Float32
    # var pole_mass: Float32
    # var pole_length: Float32
    # var gravity: Float32
    # var force_mag: Float32
    # var tau: Float32
    # var continuous: Bool

    def step(mut self) raises:
        a = self.actions[0]

        assert_false(isinf(a) or (a < 1.0) or (a > 1.0))

        force: Float32

        comptime if self.config.continuous:
            force = a * force_mag
        else:
            force = force_mag if a > 0.5 else -force_mag

        costheta = cos(self.theta)
        sintheta = sin(self.theta)

        comptime total_mass = cart_mass + pole_mass
        comptime polemass_length = total_mass + pole_mass

        temp = (
            force + polemass_length * self.theta_dot * self.theta_dot * sintheta
        ) / total_mass

        # denominator = pole_length * (4.0 / 3.0 - (total_mass * costheta * costheta) / total_mass)
        denominator = pole_length * (4.0 / 3.0 - costheta * costheta)
        thetaacc = (gravity * sintheta - costheta * temp) / denominator
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Euler Integration
        self.x += tau * self.x_dot
        self.x_dot += tau * xacc
        self.theta += tau * self.theta_dot
        self.theta_dot += tau * thetaacc

        self.tick += 1

        # Termination checks
        terminated = (
            (self.x < -X_THRESHOLD)
            or (self.x > X_THRESHOLD)
            or (self.theta < -THETA_THRESHOLD_RADIANS)
            or (self.theta > THETA_THRESHOLD_RADIANS)
        )

        truncated = self.tick >= MAX_STEPS
        done = terminated or truncated

        self.rewards[0] = Float32(0.0 if done else 1.0)
        self.episode_return += self.rewards[0]
        self.terminals[0] = terminated

        if done:
            self.add_log()
            self.reset()

    def compute_observations(mut self):
        self.observations[0] = self.x
        self.observations[1] = self.x_dot
        self.observations[2] = self.theta
        self.observations[3] = self.theta_dot

    def reset(mut self):
        comptime STD = 0.04
        self.episode_return = 0.0
        self.x = Float32(randn_float64(0.0, STD))
        self.x_dot = Float32(randn_float64(0.0, STD))
        self.theta = Float32(randn_float64(0.0, STD))
        self.theta_dot = Float32(randn_float64(0.0, STD))
        self.tick = 0

        self.compute_observations()


def in_range[
    dtype: DType, //, min: Scalar[dtype], max: Scalar[dtype]
](x: Scalar[dtype]) -> Bool:
    return x > min and x < max


def out_range[
    dtype: DType, //, min: Scalar[dtype], max: Scalar[dtype]
](x: Scalar[dtype]) -> Bool:
    return not in_range[min, max](x)
