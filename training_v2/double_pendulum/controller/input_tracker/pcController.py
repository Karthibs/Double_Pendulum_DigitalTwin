import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController


class PiecewiseConstantController(AbstractController):
    def __init__(self, U, dt, torque_limit=[1.0, 1.0]):
        """
        Parameters
        ----------
        U : array-like, shape (N, 2)
            Piecewise constant control inputs
        dt : float
            Duration of each control segment
        torque_limit : list
            მაქ torque limits per joint
        """
        self.U = np.asarray(U)
        self.dt = dt
        self.torque_limit = torque_limit

        self.N = self.U.shape[0]

    def init(self):
        pass  # nothing to reset

    def get_control_output(self, x, t=None):
        """
        Select control based on time t.
        """
        if t is None:
            raise ValueError("Time 't' must be provided for PCC controller.")

        # determine index from time
        idx = int(t // self.dt)

        # clamp index to valid range
        if idx >= self.N:
            idx = self.N - 1

        u = self.U[idx]

        # apply torque limits
        u1 = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])

        return np.array([u1, u2])

    def get_init_trajectory(self):
        """
        Optional: return time grid and control sequence
        """
        T = np.arange(0, self.N * self.dt, self.dt)
        return T, None, self.U
