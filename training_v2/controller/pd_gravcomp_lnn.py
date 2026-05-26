
import numpy as np
import torch

from double_pendulum.controller.abstract_controller import AbstractController


class PDGravCompLNN(AbstractController):

    def __init__(self, model, q_target,
                 Kp=(0.5, 0.5), Kd=(0.05, 0.05),
                 torque_limit=(0.15, 0.15),
                 grav_scale=1.0):
        self.model = model
        self.q_target = np.asarray(q_target, dtype=float)
        self.Kp = np.asarray(Kp, dtype=float)
        self.Kd = np.asarray(Kd, dtype=float)
        self.torque_limit = np.asarray(torque_limit, dtype=float)
        self.grav_scale = float(grav_scale)
        self.model.eval()

    def init(self):
        pass

    def get_control_output(self, x, t=None):
        x = np.asarray(x, dtype=float)
        q   = x[:2]
        qd  = x[2:]

        q_t  = torch.as_tensor(q.astype(np.float32)).unsqueeze(0)
        qd_t = torch.zeros_like(q_t)
        with torch.no_grad():
            out = self.model._dyn_model(q_t, qd_t, torch.zeros_like(q_t))
        g_learned = out[3].squeeze(0).cpu().numpy()
        tau_ff = self.grav_scale * g_learned

        tau_fb = self.Kp * (self.q_target - q) + self.Kd * (0.0 - qd)

        tau = tau_ff + tau_fb
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        return tau.astype(float).tolist()
