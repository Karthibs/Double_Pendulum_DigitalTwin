import numpy as np
import torch

from double_pendulum.controller.abstract_controller import AbstractController


class GravityCompensationLNN(AbstractController):

    def __init__(self, model, torque_limit=(0.15, 0.15),
                 scale=1.0):
        self.model = model
        self.torque_limit = np.asarray(torque_limit, dtype=float)
        self.scale = float(scale)
        self.model.eval()

    def init(self):
        pass

    def get_control_output(self, x, t=None):
        q  = torch.as_tensor(np.asarray(x[:2], dtype=np.float32)).unsqueeze(0)
        qd = torch.zeros_like(q)
        with torch.no_grad():
            out = self.model._dyn_model(q, qd, torch.zeros_like(q))
        g = out[3].squeeze(0).cpu().numpy()    

        tau = self.scale * g
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        return tau.astype(float).tolist()