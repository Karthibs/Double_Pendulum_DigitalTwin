import numpy as np
import torch

from double_pendulum.controller.abstract_controller import AbstractController
from delan import DeepLagrangianNetwork


class GravityCompensationLNN(AbstractController):

    def __init__(self, torque_limit=(0.15, 0.15),
                 scale=1, gen = 0, mainc = 'delan_best_4_strides2.pt', active='SoftPlus'):
        # self.model = model
        self.torque_limit = np.asarray(torque_limit, dtype=float)
        self.gen = gen
        self.scale = float(scale)
        self.model_selected_printed = False
        self.mainc = mainc
        self.active = active
        print(f"activation fn is {self.active}")
        # self.model.eval()

    def init(self):
        pass

    def get_control_output(self, x, t=None, cell_id=None):

        if self.gen == 0:
            model_c = self.mainc
    
            if not self.model_selected_printed:
                print(f"General Model is selected-{self.mainc}")
                self.model_selected_printed = True
    
        else:
            if cell_id == '16':
                model_c = 'delan_best16_1.pt'
                msg = "Model 16 is selected"
    
            elif cell_id == '169':
                model_c = 'delan_best169_1.pt'
                msg = "Model 169 is selected"
    
            elif cell_id == '171':
                model_c = 'delan_best171_1.pt'
                msg = "Model 171 is selected"
    
            else:
                raise ValueError("Select correct model")
    
            if not self.model_selected_printed:
                print(msg)
                self.model_selected_printed = True
    
        model = DeepLagrangianNetwork(
            n_dof=2, learn_friction=True,
            n_width=64, n_depth=2, activation=self.active,
            diagonal_epsilon=1e-3, b_diag_init=0.5,
        )
    
        ck = torch.load(model_c, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        model.eval()
    
        q  = torch.as_tensor(np.asarray(x[:2], dtype=np.float32)).unsqueeze(0)
        qd = torch.zeros_like(q)
    
        with torch.no_grad():
            out = model._dyn_model(q, qd, torch.zeros_like(q))
    
        g = out[3].squeeze(0).cpu().numpy()
    
        tau = self.scale * g
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        print(f"tau={tau}")
    
        return tau.astype(float).tolist()