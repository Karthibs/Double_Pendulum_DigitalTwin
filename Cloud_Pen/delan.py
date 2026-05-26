import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class LowTri:
    def __init__(self, m):
        self._m = m
        self._idx = np.tril_indices(self._m)

    def __call__(self, l):
        batch_size = l.shape[0]
        self._L = torch.zeros(batch_size, self._m, self._m).type_as(l)
        self._L[:batch_size, self._idx[0], self._idx[1]] = l[:]
        return self._L[:batch_size]


class SoftplusDer(nn.Module):
    def __init__(self, beta=1.):
        super().__init__()
        self._beta = beta

    def forward(self, x):
        cx = torch.clamp(x, -20., 20.)
        exp_x = torch.exp(self._beta * cx)
        out = exp_x / (exp_x + 1.0)
        return out


class ReLUDer(nn.Module):
    def forward(self, x):
        return torch.ceil(torch.clamp(x, 0, 1))
    
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class SinDer(nn.Module):
    def forward(self, x):
        return torch.cos(x)



class Linear(nn.Module):
    def forward(self, x):
        return x


class LinearDer(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 1, 1)


class LagrangianLayer(nn.Module):
    def __init__(self, input_size, n_dof, activation="ReLu"):
        super().__init__()
        self.n_dof = n_dof
        self.weight = nn.Parameter(torch.Tensor(n_dof, input_size))
        self.bias   = nn.Parameter(torch.Tensor(n_dof))

        if activation == "ReLu":
            self.g, self.g_prime = nn.ReLU(), ReLUDer()
        elif activation == "SoftPlus":
            self.g, self.g_prime = nn.Softplus(beta=1.0), SoftplusDer(beta=1.0)
        elif activation == "Sine":
            self.g, self.g_prime = Sin(), SinDer()
        elif activation == "Linear":
            self.g, self.g_prime = Linear(), LinearDer()
        else:
            raise ValueError(f"Activation '{activation}' not in [ReLu, SoftPlus, Sine, Linear]")

    def forward(self, q, der_prev):
        a = F.linear(q, self.weight, self.bias)
        out = self.g(a)
        der = torch.matmul(self.g_prime(a).view(-1, self.n_dof, 1) * self.weight, der_prev)
        return out, der


# ---------------------------------------------------------------------------
# DeLaN network
# ---------------------------------------------------------------------------
class DeepLagrangianNetwork(nn.Module):
    def __init__(self, n_dof, learn_friction=False, **kwargs):
        super().__init__()
        # Hyper-params with Lutter's defaults
        self.n_width   = kwargs.get("n_width", 64)
        self.n_hidden  = kwargs.get("n_depth", 2)
        self._b0       = kwargs.get("b_init", 0.1)
        self._b0_diag  = kwargs.get("b_diag_init", 0.5)
        self._w_init   = kwargs.get("w_init", "xavier_normal")
        self._g_hidden = kwargs.get("g_hidden", np.sqrt(2.))
        self._g_output = kwargs.get("g_output", 0.125)
        self._p_sparse = kwargs.get("p_sparse", 0.2)
        self._epsilon  = kwargs.get("diagonal_epsilon", 1e-3)
        non_linearity  = kwargs.get("activation", "ReLu")

        # ---- Init helpers ----
        if self._w_init == "xavier_normal":
            def init_hidden(layer):
                hidden_gain = (torch.nn.init.calculate_gain('relu')
                               if self._g_hidden <= 0.0 else self._g_hidden)
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, hidden_gain)

            def init_output(layer):
                output_gain = (torch.nn.init.calculate_gain('linear')
                               if self._g_output <= 0.0 else self._g_output)
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, output_gain)
        elif self._w_init == "orthogonal":
            def init_hidden(layer):
                hidden_gain = (torch.nn.init.calculate_gain('relu')
                               if self._g_hidden <= 0.0 else self._g_hidden)
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, hidden_gain)

            def init_output(layer):
                output_gain = (torch.nn.init.calculate_gain('linear')
                               if self._g_output <= 0.0 else self._g_output)
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, output_gain)
        elif self._w_init == "sparse":
            assert 0.0 <= self._p_sparse < 1.0

            def init_hidden(layer):
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, self._p_sparse, self._g_hidden)

            def init_output(layer):
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, self._p_sparse, self._g_output)
        else:
            raise ValueError(f"Unknown w_init: {self._w_init}")

        # Sizes
        self.n_dof = n_dof
        self.m = int((n_dof ** 2 + n_dof) / 2)
        l_output_size = self.m
        l_lower_size  = l_output_size - n_dof

        # Indexing for reassembling l from (l_diag, l_lower)
        idx_diag = np.arange(self.n_dof) + 1
        idx_diag = idx_diag * (idx_diag + 1) / 2 - 1
        idx_tril = np.extract([x not in idx_diag for x in np.arange(l_output_size)],
                              np.arange(l_output_size))
        cat_idx = np.hstack((idx_diag, idx_tril))
        order = np.argsort(cat_idx)
        self._idx = np.arange(cat_idx.size)[order]

        self._eye = torch.eye(self.n_dof).view(1, self.n_dof, self.n_dof)
        self.low_tri = LowTri(self.n_dof)

        # Shared trunk
        self.layers = nn.ModuleList()
        self.layers.append(LagrangianLayer(self.n_dof, self.n_width, activation=non_linearity))
        init_hidden(self.layers[-1])
        for _ in range(1, self.n_hidden):
            self.layers.append(LagrangianLayer(self.n_width, self.n_width, activation=non_linearity))
            init_hidden(self.layers[-1])

        # Three heads: V(q), off-diag L entries, diag L entries
        self.net_g  = LagrangianLayer(self.n_width, 1, activation="Linear")
        init_output(self.net_g)
        self.net_lo = LagrangianLayer(self.n_width, l_lower_size, activation="Linear")
        init_hidden(self.net_lo)
        self.net_ld = LagrangianLayer(self.n_width, self.n_dof, activation="ReLu")
        init_hidden(self.net_ld)
        torch.nn.init.constant_(self.net_ld.bias, self._b0_diag)

        self.learn_friction = learn_friction
        if learn_friction:
            self.log_b_friction = nn.Parameter(torch.log(torch.full((n_dof,), 1e-3)))
            self.log_c_friction = nn.Parameter(torch.log(torch.full((n_dof,), 1e-3)))
        
        self.device = torch.device("cpu")


    # ---- Friction term ----
    def friction(self, qd):
        if not self.learn_friction:
            return torch.zeros_like(qd)
        b = torch.exp(self.log_b_friction)
        c = torch.exp(self.log_c_friction)
        return b * qd + c * torch.atan(100.0 * qd)


    # ---- Core dynamics computation ----
    def _dyn_model(self, q, qd, qdd):
        qd_3d = qd.view(-1, self.n_dof, 1)
        qd_4d = qd.view(-1, 1, self.n_dof, 1)

        # dq/dq = I at the input
        der = self._eye.repeat(q.shape[0], 1, 1).type_as(q)

        # Shared trunk
        y, der = self.layers[0](q, der)
        for i in range(1, len(self.layers)):
            y, der = self.layers[i](y, der)

        # Heads
        l_lower, der_l_lower = self.net_lo(y, der)
        l_diag,  der_l_diag  = self.net_ld(y, der)

        V, der_V = self.net_g(y, der)
        V = V.squeeze()
        g = der_V.squeeze(1) if der_V.dim() == 3 else der_V.squeeze()

        # Reassemble l so positions match tril_indices order
        l     = torch.cat((l_diag,     l_lower),     1)[:, self._idx]
        der_l = torch.cat((der_l_diag, der_l_lower), 1)[:, self._idx, :]

        # H = L L^T + eps I
        L = self.low_tri(l)
        LT = L.transpose(dim0=1, dim1=2)
        H = torch.matmul(L, LT) + self._epsilon * torch.eye(self.n_dof).type_as(L)

        # dH/dt
        Ldt = self.low_tri(torch.matmul(der_l, qd_3d).view(-1, self.m))
        Hdt = torch.matmul(L, Ldt.transpose(dim0=1, dim1=2)) + torch.matmul(Ldt, LT)

        # dH/dq
        Ldq = self.low_tri(der_l.transpose(2, 1).reshape(-1, self.m)
                           ).reshape(-1, self.n_dof, self.n_dof, self.n_dof)
        Hdq = (torch.matmul(Ldq, LT.view(-1, 1, self.n_dof, self.n_dof))
               + torch.matmul(L.view(-1, 1, self.n_dof, self.n_dof), Ldq.transpose(2, 3)))

        # Coriolis & centrifugal:  c = Hdt q_dot - 0.5 ∂(q_dot^T H q_dot)/∂q
        Hdt_qd  = torch.matmul(Hdt, qd_3d).view(-1, self.n_dof)
        quad_dq = torch.matmul(qd_4d.transpose(dim0=2, dim1=3),
                               torch.matmul(Hdq, qd_4d)).view(-1, self.n_dof)
        c = Hdt_qd - 0.5 * quad_dq

        # Inverse dynamics: τ = H q_ddot + c + g (+ friction, if enabled)
        H_qdd = torch.matmul(H, qdd.view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        tau_pred = H_qdd + c + g
        if self.learn_friction:
            tau_pred = tau_pred + self.friction(qd)

        # Kinetic & potential energies and their time derivatives
        H_qd = torch.matmul(H, qd_3d).view(-1, self.n_dof)
        T = 0.5 * torch.matmul(qd_4d.transpose(dim0=2, dim1=3),
                               H_qd.view(-1, 1, self.n_dof, 1)).view(-1)

        qd_H_qdd  = torch.matmul(qd_4d.transpose(dim0=2, dim1=3),
                                 H_qdd.view(-1, 1, self.n_dof, 1)).view(-1)
        qd_Hdt_qd = torch.matmul(qd_4d.transpose(dim0=2, dim1=3),
                                 Hdt_qd.view(-1, 1, self.n_dof, 1)).view(-1)
        dTdt = qd_H_qdd + 0.5 * qd_Hdt_qd

        dVdt = torch.matmul(qd_4d.transpose(dim0=2, dim1=3),
                            g.view(-1, 1, self.n_dof, 1)).view(-1)
        return tau_pred, H, c, g, T, V, dTdt, dVdt

    
    def forward(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        tau_pred = out[0]
        dEdt = out[6] + out[7]
        return tau_pred, dEdt

    def inv_dyn(self, q, qd, qdd):
        return self._dyn_model(q, qd, qdd)[0]

    def for_dyn(self, q, qd, tau):
        # Same trick as Lutter: feed zero qdd to recover H, c, g without
        # contributing to the inverse-dynamics output, then solve for qdd.
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        H, c, g = out[1], out[2], out[3]
        rhs = tau - c - g
        if self.learn_friction:
            rhs = rhs - self.friction(qd)
        invH = torch.inverse(H)
        qdd_pred = torch.matmul(invH, rhs.view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        return qdd_pred

    def energy(self, q, qd):
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        return out[4] + out[5]

    def energy_dot(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        return out[6] + out[7]

    def kinetic(self, q, qd):
        # convenience wrapper for evaluation code
        return self._dyn_model(q, qd, torch.zeros_like(q))[4]

    def potential(self, q):
        # need a dummy qd; potential only depends on q
        qd = torch.zeros_like(q)
        return self._dyn_model(q, qd, torch.zeros_like(q))[5]

    def mass_matrix(self, q):
        qd = torch.zeros_like(q)
        return self._dyn_model(q, qd, torch.zeros_like(q))[1]

    def cuda(self, device=None):
        super().cuda(device=device)
        self._eye = self._eye.cuda()
        self.device = self._eye.device
        return self

    def cpu(self):
        super().cpu()
        self._eye = self._eye.cpu()
        self.device = self._eye.device
        return self

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        # keep _eye on the right device
        try:
            self._eye = self._eye.to(*args, **kwargs)
            self.device = self._eye.device
        except Exception:
            pass
        return ret
