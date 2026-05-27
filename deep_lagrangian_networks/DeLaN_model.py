"""
deep_lagrangian_networks/DeLaN_model_v0.py

DeLaN_model
  - FormulaFriction
  - friction_type ∈ {None, "rayleigh", "formula"}
  - forward return (tau_pred, dEdt, tau_f, p_diss)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        if torch.isnan(out).any():
            print("SoftPlus Forward output is NaN.")
        return out


class ReLUDer(nn.Module):
    def forward(self, x):
        return torch.ceil(torch.clamp(x, 0, 1))


class Linear(nn.Module):
    def forward(self, x):
        return x


class LinearDer(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 1, 1)


class Cos(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class CosDer(nn.Module):
    def forward(self, x):
        return -torch.sin(x)


class LagrangianLayer(nn.Module):
    def __init__(self, input_size, n_dof, activation="SoftPlus"):
        super().__init__()
        self.n_dof = n_dof
        self.weight = nn.Parameter(torch.Tensor(n_dof, input_size))
        self.bias = nn.Parameter(torch.Tensor(n_dof))

        if activation == "ReLu":
            self.g, self.g_prime = nn.ReLU(), ReLUDer()
        elif activation == "SoftPlus":
            self.softplus_beta = 1.0
            self.g = nn.Softplus(beta=self.softplus_beta)
            self.g_prime = SoftplusDer(beta=self.softplus_beta)
        elif activation == "Cos":
            self.g, self.g_prime = Cos(), CosDer()
        elif activation == "Linear":
            self.g, self.g_prime = Linear(), LinearDer()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, q, der_prev):
        a = F.linear(q, self.weight, self.bias)
        out = self.g(a)
        der = torch.matmul(self.g_prime(a).view(-1, self.n_dof, 1) * self.weight, der_prev)
        return out, der


class FormulaFriction(nn.Module):
    """
    τ_f_i = -(τ_c_i + τ_s_i * exp(-(q̇_i/ν_i)^2)) * tanh(q̇_i/ε_i) - d_i * q̇_i
    """
    def __init__(self, n_dof,
                 init_tau_c=0.05, init_tau_s=0.05,
                 init_nu=0.5,     init_d=0.05,
                 init_eps=0.05,   eps_min=1e-3):
        super().__init__()
        self.n_dof, self.eps_min = n_dof, eps_min

        def _inv_sp(x):
            x = torch.as_tensor(x, dtype=torch.float32)
            return torch.log(torch.expm1(x.clamp_min(1e-6)))

        self.raw_tau_c = nn.Parameter(_inv_sp(torch.full((n_dof,), init_tau_c)))
        self.raw_tau_s = nn.Parameter(_inv_sp(torch.full((n_dof,), init_tau_s)))
        self.raw_nu    = nn.Parameter(_inv_sp(torch.full((n_dof,), init_nu)))
        self.raw_d     = nn.Parameter(_inv_sp(torch.full((n_dof,), init_d)))
        self.raw_eps   = nn.Parameter(_inv_sp(torch.full((n_dof,), init_eps)))

    def params(self):
        sp = F.softplus
        return (sp(self.raw_tau_c),
                sp(self.raw_tau_s),
                sp(self.raw_nu) + 1e-4,
                sp(self.raw_d),
                sp(self.raw_eps) + self.eps_min)

    def forward(self, qd):
        tau_c, tau_s, nu, d, eps = self.params()
        stribeck   = tau_s * torch.exp(-(qd / nu) ** 2)
        smooth_sgn = torch.tanh(qd / eps)
        tau_f  = -(tau_c + stribeck) * smooth_sgn - d * qd
        p_diss = -(qd * tau_f).sum(dim=1, keepdim=True)  # ≥0
        return tau_f, p_diss

    def extra_repr(self):
        tc, ts, nu, d, eps = [p.detach().cpu().numpy() for p in self.params()]
        return f"tau_c={tc}, tau_s={ts}, nu={nu}, d={d}, eps={eps}"


class DeepLagrangianNetwork(nn.Module):

    def __init__(self, n_dof, **kwargs):
        super().__init__()


        self.n_width  = kwargs.get("n_width", 128)
        self.n_hidden = kwargs.get("n_depth", 1)
        self._b0      = kwargs.get("b_init", 0.1)
        self._b0_diag = kwargs.get("b_diag_init", 0.1)
        self._w_init  = kwargs.get("w_init", "xavier_normal")
        self._g_hidden = kwargs.get("g_hidden", np.sqrt(2.))
        self._g_output = kwargs.get("g_hidden", 0.125)
        self._p_sparse = kwargs.get("p_sparse", 0.2)
        self._epsilon  = kwargs.get("diagonal_epsilon", 1.e-5)


        if self._w_init == "xavier_normal":
            def init_hidden(layer):
                gain = torch.nn.init.calculate_gain('relu') if self._g_hidden <= 0 else self._g_hidden
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, gain)
            def init_output(layer):
                gain = torch.nn.init.calculate_gain('linear') if self._g_output <= 0 else self._g_output
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, gain)
        elif self._w_init == "orthogonal":
            def init_hidden(layer):
                gain = torch.nn.init.calculate_gain('relu') if self._g_hidden <= 0 else self._g_hidden
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, gain)
            def init_output(layer):
                gain = torch.nn.init.calculate_gain('linear') if self._g_output <= 0 else self._g_output
                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, gain)
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


        self.n_dof = n_dof
        self.m = int((n_dof ** 2 + n_dof) / 2)
        l_output_size = self.m
        l_lower_size  = l_output_size - self.n_dof
        idx_diag = np.arange(self.n_dof) + 1
        idx_diag = idx_diag * (idx_diag + 1) / 2 - 1
        idx_tril = np.extract([x not in idx_diag for x in np.arange(l_output_size)],
                              np.arange(l_output_size))
        cat_idx = np.hstack((idx_diag, idx_tril))
        self._idx = np.arange(cat_idx.size)[np.argsort(cat_idx)]

        self._eye = torch.eye(self.n_dof).view(1, self.n_dof, self.n_dof)
        self.low_tri = LowTri(self.n_dof)

        self.layers = nn.ModuleList()
        non_linearity = kwargs.get("activation", "ReLu")
        self.layers.append(LagrangianLayer(self.n_dof, self.n_width, activation=non_linearity))
        init_hidden(self.layers[-1])
        for _ in range(1, self.n_hidden):
            self.layers.append(LagrangianLayer(self.n_width, self.n_width, activation=non_linearity))
            init_hidden(self.layers[-1])

        self.net_g = LagrangianLayer(self.n_width, 1, activation="Linear")
        init_output(self.net_g)
        self.net_lo = LagrangianLayer(self.n_width, l_lower_size, activation="Linear")
        init_hidden(self.net_lo)
        self.net_ld = LagrangianLayer(self.n_width, self.n_dof, activation="ReLu")
        init_hidden(self.net_ld)
        torch.nn.init.constant_(self.net_ld.bias, self._b0_diag)

        friction_type = kwargs.get("friction_type", None)
        if friction_type is None:
            friction_type = "rayleigh" if kwargs.get("use_friction", False) else None
        self.friction_type = friction_type

        if friction_type == "rayleigh":
            f_hidden = kwargs.get("friction_hidden", (32, 32))
            layers, d = [], n_dof
            for h in f_hidden:
                layers += [nn.Linear(d, h), nn.Softplus()]
                d = h
            layers += [nn.Linear(d, n_dof)]
            self.friction_phi = nn.Sequential(*layers)
            self.register_buffer("_qd_zero", torch.zeros(1, n_dof))
        elif friction_type == "formula":
            self.friction_formula = FormulaFriction(
                n_dof,
                init_tau_c=kwargs.get("init_tau_c", 0.05),
                init_tau_s=kwargs.get("init_tau_s", 0.05),
                init_nu   =kwargs.get("init_nu",    0.5),
                init_d    =kwargs.get("init_d",     0.05),
                init_eps  =kwargs.get("init_eps",   0.05),
            )
        elif friction_type is None:
            pass
        else:
            raise ValueError(f"Unknown friction_type: {friction_type}")

    @property
    def use_friction(self):
        return self.friction_type is not None

    # ---------- Rayleigh  ----------
    def _rayleigh(self, qd):
        with torch.enable_grad():
            qd_in = qd if qd.requires_grad else qd.detach().requires_grad_(True)
            phi_qd = self.friction_phi(qd_in)
            phi_0  = self.friction_phi(self._qd_zero.expand_as(qd_in))
            R = 0.5 * ((phi_qd - phi_0) ** 2).sum(-1, keepdim=True)
            tau_f = torch.autograd.grad(
                R.sum(), qd_in,
                create_graph=self.training,
                retain_graph=self.training)[0]
        return R, tau_f

    # ---------- forward ----------
    def forward(self, q, qd, qdd):
        tau_pred, H, c, g, T, V, dTdt, dVdt = self._dyn_model(q, qd, qdd)
        dEdt = dTdt + dVdt

        if self.friction_type == "rayleigh":
            _, tau_f = self._rayleigh(qd)
            if not self.training:
                tau_f = tau_f.detach()
            p_diss = (qd * tau_f).sum(-1, keepdim=True)
        elif self.friction_type == "formula":
            tau_f, p_diss = self.friction_formula(qd)
        else:
            tau_f  = torch.zeros_like(qd)
            p_diss = torch.zeros(qd.shape[0], 1, device=qd.device, dtype=qd.dtype)

        tau_pred = tau_pred + tau_f
        return tau_pred, dEdt, tau_f, p_diss

    def _dyn_model(self, q, qd, qdd):
        qd_3d = qd.view(-1, self.n_dof, 1)
        qd_4d = qd.view(-1, 1, self.n_dof, 1)

        der = self._eye.repeat(q.shape[0], 1, 1).type_as(q)
        y, der = self.layers[0](q, der)
        for i in range(1, len(self.layers)):
            y, der = self.layers[i](y, der)

        l_lower, der_l_lower = self.net_lo(y, der)
        l_diag,  der_l_diag  = self.net_ld(y, der)

        V, der_V = self.net_g(y, der)
        V = V.squeeze()
        g = der_V.squeeze()

        l     = torch.cat((l_diag, l_lower), 1)[:, self._idx]
        der_l = torch.cat((der_l_diag, der_l_lower), 1)[:, self._idx, :]

        L  = self.low_tri(l)
        LT = L.transpose(1, 2)
        H  = torch.matmul(L, LT) + self._epsilon * torch.eye(self.n_dof).type_as(L)

        Ldt = self.low_tri(torch.matmul(der_l, qd_3d).view(-1, self.m))
        Hdt = torch.matmul(L, Ldt.transpose(1, 2)) + torch.matmul(Ldt, LT)

        Ldq = self.low_tri(der_l.transpose(2, 1).reshape(-1, self.m)) \
                .reshape(-1, self.n_dof, self.n_dof, self.n_dof)
        Hdq = torch.matmul(Ldq, LT.view(-1, 1, self.n_dof, self.n_dof)) + \
              torch.matmul(L.view(-1, 1, self.n_dof, self.n_dof), Ldq.transpose(2, 3))

        Hdt_qd  = torch.matmul(Hdt, qd_3d).view(-1, self.n_dof)
        quad_dq = torch.matmul(qd_4d.transpose(2, 3),
                               torch.matmul(Hdq, qd_4d)).view(-1, self.n_dof)
        c = Hdt_qd - 0.5 * quad_dq

        H_qdd = torch.matmul(H, qdd.view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        tau_pred = H_qdd + c + g

        H_qd = torch.matmul(H, qd_3d).view(-1, self.n_dof)
        T = 0.5 * torch.matmul(qd_4d.transpose(2, 3),
                               H_qd.view(-1, 1, self.n_dof, 1)).view(-1)

        qd_H_qdd  = torch.matmul(qd_4d.transpose(2, 3),
                                 H_qdd.view(-1, 1, self.n_dof, 1)).view(-1)
        qd_Hdt_qd = torch.matmul(qd_4d.transpose(2, 3),
                                 Hdt_qd.view(-1, 1, self.n_dof, 1)).view(-1)
        dTdt = qd_H_qdd + 0.5 * qd_Hdt_qd
        dVdt = torch.matmul(qd_4d.transpose(2, 3),
                            g.view(-1, 1, self.n_dof, 1)).view(-1)
        return tau_pred, H, c, g, T, V, dTdt, dVdt

    # --------------------
    def inv_dyn(self, q, qd, qdd):
        return self._dyn_model(q, qd, qdd)[0]

    def for_dyn(self, q, qd, tau):
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        H, c, g = out[1], out[2], out[3]
        invH = torch.inverse(H)
        return torch.matmul(invH, (tau - c - g).view(-1, self.n_dof, 1)).view(-1, self.n_dof)

    def energy(self, q, qd):
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        return out[4] + out[5]

    def energy_dot(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        return out[6] + out[7]

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
