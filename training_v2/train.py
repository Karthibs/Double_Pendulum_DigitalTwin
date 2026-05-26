"""
Training DeLaN on a dataset of filtered (q, qd, qdd, tau) tuples.

Loss (matches Lutter's ICLR 2019 setup):

    L = MSE( tau_pred, tau_meas )                       # inverse-dynamics
      + lambda_E * MSE( dE/dt, P_dissipated )           # energy conservation

For a frictionless system Lutter sets P_dissipated = qd^T tau, since power
input equals rate of change of total energy.  When we add friction, the
correct expression is dE/dt = qd^T (tau - F(qd)). Both versions are
implemented below; pick `energy_loss="ideal"` (Lutter) or `"with_friction"`
(more accurate when friction is significant).

The second term is what Lutter calls the "energy regularizer" in the paper;
it disambiguates the otherwise non-unique Lagrangian (Eq. 9 in the paper).
"""
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def evaluate(model, dataset, batch_size=4096):
    """Mean abs error on tau (inverse) and qdd (forward)."""
    model.eval()
    n = len(dataset)
    sum_abs_tau = 0.0; sum_abs_qdd = 0.0
    sum_sq_tau  = 0.0; sum_sq_qdd  = 0.0
    count = 0
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        q   = dataset.q[s:e]
        qd  = dataset.qd[s:e]
        qdd = dataset.qdd[s:e]
        tau = dataset.tau[s:e]
        with torch.no_grad():
            tau_pred = model.inv_dyn(q, qd, qdd)
            qdd_pred = model.for_dyn(q, qd, tau)
        sum_abs_tau += (tau_pred - tau).abs().sum().item()
        sum_abs_qdd += (qdd_pred - qdd).abs().sum().item()
        sum_sq_tau  += ((tau_pred - tau) ** 2).sum().item()
        sum_sq_qdd  += ((qdd_pred - qdd) ** 2).sum().item()
        count += (e - s)
    return {
        "mae_tau":  sum_abs_tau / (count * model.n_dof),
        "mae_qdd":  sum_abs_qdd / (count * model.n_dof),
        "rmse_tau": (sum_sq_tau / (count * model.n_dof)) ** 0.5,
        "rmse_qdd": (sum_sq_qdd / (count * model.n_dof)) ** 0.5,
    }


def train(model, train_set, val_set,
          epochs=500, batch_size=512, lr=1e-3, weight_decay=1e-5,
          lambda_E=0.1, energy_loss="with_friction",
          log_every=10, val_every=10, ckpt_path=None,
          device="cpu", verbose=True):
    """Train DeLaN.

    Parameters
    ----------
    model : DeepLagrangianNetwork
    train_set, val_set : DoublePendulumDataset (must expose .q, .qd, .qdd, .tau tensors)
    epochs, batch_size, lr, weight_decay : standard
    lambda_E : weight on the energy-conservation regularizer (0 disables it)
    energy_loss : "ideal" or "with_friction"
        ideal:           target dE/dt = qd^T tau  (Lutter, frictionless)
        with_friction:   target dE/dt = qd^T (tau - F(qd))  (real systems)
    ckpt_path : str or None
        If provided, save best-by-val-MAE-tau checkpoint here.
    device : "cpu" | "mps" | "cuda"
        On Mac use "cpu" (safest) or "mps" (Apple Silicon).
    """
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    hist = {"epoch": [], "loss": [], "L_inv": [], "L_E": [],
            "val_mae_tau": [], "val_mae_qdd": [],
            "val_rmse_tau": [], "val_rmse_qdd": [], "val_epoch": []}
    best_val = float("inf")
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = ep_inv = ep_E = 0.0; n_batches = 0
        for q, qd, qdd, tau in loader:
            tau_pred, dEdt = model(q, qd, qdd)
            L_inv = F.mse_loss(tau_pred, tau)

            if lambda_E > 0:
                # Power being delivered to the system at this instant.
                # Ideal (Lutter): power = qd^T tau
                # With friction: power = qd^T (tau - F(qd))
                if energy_loss == "with_friction" and model.learn_friction:
                    P_in = (qd * (tau - model.friction(qd))).sum(dim=-1)
                else:
                    P_in = (qd * tau).sum(dim=-1)
                L_E = F.mse_loss(dEdt, P_in)
            else:
                L_E = torch.tensor(0.0, device=q.device)

            loss = L_inv + lambda_E * L_E
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            ep_loss += loss.detach().item()
            ep_inv  += L_inv.detach().item()
            ep_E    += L_E.detach().item() if isinstance(L_E, torch.Tensor) else float(L_E)
            n_batches += 1

        sched.step()
        hist["epoch"].append(ep)
        hist["loss"].append(ep_loss / n_batches)
        hist["L_inv"].append(ep_inv / n_batches)
        hist["L_E"].append(ep_E / n_batches)

        if ep % val_every == 0 or ep == epochs:
            stats = evaluate(model, val_set)
            hist["val_epoch"].append(ep)
            hist["val_mae_tau"].append(stats["mae_tau"])
            hist["val_mae_qdd"].append(stats["mae_qdd"])
            hist["val_rmse_tau"].append(stats["rmse_tau"])
            hist["val_rmse_qdd"].append(stats["rmse_qdd"])
            if stats["mae_tau"] < best_val:
                best_val = stats["mae_tau"]
                if ckpt_path is not None:
                    torch.save({"model": model.state_dict(),
                                "epoch": ep, "val": stats}, ckpt_path)
            if verbose and (ep % log_every == 0 or ep in (1, epochs)):
                print(f"[ep {ep:4d}] loss={hist['loss'][-1]:.4e} "
                      f"L_inv={hist['L_inv'][-1]:.4e} L_E={hist['L_E'][-1]:.4e}"
                      f"  | val MAE tau={stats['mae_tau']:.4e} qdd={stats['mae_qdd']:.4e}"
                      f"  ({time.time()-t0:.0f}s)")
        else:
            if verbose and (ep % log_every == 0 or ep == 1):
                print(f"[ep {ep:4d}] loss={hist['loss'][-1]:.4e} "
                      f"L_inv={hist['L_inv'][-1]:.4e} L_E={hist['L_E'][-1]:.4e}")

    return hist
