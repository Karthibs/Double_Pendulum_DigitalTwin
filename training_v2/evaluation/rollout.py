
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def _f(model, q, qd, tau):
    q_t   = torch.as_tensor(q,   dtype=torch.float32).unsqueeze(0)
    qd_t  = torch.as_tensor(qd,  dtype=torch.float32).unsqueeze(0)
    tau_t = torch.as_tensor(tau, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        qdd = model.for_dyn(q_t, qd_t, tau_t)
    return qdd.cpu().numpy().squeeze(0)


def rollout(model, x0, dt, tau_seq, integrator="rk4"):
    """Integrate the learned model from x0 for len(tau_seq) steps.
    Parameters
    ----------
    model : DeepLagrangianNetwork (eval mode)
    x0 : (4,) array - [q1, q2, q1d, q2d]
    dt : float
    tau_seq : (N, 2) torques to apply at each step
    integrator : "euler" | "rk4"
    """
    model.eval()
    N = len(tau_seq)
    X = np.zeros((N + 1, 4)); X[0] = np.asarray(x0, dtype=float)
    t = np.arange(N + 1) * dt
    for k in range(N):
        q, qd = X[k, :2], X[k, 2:]
        tau = tau_seq[k]
        if integrator == "euler":
            qdd = _f(model, q, qd, tau)
            X[k + 1, :2] = q + dt * qd
            X[k + 1, 2:] = qd + dt * qdd
        elif integrator == "rk4":
            def deriv(qk, qdk):
                qddk = _f(model, qk, qdk, tau)
                return qdk, qddk
            k1q, k1qd = deriv(q,                qd)
            k2q, k2qd = deriv(q + 0.5*dt*k1q,   qd + 0.5*dt*k1qd)
            k3q, k3qd = deriv(q + 0.5*dt*k2q,   qd + 0.5*dt*k2qd)
            k4q, k4qd = deriv(q + dt*k3q,       qd + dt*k3qd)
            X[k + 1, :2] = q  + (dt/6.0)*(k1q  + 2*k2q  + 2*k3q  + k4q)
            X[k + 1, 2:] = qd + (dt/6.0)*(k1qd + 2*k2qd + 2*k3qd + k4qd)
        else:
            raise ValueError(integrator)
    return t, X


# ---------------------------------------------------------------------------
# One-step metrics
# ---------------------------------------------------------------------------
def one_step_metrics(model, dataset, batch_size=4096):
    model.eval()
    taus, taus_pred, qdds, qdds_pred = [], [], [], []
    n = len(dataset)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        q   = dataset.q[s:e]
        qd  = dataset.qd[s:e]
        qdd = dataset.qdd[s:e]
        tau = dataset.tau[s:e]
        with torch.no_grad():
            tau_pred = model.inv_dyn(q, qd, qdd)
            qdd_pred = model.for_dyn(q, qd, tau)
        taus.append(tau.cpu().numpy());     taus_pred.append(tau_pred.cpu().numpy())
        qdds.append(qdd.cpu().numpy());     qdds_pred.append(qdd_pred.cpu().numpy())
    tau_all      = np.concatenate(taus);     tau_pred_all = np.concatenate(taus_pred)
    qdd_all      = np.concatenate(qdds);     qdd_pred_all = np.concatenate(qdds_pred)
    err_tau = tau_pred_all - tau_all
    err_qdd = qdd_pred_all - qdd_all
    return {
        "tau":      tau_all,      "tau_pred": tau_pred_all,
        "qdd":      qdd_all,      "qdd_pred": qdd_pred_all,
        "mae_tau":  float(np.mean(np.abs(err_tau))),
        "rmse_tau": float(np.sqrt(np.mean(err_tau ** 2))),
        "mae_qdd":  float(np.mean(np.abs(err_qdd))),
        "rmse_qdd": float(np.sqrt(np.mean(err_qdd ** 2))),
        "mae_tau_joint": np.mean(np.abs(err_tau), axis=0),
        "mae_qdd_joint": np.mean(np.abs(err_qdd), axis=0),
    }


# ---------------------------------------------------------------------------
# Rollout metrics
# ---------------------------------------------------------------------------
def rollout_metrics(t, X_pred, X_meas, divergence_thresh=0.3):
    n = min(len(t), len(X_pred), len(X_meas))
    t = t[:n]; X_pred = X_pred[:n]; X_meas = X_meas[:n]
    dq = X_pred[:, :2] - X_meas[:, :2]
    dq = (dq + np.pi) % (2 * np.pi) - np.pi
    err_pos = np.linalg.norm(dq, axis=1)
    err_vel = np.linalg.norm(X_pred[:, 2:] - X_meas[:, 2:], axis=1)

    res = {"t": t, "err_pos": err_pos, "err_vel": err_vel}
    for horizon in (0.5, 1.0, 2.0):
        idx = np.searchsorted(t, horizon)
        if 0 < idx < len(t):
            res[f"rmse_pos_at_{horizon}s"] = float(np.sqrt(np.mean(err_pos[:idx + 1] ** 2)))
    diverged = np.where(err_pos > divergence_thresh)[0]
    res["time_to_divergence"] = float(t[diverged[0]]) if len(diverged) else float(t[-1])
    return res


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
def total_energy(model, X):
    """Returns E (=T+V), T, V along the trajectory using the learned Lagrangian."""
    model.eval()
    q  = torch.as_tensor(X[:, :2], dtype=torch.float32)
    qd = torch.as_tensor(X[:, 2:], dtype=torch.float32)
    with torch.no_grad():
        T = model.kinetic(q, qd).cpu().numpy()
        V = model.potential(q).cpu().numpy()
    return T + V, T, V
