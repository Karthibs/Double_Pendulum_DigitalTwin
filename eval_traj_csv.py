"""
eval_traj_csv.py
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork


def build_model_from_ckpt(state, device):
    hyper = dict(state["hyper"])
    n_dof = hyper.pop("n_dof", 2)
    if "friction_type" not in hyper:
        hyper["friction_type"] = "rayleigh" if hyper.get("use_friction", False) else None
    model = DeepLagrangianNetwork(n_dof, **hyper).to(device)
    model.load_state_dict(state["state_dict"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/delan_model_formula.torch")
    ap.add_argument("--csv",  required=True)
    ap.add_argument("--out",  default="eval_traj.png")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--smooth", type=int, default=0,
                    help="Savitzky-Golay filter window size for velocity smoothing (0 to disable)")
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        model = build_model_from_ckpt(state, device)
    else:
        model = state.to(device)
    model.eval()
    friction_type = getattr(model, "friction_type", None)
    needs_grad = (friction_type == "rayleigh")

    df = pd.read_csv(args.csv).sort_values("time").reset_index(drop=True)
    t  = df["time"].to_numpy()
    qp = df[["pos1", "pos2"]].to_numpy()
    qv = df[["vel1", "vel2"]].to_numpy()
    tau = df[["tau1", "tau2"]].to_numpy()

    if args.smooth and args.smooth >= 5:
        qv = np.stack([savgol_filter(qv[:, i], args.smooth, 3)
                       for i in range(qv.shape[1])], axis=1)
    qa = np.gradient(qv, t, axis=0)

    N = len(t)
    tau_hat   = np.zeros_like(tau)
    dEdt_hat  = np.zeros(N)
    tauf_hat  = np.zeros_like(tau)
    pdiss_hat = np.zeros(N)

    for s in range(0, N, args.chunk):
        e = min(s + args.chunk, N)
        q  = torch.from_numpy(qp[s:e]).float().to(device)
        qd = torch.from_numpy(qv[s:e]).float().to(device)
        if needs_grad:
            qd = qd.requires_grad_(True)
        qdd = torch.from_numpy(qa[s:e]).float().to(device)

        ctx = torch.enable_grad() if needs_grad else torch.no_grad()
        with ctx:
            out = model(q, qd, qdd)

        if len(out) == 4:
            th, de, tf, pd_ = out
        else:
            th, de = out[0], out[1]
            tf  = torch.zeros_like(th)
            pd_ = torch.zeros(e - s, 1, device=device)

        tau_hat[s:e]   = th.detach().cpu().numpy()
        dEdt_hat[s:e]  = de.detach().cpu().numpy().ravel()
        tauf_hat[s:e]  = tf.detach().cpu().numpy()
        pdiss_hat[s:e] = pd_.detach().cpu().numpy().ravel()

    test_dEdt = np.sum(tau * qv, axis=1)
    mse_tau   = np.mean((tau_hat - tau) ** 2)
    rmse_j    = np.sqrt(np.mean((tau_hat - tau) ** 2, axis=0))
    nrmse_j   = rmse_j / (tau.max(0) - tau.min(0) + 1e-9)
    mse_dEdt  = np.mean((dEdt_hat - test_dEdt) ** 2)
    energy_consistency = np.mean((dEdt_hat + pdiss_hat - test_dEdt) ** 2)
    neg_ratio = (pdiss_hat < -1e-8).mean() * 100

    speed = np.linalg.norm(qv, axis=1)
    thr = np.quantile(speed, 0.3)
    lo, hi = speed <= thr, speed > thr
    mse_lo = float(np.mean((tau_hat[lo] - tau[lo]) ** 2)) if lo.any() else float('nan')
    mse_hi = float(np.mean((tau_hat[hi] - tau[hi]) ** 2)) if hi.any() else float('nan')

    print(f"friction_type   = {friction_type}")
    print(f"N               = {N}")
    print(f"MSE τ           = {mse_tau:.3e}")
    print(f"RMSE per joint  = {rmse_j}")
    print(f"NRMSE per joint = {nrmse_j}")
    print(f"MSE dE/dt       = {mse_dEdt:.3e}")
    print(f"Energy balance MSE (dÊ/dt+P_diss-τᵀq̇)² = {energy_consistency:.3e}")
    print(f"mean |P_diss|   = {np.mean(np.abs(pdiss_hat)):.3e}")
    #print(f"P_diss<0 ratio  = {neg_ratio:.2f}%")
    print(f"MSE τ low-speed (|q̇|≤{thr:.3f}) = {mse_lo:.3e}")
    print(f"MSE τ high-speed                = {mse_hi:.3e}")

    if friction_type == "formula" and hasattr(model, "friction_formula"):
        tc, ts, nu, d, eps = model.friction_formula.params()
        print("\nLearned friction parameters (per joint):")
        print(f"  τ_c = {tc.detach().cpu().numpy()}")
        print(f"  τ_s = {ts.detach().cpu().numpy()}")
        print(f"  ν   = {nu.detach().cpu().numpy()}")
        print(f"  d   = {d.detach().cpu().numpy()}")
        print(f"  ε   = {eps.detach().cpu().numpy()}")

    print("\nSample predictions vs measurements:")
    for i in np.linspace(0, N - 1, min(10, N), dtype=int):
        print(f"t={t[i]:.3f}s: "
              f"τ_meas=[{tau[i,0]:.4f},{tau[i,1]:.4f}], "
              f"τ̂=[{tau_hat[i,0]:.4f},{tau_hat[i,1]:.4f}], "
              f"τ_f=[{tauf_hat[i,0]:.4f},{tauf_hat[i,1]:.4f}], "
              f"P_diss={pdiss_hat[i]:.3e}")

    # visualize results:
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for j in range(tau.shape[1]):
        ax[0].plot(t, tau[:, j],     label=f"meas τ{j+1}")
        ax[0].plot(t, tau_hat[:, j], "--", label=f"DeLaN τ{j+1}")
    ax[0].set_ylabel("torque [Nm]"); ax[0].legend(ncol=2); ax[0].grid(True)

    ax[1].plot(t, test_dEdt, label="meas τᵀq̇")
    ax[1].plot(t, dEdt_hat, "--", label="DeLaN dE/dt")
    ax[1].set_ylabel("power [W]"); ax[1].legend(); ax[1].grid(True)

    ax[2].plot(t, pdiss_hat, label="P_diss")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("P_diss [W]"); ax[2].legend(); ax[2].grid(True)

    for j in range(tau.shape[1]):
        ax[3].plot(t, tauf_hat[:, j], label=f"τ_f{j+1}")
    ax[3].set_xlabel("time [s]"); ax[3].set_ylabel("friction τ_f [Nm]")
    ax[3].legend(); ax[3].grid(True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.show()
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
