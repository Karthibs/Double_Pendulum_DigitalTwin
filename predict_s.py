# eval_traj_csv.py
import argparse, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/delan_model.torch")
    ap.add_argument("--csv",  required=True)
    ap.add_argument("--out",  default="eval_traj.png")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--smooth", type=int, default=0, help="Savitzky-Golay 窗口(奇数),0=关")
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")


    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        hyper = state["hyper"]
        n_dof = hyper.pop("n_dof", 2)
        model = DeepLagrangianNetwork(n_dof, **hyper).to(device)
        model.load_state_dict(state["state_dict"])
    else:
        model = state.to(device)
    model.eval()


    df = pd.read_csv(args.csv).sort_values("time").reset_index(drop=True)
    t  = df["time"].to_numpy()
    qp = df[["pos_meas1","pos_meas2"]].to_numpy()
    qv = df[["vel_meas1","vel_meas2"]].to_numpy()
    tau= df[["tau_meas1","tau_meas2"]].to_numpy()
    #qp = df[["pos1", "pos2"]].to_numpy()
    #qv = df[["vel1", "vel2"]].to_numpy()
    #tau= df[["tau1", "tau2"]].to_numpy()
    if args.smooth and args.smooth >= 5:
        qv = np.stack([savgol_filter(qv[:,i], args.smooth, 3) for i in range(qv.shape[1])], axis=1)

    qa = np.gradient(qv, t, axis=0)  # 数值微分得 q̈


    N = len(t)
    tau_hat   = np.zeros_like(tau)
    dEdt_hat  = np.zeros(N)
    tauf_hat  = np.zeros_like(tau)
    pdiss_hat = np.zeros(N)

    for s in range(0, N, args.chunk):
        e = min(s + args.chunk, N)
        q  = torch.from_numpy(qp[s:e]).float().to(device)
        qd = torch.from_numpy(qv[s:e]).float().to(device).requires_grad_(True)
        qdd= torch.from_numpy(qa[s:e]).float().to(device)
        with torch.enable_grad():
            out = model(q, qd, qdd)

        if len(out) == 4:
            th, de, tf, pd_ = out
        else:
            th, de = out[0], out[1]
            tf = torch.zeros_like(th); pd_ = torch.zeros(e-s,1,device=device)
        tau_hat[s:e]  = th.detach().cpu().numpy()
        dEdt_hat[s:e] = de.detach().cpu().numpy().ravel()
        tauf_hat[s:e] = tf.detach().cpu().numpy()
        pdiss_hat[s:e]= pd_.detach().cpu().numpy().ravel()


    test_dEdt = np.sum(tau * qv, axis=1)
    mse_tau   = np.mean((tau_hat - tau)**2)
    rmse_j    = np.sqrt(np.mean((tau_hat - tau)**2, axis=0))
    nrmse_j   = rmse_j / (tau.max(0) - tau.min(0) + 1e-9)
    mse_dEdt  = np.mean((dEdt_hat - test_dEdt)**2)
    energy_consistency = np.mean((dEdt_hat + pdiss_hat - test_dEdt)**2)
    neg_ratio = (pdiss_hat < -1e-8).mean() * 100

    print(f"N = {N}")
    print(f"MSE  τ          = {mse_tau:.3e}")
    print(f"RMSE per joint  = {rmse_j}")
    print(f"NRMSE per joint = {nrmse_j}")
    print(f"MSE  dE/dt      = {mse_dEdt:.3e}")
    print(f"Energy balance MSE (dÊ/dt+P_diss-τᵀq̇)² = {energy_consistency:.3e}")
    print(f"mean |P_diss|   = {np.mean(np.abs(pdiss_hat)):.3e}")
    print(f"P_diss<0 %    = {neg_ratio:.2f}%")
    # sample predictions
    print("\nSample predictions vs measurements:")
    for i in np.linspace(0, N-1, min(10, N), dtype=int):
        #print sample
        print(f"t={t[i]:.3f}s: τ_meas=[{tau[i][0].item():.4f}, {tau[i][1].item():.4f}], τ_DeLaN=[{tau_hat[i][0].item():.4f}, {tau_hat[i][1].item():.4f}], dE/dt_meas={test_dEdt[i]:.4f}, dE/dt_DeLaN={dEdt_hat[i]:.4f}, P_diss={pdiss_hat[i]:.4e}")

    # visualize:
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for j in range(tau.shape[1]):
        ax[0].plot(t, tau[:,j],     label=f"meas τ{j+1}")
        ax[0].plot(t, tau_hat[:,j], "--", label=f"DeLaN τ{j+1}")
    ax[0].set_ylabel("torque [Nm]"); ax[0].legend(ncol=2); ax[0].grid(True)

    ax[1].plot(t, test_dEdt, label="meas τᵀq̇")
    ax[1].plot(t, dEdt_hat,  "--", label="DeLaN dE/dt")
    ax[1].set_ylabel("power [W]"); ax[1].legend(); ax[1].grid(True)

    ax[2].plot(t, pdiss_hat, label="P_diss (friction)")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("P_diss [W]")
    ax[2].legend(); ax[2].grid(True)

    plt.tight_layout(); plt.savefig(args.out, dpi=150); plt.show()
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()
