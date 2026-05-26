import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from training_v2.evaluation import rollout, total_energy


def compare_csv_to_model(csv_path, model, dt=None,
                         rollout_horizon=None,
                         integrator='rk4',
                         vpt_threshold_mse=1e-2,
                         show=True):
    # ---- Load data ----
    df = pd.read_csv(csv_path)
    name = os.path.basename(csv_path)
    t_meas = df['time'].to_numpy()
    X_meas = df[['pos1', 'pos2', 'vel1', 'vel2']].to_numpy()
    tau    = df[['tau1', 'tau2']].to_numpy()

    if dt is None:
        dt = float(np.median(np.diff(t_meas)))

    N = len(tau)
    if rollout_horizon is not None:
        N = min(N, int(rollout_horizon / dt))

    # ---- Rollout ----
    model.eval()
    t_pred, X_pred = rollout(model, X_meas[0], dt, tau[:N], integrator=integrator)
    n = min(len(t_pred), len(X_meas), N + 1)
    t_pred = t_pred[:n]; X_pred = X_pred[:n]; X_meas = X_meas[:n]

    # ---- VPT (Valid Prediction Time) ----
    pos_mse = np.mean((X_pred[:, :2] - X_meas[:, :2]) ** 2, axis=1)
    over = np.where(pos_mse > vpt_threshold_mse)[0]
    vpt = float(t_pred[over[0]]) if len(over) else float(t_pred[-1])

    # ---- Position error (wrap to [-pi, pi]) ----
    dq = X_pred[:, :2] - X_meas[:, :2]
    dq = (dq + np.pi) % (2 * np.pi) - np.pi
    pos_err = np.linalg.norm(dq, axis=1)

    # ---- Energy along the rollout (from the model) ----
    E_pred, T_pred, V_pred = total_energy(model, X_pred)
    E_meas, _, _           = total_energy(model, X_meas)

    # ---- Plot ----
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
    # Row 0: q1, q2
    for j, lbl in enumerate(['q1', 'q2']):
        ax = axes[0, j]
        ax.plot(t_pred, X_meas[:, j],         color='C0', lw=1.4, label='measured')
        ax.plot(t_pred, X_pred[:, j], '--',   color='C3', lw=1.2, label='LNN rollout')
        ax.set_ylabel(f'{lbl} [rad]')
        ax.grid(True, alpha=0.3)
        if j == 0: ax.legend(loc='best', fontsize=9)

    # Row 1: dq1, dq2
    for j, lbl in enumerate(['dq1', 'dq2']):
        ax = axes[1, j]
        ax.plot(t_pred, X_meas[:, 2+j],       color='C0', lw=1.4)
        ax.plot(t_pred, X_pred[:, 2+j], '--', color='C3', lw=1.2)
        ax.set_ylabel(f'{lbl} [rad/s]')
        ax.grid(True, alpha=0.3)

    # Row 2: position error + VPT marker
    ax = axes[2, 0]
    ax.semilogy(t_pred, pos_err, color='C2', lw=1.2)
    ax.axhline(np.sqrt(vpt_threshold_mse * 2), color='r', ls=':', lw=0.8,
               label=f'VPT thresh')
    ax.axvline(vpt, color='r', ls='--', lw=0.8, label=f'VPT = {vpt:.2f}s')
    ax.set_ylabel('|pos err| [rad]')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Position MSE
    ax = axes[2, 1]
    ax.semilogy(t_pred, pos_mse, color='C2', lw=1.2)
    ax.axhline(vpt_threshold_mse, color='r', ls=':', lw=0.8, label='1e-2')
    ax.axvline(vpt, color='r', ls='--', lw=0.8)
    ax.set_ylabel('pos MSE [rad^2]')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Row 3: total energy
    ax = axes[3, 0]
    ax.plot(t_pred, E_meas, color='C0', lw=1.4, label='E along measured')
    ax.plot(t_pred, E_pred, '--', color='C3', lw=1.2, label='E along rollout')
    ax.set_ylabel('total E [J]')
    ax.set_xlabel('t [s]')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Kinetic / potential split for the rollout
    ax = axes[3, 1]
    ax.plot(t_pred, T_pred, color='C1', lw=1.2, label='T(rollout)')
    ax.plot(t_pred, V_pred, color='C4', lw=1.2, label='V(rollout)')
    ax.set_ylabel('T, V [J]')
    ax.set_xlabel('t [s]')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'{name}    dt={dt*1000:.2f}ms    N={n}    horizon={t_pred[-1]:.2f}s    VPT={vpt:.2f}s',
        fontsize=11, y=0.995,
    )
    plt.tight_layout()
    if show: plt.show()

    # ---- Numeric summary ----
    rmse_pos_05 = float(np.sqrt(np.mean(pos_mse[t_pred <= 0.5] ** 2))) if (t_pred[-1] >= 0.5) else float('nan')
    rmse_pos_10 = float(np.sqrt(np.mean(pos_mse[t_pred <= 1.0] ** 2))) if (t_pred[-1] >= 1.0) else float('nan')
    rmse_pos_20 = float(np.sqrt(np.mean(pos_mse[t_pred <= 2.0] ** 2))) if (t_pred[-1] >= 2.0) else float('nan')

    print(f'\n{name}:')
    print(f'  dt:               {dt*1000:.3f} ms')
    print(f'  rollout horizon:  {t_pred[-1]:.2f} s ({n} steps)')
    print(f'  VPT (MSE<{vpt_threshold_mse}): {vpt:.2f} s')
    print(f'  RMSE pos @ 0.5s:  {rmse_pos_05:.4f} rad')
    print(f'  RMSE pos @ 1.0s:  {rmse_pos_10:.4f} rad')
    print(f'  RMSE pos @ 2.0s:  {rmse_pos_20:.4f} rad')
    print(f'  energy drift:     {abs(E_pred[-1]-E_pred[0]):.4e} J  '
          f'(rel: {abs(E_pred[-1]-E_pred[0])/max(abs(E_pred[0]), 1e-9):.2%})')

    return {
        't': t_pred, 'X_meas': X_meas, 'X_pred': X_pred,
        'pos_err': pos_err, 'pos_mse': pos_mse,
        'E_pred': E_pred, 'E_meas': E_meas,
        'T_pred': T_pred, 'V_pred': V_pred,
        'vpt': vpt, 'dt': dt,
    }


def plot_inverse_dynamics(csv_path, model, title=None, show=True,
                          time_range=None):
    """Robust inverse-dynamics plotter."""
    df = pd.read_csv(csv_path)
    needed = ['time', 'pos1', 'pos2', 'vel1', 'vel2',
              'acc1', 'acc2', 'tau1', 'tau2']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    # Drop any NaN rows (just in case)
    df = df.dropna(subset=needed).reset_index(drop=True)

    if time_range is not None:
        t0, t1 = time_range
        df = df[(df['time'] >= t0) & (df['time'] <= t1)].reset_index(drop=True)

    if len(df) < 2:
        raise ValueError('Not enough data to plot')

    # Use sample index instead of time on x-axis to avoid timing glitches
    # disrupting the plot. We'll show time on the title instead.
    idx = np.arange(len(df))
    t   = df['time'].to_numpy()
    q   = torch.tensor(df[['pos1', 'pos2']].to_numpy(), dtype=torch.float32)
    qd  = torch.tensor(df[['vel1', 'vel2']].to_numpy(), dtype=torch.float32)
    qdd = torch.tensor(df[['acc1', 'acc2']].to_numpy(), dtype=torch.float32)
    tau_meas = df[['tau1', 'tau2']].to_numpy()

    model.eval()
    with torch.no_grad():
        tau_pred = model.inv_dyn(q, qd, qdd).cpu().numpy()

    err = tau_pred - tau_meas
    mae  = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    peak = np.max(np.abs(tau_meas), axis=0) + 1e-12

    # Two figures: time series, and parity.
    # Plot against sample index (uniform x-axis) - no surprises from time glitches
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    # Row 0: tau1, tau2 overlaid
    for j, joint in enumerate(['tau1', 'tau2']):
        ax = axes[0, j]
        ax.plot(idx, tau_meas[:, j], color='C0', lw=1.0, label='measured')
        ax.plot(idx, tau_pred[:, j], '--', color='C3', lw=1.0, label='predicted')
        ax.set_xlabel('sample index')
        ax.set_ylabel(f'{joint} [Nm]')
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        ax.set_title(f'{["shoulder","elbow"][j]}  MAE={mae[j]:.4f} '
                     f'({mae[j]/peak[j]:.1%} of peak)')

    # Row 1: error signal
    for j, joint in enumerate(['tau1', 'tau2']):
        ax = axes[1, j]
        ax.plot(idx, err[:, j], color='C2', lw=1.0)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('sample index')
        ax.set_ylabel(f'{joint} error [Nm]')
        ax.grid(alpha=0.3)
        ax.set_title(f'pred - measured   RMSE={rmse[j]:.4f}')

    name = title or os.path.basename(csv_path)
    fig.suptitle(
        f'Inverse dynamics: {name}    '
        f'({len(df)} samples, t=[{t.min():.2f}, {t.max():.2f}]s)',
        fontsize=11,
    )
    plt.tight_layout()
    if show: plt.show()

    # Parity in a second figure
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 5))
    for j, joint in enumerate(['tau1', 'tau2']):
        ax = ax2[j]
        ax.scatter(tau_meas[:, j], tau_pred[:, j], s=3, alpha=0.4, color='C0')
        lim = max(peak[j], np.max(np.abs(tau_pred[:, j]))) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'r--', lw=0.8, label='y=x')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel(f'measured {joint} [Nm]')
        ax.set_ylabel(f'predicted {joint} [Nm]')
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        ax.set_title(f'{joint} parity')
    fig2.suptitle(f'Parity: {name}', fontsize=11)
    plt.tight_layout()
    if show: plt.show()

    # Numerical summary
    print(f'\n{name}')
    print(f'  samples:    {len(df)}')
    print(f'  time range: [{t.min():.2f}, {t.max():.2f}] s  '
          f'(duration {t.max()-t.min():.2f}s)')
    print(f'  median dt:  {np.median(np.diff(t))*1000:.2f} ms')
    print(f'  tau1 MAE:   {mae[0]:.5f} Nm  ({mae[0]/peak[0]:.2%} of peak {peak[0]:.4f})')
    print(f'  tau2 MAE:   {mae[1]:.5f} Nm  ({mae[1]/peak[1]:.2%} of peak {peak[1]:.4f})')

    return {'t': t, 'tau_meas': tau_meas, 'tau_pred': tau_pred,
            'error': err, 'mae': mae, 'rmse': rmse}