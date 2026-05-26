import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualize_csv(csv_path, q_limit=4*np.pi, save=False):
    df = pd.read_csv(csv_path)
    name = os.path.basename(csv_path)
    t  = df['time'].to_numpy()
    q  = df[['pos1', 'pos2']].to_numpy()
    qd = df[['vel1', 'vel2']].to_numpy()
    tau = df[['tau1', 'tau2']].to_numpy()

    # Where are the outliers?
    bad = (np.abs(q) > q_limit).any(axis=1)
    n_bad = int(bad.sum())

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)

    # --- Row 0: positions ---
    for j, label in enumerate(['q1', 'q2']):
        ax = axes[0, j]
        ax.plot(t, q[:, j], color='C0', lw=1.0)
        if n_bad:
            ax.scatter(t[bad], q[bad, j], color='red', s=10, zorder=5,
                       label=f'|q|>{q_limit:.1f}')
            ax.legend(loc='best', fontsize=8)
        # reference lines at +/- pi and +/- 2*pi
        for y in (-2*np.pi, -np.pi, np.pi, 2*np.pi):
            ax.axhline(y, color='gray', ls=':', lw=0.4)
        ax.axhline(0, color='k', ls='--', lw=0.4)
        ax.set_ylabel(f'{label} [rad]')
        ax.grid(True, alpha=0.3)

    # --- Row 1: velocities ---
    for j, label in enumerate(['dq1', 'dq2']):
        ax = axes[1, j]
        ax.plot(t, qd[:, j], color='C1', lw=1.0)
        if n_bad:
            ax.scatter(t[bad], qd[bad, j], color='red', s=10, zorder=5)
        ax.axhline(0, color='k', ls='--', lw=0.4)
        # hardware velocity limits
        vlim = 40 if j == 0 else 50
        ax.axhline( vlim, color='r', ls=':', lw=0.4)
        ax.axhline(-vlim, color='r', ls=':', lw=0.4)
        ax.set_ylabel(f'{label} [rad/s]')
        ax.grid(True, alpha=0.3)

    # --- Row 2: torques ---
    for j, label in enumerate(['tau1', 'tau2']):
        ax = axes[2, j]
        ax.plot(t, tau[:, j], color='C2', lw=1.0)
        if n_bad:
            ax.scatter(t[bad], tau[bad, j], color='red', s=10, zorder=5)
        ax.axhline( 0.15, color='r', ls=':', lw=0.5, label='torque limit')
        ax.axhline(-0.15, color='r', ls=':', lw=0.5)
        ax.axhline(0, color='k', ls='--', lw=0.4)
        ax.set_xlabel('t [s]')
        ax.set_ylabel(f'{label} [Nm]')
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc='best', fontsize=8)

    title = f'{name}    n={len(df)}    n_bad={n_bad} ({100*n_bad/len(df):.1f}%)'
    fig.suptitle(title, fontsize=11, y=0.995)
    plt.tight_layout()

    if save:
        out = os.path.splitext(csv_path)[0] + '_viz.png'
        fig.savefig(out, dpi=110)
        print(f'  saved: {out}')

    plt.show()

    # ---- summary ----
    print(f'\n{name}:')
    print(f'  duration:    {t[-1]-t[0]:.2f}s ({len(df)} samples, dt~{np.median(np.diff(t))*1000:.1f}ms)')
    print(f'  q1 range:    [{q[:,0].min():+7.3f}, {q[:,0].max():+7.3f}] rad')
    print(f'  q2 range:    [{q[:,1].min():+7.3f}, {q[:,1].max():+7.3f}] rad')
    print(f'  dq1 range:   [{qd[:,0].min():+7.3f}, {qd[:,0].max():+7.3f}] rad/s')
    print(f'  dq2 range:   [{qd[:,1].min():+7.3f}, {qd[:,1].max():+7.3f}] rad/s')
    print(f'  tau1 range:  [{tau[:,0].min():+7.4f}, {tau[:,0].max():+7.4f}] Nm')
    print(f'  tau2 range:  [{tau[:,1].min():+7.4f}, {tau[:,1].max():+7.4f}] Nm')
    if n_bad:
        bad_idx = np.where(bad)[0]
        print(f'  bad samples: {n_bad}')
        print(f'    first bad at t={t[bad_idx[0]]:.3f}s (index {bad_idx[0]})')
        print(f'    last  bad at t={t[bad_idx[-1]]:.3f}s (index {bad_idx[-1]})')
        # show a small context around the first bad sample
        i = bad_idx[0]
        lo = max(0, i-3); hi = min(len(df), i+4)
        print(f'\n  context around first bad sample (rows {lo}..{hi-1}):')
        print(df[['time','pos1','pos2','vel1','vel2','tau1','tau2']]
              .iloc[lo:hi].to_string(index=True))


