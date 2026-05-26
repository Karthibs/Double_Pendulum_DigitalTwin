"""
Generate test reference trajectories that are structurally different from
the training set.

Training set patterns (avoid these):
  - both-sine with similar frequencies on both joints
  - swing-up with smooth-blend pump + min-jerk
  - full circles
  - single-joint motion (only q1 or only q2)

Test trajectories below cover (different) patterns:
  T01: low-freq + high-freq summed (chirp + slow base)
  T02: square-wave-like ramps
  T03: phase-offset sinusoid pair (one joint leads the other by 90deg)
  T04: detuned dual frequency (Lissajous-like)
  T05: amplitude-modulated sine (carrier + slow envelope)
  T06: figure-8 in (q1, q2)
  T07: counter-rotating sinusoids (anti-phase)
  T08: triangular ramp
  T09: 1/f-ish slow random reference (low-pass filtered noise)
  T10: spiral inward (slowly decaying amplitude)

Each trajectory is 20 s at dt=0.002 s. Joint angles, velocities, and
accelerations are returned together so they can be fed to the same PID
controller used during training-data collection.

Hardware limits respected: |q_dot| < 5 rad/s typical, well under 40 rad/s.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


T_FINAL = 20.0
DT = 0.005


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def differentiate(x, dt):
    """Centered finite difference for first derivative, edges via one-sided."""
    xd = np.zeros_like(x)
    xd[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    xd[0]    = (x[1] - x[0]) / dt
    xd[-1]   = (x[-1] - x[-2]) / dt
    return xd


def build_traj(t, q1, q2):
    """Given t, q1(t), q2(t), compute velocities and accelerations and
    return a DataFrame with the same columns as training trajectories."""
    dt = t[1] - t[0]
    qd1 = differentiate(q1, dt); qd2 = differentiate(q2, dt)
    qdd1 = differentiate(qd1, dt); qdd2 = differentiate(qdd1, dt)  # noqa
    qdd1 = differentiate(qd1, dt); qdd2 = differentiate(qd2, dt)
    return pd.DataFrame({
        'time': t,
        'pos1': q1, 'pos2': q2,
        'vel1': qd1, 'vel2': qd2,
        'acc1': qdd1, 'acc2': qdd2,
    })


def lowpass_random(t, cutoff_hz, seed):
    """Generate a smooth random reference: white noise low-pass filtered."""
    rng = np.random.default_rng(seed)
    n = len(t); dt = t[1] - t[0]
    raw = rng.standard_normal(n)
    # Simple cascaded one-pole low-pass for smoothness
    tau = 1.0 / (2 * np.pi * cutoff_hz)
    alpha = dt / (tau + dt)
    out = np.zeros_like(raw)
    out[0] = raw[0]
    for i in range(1, n):
        out[i] = alpha * raw[i] + (1 - alpha) * out[i-1]
    # Run again forward for extra smoothness
    out2 = np.zeros_like(out)
    out2[0] = out[0]
    for i in range(1, n):
        out2[i] = alpha * out[i] + (1 - alpha) * out2[i-1]
    return out2 / (np.max(np.abs(out2)) + 1e-9)


# ---------------------------------------------------------------------------
# Trajectory definitions
# ---------------------------------------------------------------------------
def T01_sumfreq(t):
    """Two summed frequencies on each joint (low + high)."""
    q1 = 2*np.sin(2*np.pi*0.3*t) + 0.25*np.sin(2*np.pi*1.2*t)
    q2 = 0.5*np.sin(2*np.pi*0.4*t) + 0.20*np.sin(2*np.pi*1.5*t + 0.7)
    return q1, q2


def T02_smoothramps(t):
    """Approximate square wave (smoothed) - alternating set-points."""
    period = 4.0
    sharpness = 8.0
    # Smooth square via tanh of sine
    s = np.tanh(sharpness * np.sin(2*np.pi*t/period))
    q1 = 0.8 * s
    q2 = 0.6 * np.tanh(sharpness * np.sin(2*np.pi*t/period - np.pi/3))
    return q1, q2


def T03_phaseoffset(t):
    """One joint leads the other by 90 degrees - circular motion at end-eff."""
    f = 0.5
    q1 = 0.7 * np.sin(2*np.pi*f*t)
    q2 = 0.7 * np.sin(2*np.pi*f*t + np.pi/2)   # exactly 90deg ahead
    return q1, q2


def T04_lissajous(t):
    """Lissajous pattern: detuned frequencies on each joint."""
    q1 = 0.6 * np.sin(2*np.pi*0.4*t)
    q2 = 0.6 * np.sin(2*np.pi*0.6*t + np.pi/4)   # 3:2 ratio with phase
    return q1, q2


def T05_amp_modulated(t):
    """Carrier with slowly varying envelope."""
    carrier_f = 1.0
    env_f = 0.1
    envelope = 0.4 + 0.4*np.sin(2*np.pi*env_f*t)
    q1 = envelope * np.sin(2*np.pi*carrier_f*t)
    q2 = envelope * np.sin(2*np.pi*carrier_f*t + np.pi)   # antiphase elbow
    return q1, q2


def T06_figure8(t):
    """Figure-8 in (q1, q2) joint space."""
    f = 0.3
    q1 = 0.7 * np.sin(2*np.pi*f*t)
    q2 = 0.5 * np.sin(2*np.pi*2*f*t)   # double frequency = lemniscate
    return q1, q2


def T07_counter_rotating(t):
    """Anti-phase joints: shoulder up while elbow down."""
    f = 0.6
    q1 = 0.6 * np.sin(2*np.pi*f*t)
    q2 = -0.6 * np.sin(2*np.pi*f*t)     # negative copy = mirror motion
    return q1, q2


def T08_triangle(t):
    """Triangular ramp using Fourier series (smooth enough for PID)."""
    period = 5.0
    # 5-term Fourier triangle wave
    tri = np.zeros_like(t)
    for k in range(1, 6):
        n = 2*k - 1
        tri += ((-1)**k) * np.sin(2*np.pi*n*t/period) / (n*n)
    tri *= -8.0/(np.pi**2)
    q1 = 0.7 * tri
    q2 = 0.5 * np.roll(tri, len(t)//8)   # phase shifted
    return q1, q2


def T09_lpnoise(t):
    """Low-pass-filtered random reference - tests off-pattern behavior."""
    q1 = 0.6 * lowpass_random(t, cutoff_hz=0.4, seed=42)
    q2 = 0.4 * lowpass_random(t, cutoff_hz=0.4, seed=43)
    return q1, q2


def T10_spiral_decay(t):
    """Circular motion with decaying amplitude (spiral inward)."""
    f = 0.4
    tau = T_FINAL / 2.5
    envelope = np.exp(-t / tau)
    q1 = 0.7 * envelope * np.sin(2*np.pi*f*t)
    q2 = 0.7 * envelope * np.sin(2*np.pi*f*t + np.pi/2)
    return q1, q2


TEST_TRAJECTORIES = {
    'T01':         T01_sumfreq,
    'T02':     T02_smoothramps,
    'T03':     T03_phaseoffset,
    'T04':       T04_lissajous,
    'T05':   T05_amp_modulated,
    'T06':         T06_figure8,
    'T07': T07_counter_rotating,
    'T08':        T08_triangle,
    'T09':         T09_lpnoise,
    'T10':    T10_spiral_decay,
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_all(out_dir='test_trajectories', plot=True):
    os.makedirs(out_dir, exist_ok=True)
    t = np.arange(0, T_FINAL + DT, DT)

    summary = []
    for name, fn in TEST_TRAJECTORIES.items():
        q1, q2 = fn(t)
        df = build_traj(t, q1, q2)
        csv_path = os.path.join(out_dir, f'{name}.csv')
        df.to_csv(csv_path, index=False)

        peak_v1 = float(np.max(np.abs(df['vel1'])))
        peak_v2 = float(np.max(np.abs(df['vel2'])))
        summary.append({
            'name': name, 'csv': csv_path,
            'peak_q1': float(np.max(np.abs(q1))),
            'peak_q2': float(np.max(np.abs(q2))),
            'peak_v1': peak_v1, 'peak_v2': peak_v2,
        })
        print(f'  {name:25s}  peak_q=({np.max(np.abs(q1)):.2f}, {np.max(np.abs(q2)):.2f})'
              f'  peak_v=({peak_v1:.2f}, {peak_v2:.2f}) rad/s  -> {csv_path}')

    if plot:
        n = len(TEST_TRAJECTORIES)
        fig, axes = plt.subplots(n, 3, figsize=(15, 2.2*n), sharex=True)
        for i, (name, fn) in enumerate(TEST_TRAJECTORIES.items()):
            q1, q2 = fn(t)
            df = build_traj(t, q1, q2)
            axes[i, 0].plot(t, q1, label='q1'); axes[i, 0].plot(t, q2, label='q2')
            axes[i, 0].set_ylabel(name[:14]); axes[i, 0].grid(alpha=0.3); axes[i, 0].legend(fontsize=7)
            axes[i, 1].plot(t, df['vel1']); axes[i, 1].plot(t, df['vel2'])
            axes[i, 1].axhline(40, color='r', ls=':', lw=0.5); axes[i, 1].axhline(-40, color='r', ls=':', lw=0.5)
            axes[i, 1].grid(alpha=0.3)
            axes[i, 2].plot(df['pos1'], df['pos2'])
            axes[i, 2].set_xlabel('q1'); axes[i, 2].set_ylabel('q2'); axes[i, 2].grid(alpha=0.3)
        axes[0, 0].set_title('positions')
        axes[0, 1].set_title('velocities (40 rad/s limit shown)')
        axes[0, 2].set_title('phase portrait q1 vs q2')
        plt.tight_layout()
        fig_path = os.path.join(out_dir, 'overview.png')
        fig.savefig(fig_path, dpi=110)
        print(f'\nOverview plot saved: {fig_path}')

    pd.DataFrame(summary).to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    return summary


if __name__ == '__main__':
    generate_all(out_dir='test_trajectories', plot=True)