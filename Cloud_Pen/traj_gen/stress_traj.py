"""
STRESS test trajectories: designed to push into state-space regions
poorly covered by typical PID training data.

Unlike `generate_test_trajectories.py` (which used different waveform
shapes but stayed within the training velocity range and amplitude
range), this set deliberately pushes:

  - Higher velocities  (peak 6-15 rad/s)
  - Larger amplitudes  (|q| up to pi)
  - Higher accelerations (peak 200+ rad/s^2)
  - More extreme configurations (near-vertical, inverted, large-angle bent)

Each trajectory is still hardware-safe under your limits:
  shoulder: |q_dot| <= 40 rad/s, torque <= 0.15 Nm
  elbow:    |q_dot| <= 50 rad/s, torque <= 0.15 Nm

But they go MUCH further into extrapolation than the prior test set.
A well-trained DeLaN should still hold; a poorly-trained one will
visibly diverge.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


T_FINAL = 20.0
DT = 0.005


def differentiate(x, dt):
    xd = np.zeros_like(x)
    xd[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    xd[0]    = (x[1] - x[0]) / dt
    xd[-1]   = (x[-1] - x[-2]) / dt
    return xd


def build_traj(t, q1, q2):
    dt = t[1] - t[0]
    qd1 = differentiate(q1, dt); qd2 = differentiate(q2, dt)
    qdd1 = differentiate(qd1, dt); qdd2 = differentiate(qd2, dt)
    return pd.DataFrame({
        'time': t,
        'pos1': q1, 'pos2': q2,
        'vel1': qd1, 'vel2': qd2,
        'acc1': qdd1, 'acc2': qdd2,
    })


def smooth_step(t, t0, t1):
    """Smooth Hermite step from 0 at t0 to 1 at t1."""
    u = np.clip((t - t0) / (t1 - t0 + 1e-9), 0, 1)
    return u * u * (3 - 2 * u)


# ===========================================================================
# Stress trajectories
# ===========================================================================

def S01_largeamp_slow(t):
    """LARGE amplitude motion at moderate speed.
    Reaches near-vertical positions. Peak |q| ~ pi/2 ~ 1.57.
    """
    f = 0.2
    q1 = 1.4 * np.sin(2*np.pi*f*t)              # +/- 80 deg
    q2 = 1.4 * np.sin(2*np.pi*f*t + np.pi/3)
    return q1, q2


def S02_largeamp_fast(t):
    """LARGE amplitude at moderately high speed - peak velocity ~6 rad/s."""
    f = 0.6
    q1 = 1.2 * np.sin(2*np.pi*f*t)
    q2 = 1.0 * np.sin(2*np.pi*f*t + np.pi/4)
    return q1, q2


def S03_chirp(t):
    """Frequency sweep 0.2 Hz -> 1.5 Hz. Visits a wide velocity range
    in one trajectory. Tests dynamics-at-many-speeds in one go.
    """
    f0, f1 = 0.2, 1.5
    # log-spaced freq for slow start -> fast end
    f = f0 * (f1/f0) ** (t / T_FINAL)
    phi = 2 * np.pi * f0 * T_FINAL / np.log(f1/f0) * ((f1/f0) ** (t/T_FINAL) - 1)
    q1 = 1.0 * np.sin(phi)
    q2 = 0.8 * np.sin(phi + np.pi/3)
    return q1, q2


def S04_aggressive_pump(t):
    """Pump up to near-vertical from rest. Tests transient dynamics
    far from the initial state.
    """
    # Increasing-amplitude sine until t=10s, then hold maximum amplitude
    ramp = np.minimum(t / 10.0, 1.0)
    q1 = 1.5 * ramp * np.sin(2*np.pi*0.4*t)
    q2 = 1.0 * ramp * np.sin(2*np.pi*0.4*t + np.pi/2)
    return q1, q2


def S05_step_changes(t):
    """Series of large step-like motions between bent and inverted configs.
    Smooth Hermite transitions between targets.
    """
    # Targets to step through
    targets_q1 = [0.0,  1.2,  1.2,  -1.2,  -1.2, 0.6,  0.6, -0.8,  -0.8, 0.0]
    targets_q2 = [0.0, -0.5,  0.8,   0.8,   -0.4, 0.0, -1.0,  1.0,   0.0, 0.0]
    n_steps = len(targets_q1)
    seg_t = T_FINAL / (n_steps - 1)

    q1 = np.zeros_like(t); q2 = np.zeros_like(t)
    for k in range(n_steps - 1):
        t0 = k * seg_t; t1 = (k + 1) * seg_t
        s = smooth_step(t, t0, t1)
        q1 += (targets_q1[k+1] - targets_q1[k]) * smooth_step(t, t0, t1) * (t < t1)
        q2 += (targets_q2[k+1] - targets_q2[k]) * smooth_step(t, t0, t1) * (t < t1)
        # Reset effect for previous step using indicator
    # Simpler implementation: lookup
    q1 = np.zeros_like(t); q2 = np.zeros_like(t)
    for k in range(n_steps - 1):
        t0 = k * seg_t; t1 = (k + 1) * seg_t
        mask = (t >= t0) & (t <= t1)
        s = smooth_step(t[mask], t0, t1)
        q1[mask] = targets_q1[k] + s * (targets_q1[k+1] - targets_q1[k])
        q2[mask] = targets_q2[k] + s * (targets_q2[k+1] - targets_q2[k])
    # Beyond the last segment, hold the final value
    last_mask = t > (n_steps - 1) * seg_t
    q1[last_mask] = targets_q1[-1]; q2[last_mask] = targets_q2[-1]
    return q1, q2


def S06_inverted_oscillation(t):
    """Hang out near INVERTED equilibrium, with small oscillations around it.
    Tests model in the unstable region of state space.
    Note: PID may struggle to track this; recorded trajectory will deviate.
    """
    # Smoothly ramp to near inverted, then small oscillation
    ramp = smooth_step(t, 0.0, 5.0)
    q1 = (np.pi - 0.4) * ramp + 0.3 * smooth_step(t, 5.0, 6.0) * np.sin(2*np.pi*0.5*(t-5))
    q2 = -0.6 * ramp + 0.2 * smooth_step(t, 5.0, 6.0) * np.sin(2*np.pi*0.5*(t-5) + np.pi/2)
    return q1, q2


def S07_high_accel_burst(t):
    """Mostly slow, but with brief HIGH-ACCELERATION bursts.
    Tests the inertial term M(q)*qdd specifically.
    """
    # Background slow motion
    q1_bg = 0.5 * np.sin(2*np.pi*0.2*t)
    q2_bg = 0.3 * np.sin(2*np.pi*0.2*t + np.pi/4)
    # Short high-amp / high-accel bursts
    burst1 = 0.4 * np.exp(-((t - 5.0)/0.5)**2) * np.sin(2*np.pi*2.5*t)
    burst2 = 0.3 * np.exp(-((t - 10.0)/0.5)**2) * np.sin(2*np.pi*3.0*t)
    burst3 = 0.4 * np.exp(-((t - 15.0)/0.5)**2) * np.sin(2*np.pi*2.5*t)
    q1 = q1_bg + burst1 + burst3
    q2 = q2_bg + burst2
    return q1, q2


def S08_coriolis_excite(t):
    """Both joints move fast simultaneously - maximizes Coriolis term.
    Coriolis force is proportional to product of joint velocities,
    so this is where the c(q, qd) term in the dynamics is largest.
    """
    f1, f2 = 0.7, 0.55
    q1 = 1.0 * np.sin(2*np.pi*f1*t)
    q2 = 1.0 * np.cos(2*np.pi*f2*t)
    return q1, q2


def S09_random_extrapolation(t):
    """Smooth random reference with LARGER amplitude than your training likely had.
    Pure generalization test - no periodic structure.
    """
    rng = np.random.default_rng(12345)
    n = len(t); dt = t[1] - t[0]
    raw = rng.standard_normal(n)
    tau = 1.0 / (2 * np.pi * 0.3)
    alpha = dt / (tau + dt)
    out = np.zeros_like(raw); out[0] = raw[0]
    for i in range(1, n):
        out[i] = alpha * raw[i] + (1 - alpha) * out[i-1]
    out2 = np.zeros_like(out); out2[0] = out[0]
    for i in range(1, n):
        out2[i] = alpha * out[i] + (1 - alpha) * out2[i-1]
    out2 /= (np.max(np.abs(out2)) + 1e-9)

    rng2 = np.random.default_rng(67890)
    raw2 = rng2.standard_normal(n)
    out3 = np.zeros_like(raw2); out3[0] = raw2[0]
    for i in range(1, n):
        out3[i] = alpha * raw2[i] + (1 - alpha) * out3[i-1]
    out4 = np.zeros_like(out3); out4[0] = out3[0]
    for i in range(1, n):
        out4[i] = alpha * out3[i] + (1 - alpha) * out4[i-1]
    out4 /= (np.max(np.abs(out4)) + 1e-9)

    q1 = 1.2 * out2
    q2 = 1.0 * out4
    return q1, q2


def S10_extreme_swing(t):
    """Continuous large-angle swing through near-inverted from below.
    The most aggressive test: goes through positions and velocities
    you almost certainly didn't have in training.
    """
    f = 0.35
    q1 = 1.8 * np.sin(2*np.pi*f*t)        # +/- 103 deg, more than pi/2
    q2 = 0.8 * np.sin(2*np.pi*f*t + np.pi/2)
    return q1, q2


STRESS_TRAJECTORIES = {
    'S01':       S01_largeamp_slow,
    'S02':       S02_largeamp_fast,
    'S03':               S03_chirp,
    'S04':     S04_aggressive_pump,
    'S05':        S05_step_changes,
    'S06': S06_inverted_oscillation,
    'S07':    S07_high_accel_burst,
    'S08':     S08_coriolis_excite,
    'S09': S09_random_extrapolation,
    'S10':       S10_extreme_swing,
}


# ---------------------------------------------------------------------------
# Generation + safety check
# ---------------------------------------------------------------------------
SHOULDER_VEL_LIMIT = 40.0
ELBOW_VEL_LIMIT    = 50.0
SAFETY_MARGIN      = 0.7    # stay below 70% of limit


def generate_all(out_dir='stress_trajectories', plot=True):
    os.makedirs(out_dir, exist_ok=True)
    t = np.arange(0, T_FINAL + DT, DT)

    summary = []
    print(f"{'name':<28s}{'peak_q1':>10s}{'peak_q2':>10s}{'peak_v1':>10s}{'peak_v2':>10s}"
          f"{'peak_a1':>10s}{'peak_a2':>10s}  safety")
    print('-' * 100)
    for name, fn in STRESS_TRAJECTORIES.items():
        q1, q2 = fn(t)
        df = build_traj(t, q1, q2)

        pq1 = float(np.max(np.abs(q1)));      pq2 = float(np.max(np.abs(q2)))
        pv1 = float(np.max(np.abs(df['vel1']))); pv2 = float(np.max(np.abs(df['vel2'])))
        pa1 = float(np.max(np.abs(df['acc1']))); pa2 = float(np.max(np.abs(df['acc2'])))

        safe_v1 = pv1 < SHOULDER_VEL_LIMIT * SAFETY_MARGIN
        safe_v2 = pv2 < ELBOW_VEL_LIMIT * SAFETY_MARGIN
        safe = safe_v1 and safe_v2

        csv_path = os.path.join(out_dir, f'{name}.csv')
        df.to_csv(csv_path, index=False)

        safety = "OK" if safe else "*** VELOCITY EXCEEDS SAFETY MARGIN ***"
        print(f'  {name:<26s}{pq1:>10.3f}{pq2:>10.3f}{pv1:>10.2f}{pv2:>10.2f}'
              f'{pa1:>10.1f}{pa2:>10.1f}  {safety}')

        summary.append({'name': name, 'csv': csv_path,
                       'peak_q1': pq1, 'peak_q2': pq2,
                       'peak_v1': pv1, 'peak_v2': pv2,
                       'peak_a1': pa1, 'peak_a2': pa2,
                       'safe': safe})

    print()
    print(f"Safety margins: |dq1| <= {SHOULDER_VEL_LIMIT * SAFETY_MARGIN:.0f}, "
          f"|dq2| <= {ELBOW_VEL_LIMIT * SAFETY_MARGIN:.0f}")

    if plot:
        n = len(STRESS_TRAJECTORIES)
        fig, axes = plt.subplots(n, 3, figsize=(15, 2.2*n), sharex=True)
        for i, (name, fn) in enumerate(STRESS_TRAJECTORIES.items()):
            q1, q2 = fn(t)
            df = build_traj(t, q1, q2)
            axes[i, 0].plot(t, q1, label='q1'); axes[i, 0].plot(t, q2, label='q2')
            axes[i, 0].set_ylabel(name[:14]); axes[i, 0].grid(alpha=0.3); axes[i, 0].legend(fontsize=7)
            axes[i, 1].plot(t, df['vel1']); axes[i, 1].plot(t, df['vel2'])
            axes[i, 1].axhline( 40, color='r', ls=':', lw=0.5)
            axes[i, 1].axhline(-40, color='r', ls=':', lw=0.5)
            axes[i, 1].grid(alpha=0.3)
            axes[i, 2].plot(df['pos1'], df['pos2'])
            axes[i, 2].set_xlabel('q1'); axes[i, 2].set_ylabel('q2'); axes[i, 2].grid(alpha=0.3)
        axes[0, 0].set_title('positions')
        axes[0, 1].set_title('velocities (40 rad/s limit shown)')
        axes[0, 2].set_title('phase portrait q1 vs q2')
        plt.tight_layout()
        fig_path = os.path.join(out_dir, 'overview.png')
        fig.savefig(fig_path, dpi=110)
        print(f'\nOverview plot: {fig_path}')

    pd.DataFrame(summary).to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    return summary


if __name__ == '__main__':
    generate_all(out_dir='stress_trajectories', plot=True)