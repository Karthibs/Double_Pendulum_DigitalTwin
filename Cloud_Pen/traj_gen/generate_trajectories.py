"""
Generate 100 reference trajectories for PID-tracked data collection on
the double pendulum.

Output (relative to where this script is run):
    trjs/traj01.csv ... traj25.csv   -> both-sine (Category 1)
    trjs/traj26.csv ... traj40.csv   -> swing-up  (Category 2)
    trjs/traj41.csv ... traj50.csv   -> full-circle rotation (Category 3)
    trjs/traj51.csv ... traj60.csv   -> only q1 moves (Category 4)
    trjs/traj61.csv ... traj75.csv   -> only q2 moves (Category 5)
    trjs/initial_conditions.csv      -> 25 starting states (Category 6)

CSV columns match the user's existing format:
    time, pos1, pos2, vel1, vel2, acc1, acc2

Hardware limits respected (with margin):
    Shoulder (q1):  |vel| <= 40 (avg 30) rad/s
    Elbow    (q2):  |vel| <= 50 (avg 40) rad/s
We keep reference |vel| below ~20-25 rad/s so that the PID tracking error
plus inevitable overshoot stays inside the abort threshold.
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Common settings
# ---------------------------------------------------------------------------
TF = 20.0                              # duration of every trajectory [s]
DT = 0.002                             # sampling step [s]
T  = np.arange(0.0, TF + DT, DT)       # time vector (same for all)
N  = len(T)

OUT_DIR = "trjs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_traj(idx, q1, q2, q1d, q2d, q1dd, q2dd):
    """Save (T, q, qdot, qddot) as trjs/traj{idx:02d}.csv."""
    df = pd.DataFrame({
        "time": T,
        "pos1": q1, "pos2": q2,
        "vel1": q1d, "vel2": q2d,
        "acc1": q1dd, "acc2": q2dd,
    })
    path = os.path.join(OUT_DIR, f"traj{idx:02d}.csv")
    df.to_csv(path, index=False)
    return path


def sine_components(A, f, phi=0.0, t=T):
    """Return (q, qdot, qddot) of A*sin(2*pi*f*t + phi)."""
    w = 2.0 * np.pi * f
    q   = A * np.sin(w * t + phi)
    qd  = A * w * np.cos(w * t + phi)
    qdd = -A * w * w * np.sin(w * t + phi)
    return q, qd, qdd


def minjerk(t, t0, tf, q0, qf):
    """Minimum-jerk transition from q0 to qf between t0 and tf.

    Returns (q, qdot, qddot) defined on the same time grid as `t`.
    Position is held at q0 before t0 and at qf after tf;
    velocity and acceleration are zero outside [t0, tf].
    """
    s = np.clip((t - t0) / (tf - t0), 0.0, 1.0)
    h    = 10*s**3 - 15*s**4 + 6*s**5
    hd   = (30*s**2 - 60*s**3 + 30*s**4) / (tf - t0)
    hdd  = (60*s   - 180*s**2 + 120*s**3) / (tf - t0)**2
    q   = q0 + (qf - q0) * h
    qd  = (qf - q0) * hd
    qdd = (qf - q0) * hdd
    in_window = (t >= t0) & (t <= tf)
    qd  = np.where(in_window, qd,  0.0)
    qdd = np.where(in_window, qdd, 0.0)
    return q, qd, qdd


def zero():
    """Convenience: (zeros, zeros, zeros) on the global time grid."""
    z = np.zeros_like(T)
    return z, z.copy(), z.copy()


def smooth_envelope(t_grow, t_hold, t_decay):
    """Trapezoidal envelope on [0, TF]: linearly grows from 0 to 1 over
    [0, t_grow], stays at 1 for [t_grow, t_grow+t_hold], decays to 0 over
    [..., ...+t_decay]. Useful to taper amplitude for swing-up.
    """
    env = np.zeros_like(T)
    g_end = t_grow
    h_end = g_end + t_hold
    d_end = h_end + t_decay
    env = np.where(T <= g_end, T / max(g_end, 1e-9), env)
    env = np.where((T > g_end) & (T <= h_end), 1.0, env)
    env = np.where((T > h_end) & (T <= d_end),
                   1.0 - (T - h_end) / max(t_decay, 1e-9), env)
    env = np.where(T > d_end, 0.0, env)
    return np.clip(env, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Category 1: Both sine (traj01 .. traj25)
# Vary amplitudes, frequencies, and relative phase between q1 and q2.
# All combos kept under |vel| ~ 4 rad/s, well inside hardware envelope.
# ---------------------------------------------------------------------------
BOTH_SINE_SPECS = [
    # ( A1,  A2,   f1,    f2,    phase_offset_q2 )
    (0.3, 0.3, 0.10, 0.15, 0.0),
    (0.5, 0.4, 0.15, 0.20, np.pi/4),
    (0.8, 0.6, 0.10, 0.25, np.pi/3),
    (1.0, 0.8, 0.20, 0.15, np.pi/2),
    (1.5, 1.0, 0.20, 0.20, 0.0),
    (2.0, 1.5, 0.20, 0.15, np.pi/3),       # the user's verified working set
    (2.0, 1.2, 0.15, 0.20, np.pi/6),
    (1.5, 1.5, 0.25, 0.25, 0.0),
    (0.6, 0.4, 0.30, 0.45, np.pi/4),
    (1.2, 1.0, 0.18, 0.22, np.pi),
    (0.7, 0.5, 0.40, 0.30, np.pi/2),
    (2.5, 1.8, 0.10, 0.12, 0.0),
    (2.5, 1.8, 0.10, 0.12, np.pi/2),
    (1.8, 1.5, 0.15, 0.30, np.pi/4),
    (1.0, 0.7, 0.35, 0.20, np.pi/3),
    (0.4, 0.3, 0.50, 0.60, 0.0),
    (2.0, 0.5, 0.15, 0.45, np.pi/3),       # asymmetric: large q1, small q2
    (0.5, 2.0, 0.45, 0.15, np.pi/3),       # asymmetric: small q1, large q2
    (1.6, 1.2, 0.22, 0.18, np.pi/6),
    (0.9, 0.8, 0.28, 0.32, np.pi/2),
    (2.2, 1.6, 0.12, 0.16, np.pi/4),
    (1.4, 1.1, 0.20, 0.40, 0.0),
    (1.0, 1.0, 0.10, 0.30, np.pi),
    (1.8, 1.4, 0.18, 0.22, np.pi/5),
    (2.5, 2.0, 0.08, 0.10, np.pi/3),
]

def make_category1_both_sine():
    paths = []
    for i, (A1, A2, f1, f2, dphi) in enumerate(BOTH_SINE_SPECS, start=1):
        q1, q1d, q1dd = sine_components(A1, f1, phi=0.0)
        q2, q2d, q2dd = sine_components(A2, f2, phi=dphi)
        paths.append(save_traj(i, q1, q2, q1d, q2d, q1dd, q2dd))
    return paths


# ---------------------------------------------------------------------------
# Category 2: Swing-up (traj26 .. traj40)  -- 15 trajectories
# Reference trajectories that *try* to drive q1 up to near vertical (q1=pi).
# The PID will not track these perfectly, but the resulting motion explores
# the high-energy region of state space, which is exactly the data an LNN
# needs to learn the upper half of the swing.
# ---------------------------------------------------------------------------
def make_category2_swing_up():
    paths = []
    idx = 26

    # 26-30: growing-amplitude sinusoid that ends near +pi
    for k, (f, A_final) in enumerate([(0.25, 2.8), (0.30, 2.6), (0.20, 3.0),
                                       (0.35, 2.4), (0.22, 2.9)]):
        w = 2.0 * np.pi * f
        # Use a smooth (C^2) envelope so the acceleration stays finite at t=0:
        #   A(t) = A_final * (t/TF)^2  ->  Ad = 2*A_final*t/TF^2,  Add = 2*A_final/TF^2
        A_t   = A_final * (T / TF) ** 2
        Ad_t  = 2.0 * A_final * T / (TF ** 2)
        Add_t = np.full_like(T, 2.0 * A_final / (TF ** 2))

        q1   = A_t * np.sin(w * T)
        q1d  = Ad_t * np.sin(w*T) + A_t * w * np.cos(w*T)
        q1dd = Add_t * np.sin(w*T) + 2 * Ad_t * w * np.cos(w*T) - A_t * w*w * np.sin(w*T)

        q2 = np.zeros_like(T); q2d = np.zeros_like(T); q2dd = np.zeros_like(T)
        paths.append(save_traj(idx, q1, q2, q1d, q2d, q1dd, q2dd))
        idx += 1

    # 31-35: pump first half (growing sine), smooth crossfade to a min-jerk
    # bring-up to q_target in the second half. The crossfade uses a min-jerk
    # blending weight so position, velocity and acceleration are all C^0.
    for k, (f_pump, t_blend_start, t_blend_end, q_target) in enumerate([
        (0.30, 10.0, 16.0,  np.pi),
        (0.25, 10.0, 17.0,  np.pi),
        (0.35,  8.0, 15.0,  np.pi),
        (0.30, 10.0, 16.0, -np.pi),   # swing the other way
        (0.40, 10.0, 16.0,  np.pi),
    ]):
        w = 2.0 * np.pi * f_pump
        A_max = 1.5
        # Pumping reference: amplitude grows quadratically (C^2) up to A_max at TF
        A_t   = A_max * (T / TF) ** 2
        Ad_t  = 2.0 * A_max * T / (TF ** 2)
        Add_t = np.full_like(T, 2.0 * A_max / (TF ** 2))
        q_pump   = A_t   * np.sin(w * T)
        qd_pump  = Ad_t  * np.sin(w * T) + A_t * w * np.cos(w * T)
        qdd_pump = Add_t * np.sin(w * T) + 2 * Ad_t * w * np.cos(w * T) \
                   - A_t * w * w * np.sin(w * T)

        # Min-jerk pull-up to q_target from current pump pose at t_blend_start
        # Note: this is independent of the pump signal; the blending below ties
        # them together so the transition is smooth.
        q_at_blend = A_max * (t_blend_start / TF) ** 2 * np.sin(w * t_blend_start)
        q_ramp, qd_ramp, qdd_ramp = minjerk(T, t_blend_start, t_blend_end,
                                            q_at_blend, q_target)

        # Min-jerk blending weight from 0 -> 1 across the blend window
        s = np.clip((T - t_blend_start) / max(t_blend_end - t_blend_start, 1e-9), 0, 1)
        w_blend = 10*s**3 - 15*s**4 + 6*s**5
        q1   = (1.0 - w_blend) * q_pump   + w_blend * q_ramp
        q1d  = (1.0 - w_blend) * qd_pump  + w_blend * qd_ramp
        q1dd = (1.0 - w_blend) * qdd_pump + w_blend * qdd_ramp

        # hold at target for the final ~1s
        hold = T >= (t_blend_end)
        q1   = np.where(hold, q_target, q1)
        q1d  = np.where(hold, 0.0,      q1d)
        q1dd = np.where(hold, 0.0,      q1dd)

        q2 = np.zeros_like(T); q2d = np.zeros_like(T); q2dd = np.zeros_like(T)
        paths.append(save_traj(idx, q1, q2, q1d, q2d, q1dd, q2dd))
        idx += 1

    # 36-40: pump both joints (q2 at higher freq), then crossfade to a
    # min-jerk bring-up of both joints to (q1_target, q2_target).
    for k, (f1, f2, q1_target, q2_target) in enumerate([
        (0.25, 0.50, np.pi,  0.0),
        (0.30, 0.55, np.pi,  0.0),
        (0.20, 0.45, np.pi,  np.pi),
        (0.30, 0.60, np.pi, -np.pi),
        (0.25, 0.50, np.pi,  0.5),
    ]):
        w1 = 2.0 * np.pi * f1
        w2 = 2.0 * np.pi * f2
        A1_max, A2_max = 1.3, 0.8
        t_blend_start, t_blend_end = 10.0, 16.0

        # Pumping references, smooth quadratic envelope (C^2 at t=0)
        A1_t   = A1_max * (T / TF) ** 2
        A1d_t  = 2.0 * A1_max * T / (TF ** 2)
        A1dd_t = np.full_like(T, 2.0 * A1_max / (TF ** 2))
        q1_pump   = A1_t   * np.sin(w1 * T)
        q1d_pump  = A1d_t  * np.sin(w1 * T) + A1_t * w1 * np.cos(w1 * T)
        q1dd_pump = A1dd_t * np.sin(w1 * T) + 2 * A1d_t * w1 * np.cos(w1 * T) \
                    - A1_t * w1 * w1 * np.sin(w1 * T)

        A2_t   = A2_max * (T / TF) ** 2
        A2d_t  = 2.0 * A2_max * T / (TF ** 2)
        A2dd_t = np.full_like(T, 2.0 * A2_max / (TF ** 2))
        q2_pump   = A2_t   * np.sin(w2 * T)
        q2d_pump  = A2d_t  * np.sin(w2 * T) + A2_t * w2 * np.cos(w2 * T)
        q2dd_pump = A2dd_t * np.sin(w2 * T) + 2 * A2d_t * w2 * np.cos(w2 * T) \
                    - A2_t * w2 * w2 * np.sin(w2 * T)

        # Min-jerk bring-ups
        q1_at_blend = A1_max * (t_blend_start / TF) ** 2 * np.sin(w1 * t_blend_start)
        q2_at_blend = A2_max * (t_blend_start / TF) ** 2 * np.sin(w2 * t_blend_start)
        q1_ramp, q1d_ramp, q1dd_ramp = minjerk(T, t_blend_start, t_blend_end,
                                                q1_at_blend, q1_target)
        q2_ramp, q2d_ramp, q2dd_ramp = minjerk(T, t_blend_start, t_blend_end,
                                                q2_at_blend, q2_target)

        s = np.clip((T - t_blend_start) / max(t_blend_end - t_blend_start, 1e-9), 0, 1)
        w_blend = 10*s**3 - 15*s**4 + 6*s**5

        q1   = (1 - w_blend) * q1_pump   + w_blend * q1_ramp
        q1d  = (1 - w_blend) * q1d_pump  + w_blend * q1d_ramp
        q1dd = (1 - w_blend) * q1dd_pump + w_blend * q1dd_ramp

        q2   = (1 - w_blend) * q2_pump   + w_blend * q2_ramp
        q2d  = (1 - w_blend) * q2d_pump  + w_blend * q2d_ramp
        q2dd = (1 - w_blend) * q2dd_pump + w_blend * q2dd_ramp

        hold = T >= t_blend_end
        q1   = np.where(hold, q1_target, q1)
        q1d  = np.where(hold, 0.0,       q1d)
        q1dd = np.where(hold, 0.0,       q1dd)
        q2   = np.where(hold, q2_target, q2)
        q2d  = np.where(hold, 0.0,       q2d)
        q2dd = np.where(hold, 0.0,       q2dd)

        paths.append(save_traj(idx, q1, q2, q1d, q2d, q1dd, q2dd))
        idx += 1

    return paths


# ---------------------------------------------------------------------------
# Category 3: Full circle (traj41 .. traj50) -- 10 trajectories
# q1 rotates continuously through 2*pi (one or more times); we smoothly ramp
# from rest to constant angular velocity to keep the start finite.
# ---------------------------------------------------------------------------
def make_category3_full_circle():
    paths = []
    idx = 41
    # (period_s_per_rev, direction, q2_mode)
    specs = [
        (8.0,   +1, "zero"),         # slow, CCW, q2 still
        (8.0,   -1, "zero"),         # slow, CW
        (5.0,   +1, "zero"),
        (5.0,   -1, "zero"),
        (4.0,   +1, "sin_small"),    # q2 oscillates gently
        (4.0,   -1, "sin_small"),
        (3.5,   +1, "sin_medium"),
        (6.0,   +1, "sin_medium"),
        (10.0,  +1, "two_revs"),     # 2 revolutions over the trajectory
        (5.0,   +1, "counter"),      # q2 swings opposite to q1
    ]
    for period, direction, q2_mode in specs:
        omega = direction * 2.0 * np.pi / period          # rad/s (target)
        # |omega| <= 2*pi/3.5 ~= 1.8 rad/s, safely under hardware limit.
        # Smooth ramp-up in the first 2 seconds, then constant omega.
        t_ramp = 2.0
        q1   = np.zeros_like(T)
        q1d  = np.zeros_like(T)
        q1dd = np.zeros_like(T)
        # ramp portion
        rmask = T <= t_ramp
        s = T[rmask] / t_ramp
        # min-jerk in velocity (so q is the time-integral of min-jerk v profile)
        # v(t) = omega * (10 s^3 - 15 s^4 + 6 s^5)  with s = t/t_ramp
        # q(t) = integral; close-form is messy, so integrate numerically
        v_ramp = omega * (10*s**3 - 15*s**4 + 6*s**5)
        a_ramp = omega * (30*s**2 - 60*s**3 + 30*s**4) / t_ramp
        q_ramp = np.zeros_like(T[rmask])
        q_ramp[1:] = np.cumsum((v_ramp[:-1] + v_ramp[1:]) * 0.5 * DT)
        q1[rmask]   = q_ramp
        q1d[rmask]  = v_ramp
        q1dd[rmask] = a_ramp
        # constant-omega portion: starts where ramp ended
        cmask = T > t_ramp
        q_end_ramp = q_ramp[-1]
        q1[cmask]   = q_end_ramp + omega * (T[cmask] - t_ramp)
        q1d[cmask]  = omega
        q1dd[cmask] = 0.0

        if q2_mode == "zero":
            q2 = np.zeros_like(T); q2d = np.zeros_like(T); q2dd = np.zeros_like(T)
        elif q2_mode == "sin_small":
            q2, q2d, q2dd = sine_components(0.3, 0.4)
        elif q2_mode == "sin_medium":
            q2, q2d, q2dd = sine_components(0.6, 0.3)
        elif q2_mode == "counter":
            # q2 oscillates at same frequency as q1 rotation
            f = 1.0 / period
            q2, q2d, q2dd = sine_components(0.5, f, phi=np.pi)
        elif q2_mode == "two_revs":
            q2, q2d, q2dd = sine_components(0.4, 0.2)
        else:
            q2 = np.zeros_like(T); q2d = np.zeros_like(T); q2dd = np.zeros_like(T)

        paths.append(save_traj(idx, q1, q2, q1d, q2d, q1dd, q2dd))
        idx += 1

    return paths


# ---------------------------------------------------------------------------
# Category 4: Only q1 moves (traj51 .. traj60) -- 10 trajectories
# q2 = 0 throughout; q1 sweeps various profiles.
# ---------------------------------------------------------------------------
def make_category4_q1_only():
    paths = []
    idx = 51
    profiles = [
        ("sine",      {"A": 0.8, "f": 0.20}),
        ("sine",      {"A": 1.5, "f": 0.15}),
        ("sine",      {"A": 2.0, "f": 0.10}),
        ("multisine", {"As": (0.5, 0.3, 0.2), "fs": (0.15, 0.35, 0.55)}),
        ("multisine", {"As": (0.8, 0.4, 0.2), "fs": (0.10, 0.25, 0.45)}),
        ("chirp",     {"A": 1.0, "f0": 0.10, "f1": 0.60}),
        ("chirp",     {"A": 0.6, "f0": 0.60, "f1": 0.10}),
        ("ramp_sine", {"A": 1.2, "f": 0.25}),   # growing-amplitude sinusoid
        ("steps",     {"levels": (0.0, 0.5, -0.4, 0.7, -0.2, 0.0)}),
        ("sine",      {"A": 2.5, "f": 0.08}),
    ]
    q2 = np.zeros_like(T); q2d = np.zeros_like(T); q2dd = np.zeros_like(T)

    for kind, p in profiles:
        if kind == "sine":
            q1, q1d, q1dd = sine_components(p["A"], p["f"])
        elif kind == "multisine":
            q1 = np.zeros_like(T); q1d = q1.copy(); q1dd = q1.copy()
            for A, f in zip(p["As"], p["fs"]):
                a, b, c = sine_components(A, f)
                q1 += a; q1d += b; q1dd += c
        elif kind == "chirp":
            # phase = 2*pi*(f0*t + 0.5*(f1-f0)*t^2/TF)
            A, f0, f1 = p["A"], p["f0"], p["f1"]
            f_inst = f0 + (f1 - f0) * T / TF
            phase  = 2 * np.pi * (f0 * T + 0.5 * (f1 - f0) * T**2 / TF)
            wi   = 2 * np.pi * f_inst
            widot = 2 * np.pi * (f1 - f0) / TF
            q1   = A * np.sin(phase)
            q1d  = A * wi * np.cos(phase)
            q1dd = -A * wi*wi * np.sin(phase) + A * widot * np.cos(phase)
        elif kind == "ramp_sine":
            A, f = p["A"], p["f"]
            w = 2 * np.pi * f
            env = T / TF
            q1   = A * env * np.sin(w*T)
            q1d  = (A / TF) * np.sin(w*T) + A * env * w * np.cos(w*T)
            q1dd = 2 * (A/TF) * w * np.cos(w*T) - A * env * w*w * np.sin(w*T)
        elif kind == "steps":
            levels = p["levels"]
            n_seg = len(levels) - 1
            seg_len = TF / n_seg
            q1 = np.zeros_like(T); q1d = q1.copy(); q1dd = q1.copy()
            for i in range(n_seg):
                t0 = i * seg_len
                tf_seg = t0 + seg_len * 0.7   # transition in 70% of segment
                a, b, c = minjerk(T, t0, tf_seg, levels[i], levels[i+1])
                inseg = (T >= t0) & (T < (i+1)*seg_len)
                q1[inseg]   = a[inseg]
                q1d[inseg]  = b[inseg]
                q1dd[inseg] = c[inseg]
            # tail
            q1[T >= TF] = levels[-1]
        else:
            raise ValueError(kind)
        paths.append(save_traj(idx, q1, q2, q1d, q2d, q1dd, q2dd))
        idx += 1
    return paths


# ---------------------------------------------------------------------------
# Category 5: Only q2 moves (traj61 .. traj75) -- 15 trajectories
# q1 = 0 throughout; q2 sweeps various profiles.
# ---------------------------------------------------------------------------
def make_category5_q2_only():
    paths = []
    idx = 61
    profiles = [
        ("sine",      {"A": 0.5, "f": 0.30}),
        ("sine",      {"A": 1.0, "f": 0.20}),
        ("sine",      {"A": 1.5, "f": 0.15}),
        ("sine",      {"A": 2.0, "f": 0.10}),
        ("sine",      {"A": 2.5, "f": 0.08}),
        ("multisine", {"As": (0.5, 0.3, 0.2), "fs": (0.15, 0.35, 0.55)}),
        ("multisine", {"As": (0.6, 0.4, 0.3), "fs": (0.20, 0.40, 0.65)}),
        ("multisine", {"As": (1.0, 0.5, 0.2), "fs": (0.10, 0.25, 0.45)}),
        ("chirp",     {"A": 1.0, "f0": 0.10, "f1": 0.70}),
        ("chirp",     {"A": 0.8, "f0": 0.70, "f1": 0.10}),
        ("chirp",     {"A": 1.5, "f0": 0.10, "f1": 0.30}),
        ("ramp_sine", {"A": 1.5, "f": 0.20}),
        ("ramp_sine", {"A": 2.0, "f": 0.15}),
        ("steps",     {"levels": (0.0, 0.6, -0.5, 0.8, -0.3, 0.5, 0.0)}),
        ("steps",     {"levels": (0.0, 1.0, -1.0, 0.5, -0.5, 0.0)}),
    ]
    q1 = np.zeros_like(T); q1d = q1.copy(); q1dd = q1.copy()
    for kind, p in profiles:
        if kind == "sine":
            q2, q2d, q2dd = sine_components(p["A"], p["f"])
        elif kind == "multisine":
            q2 = np.zeros_like(T); q2d = q2.copy(); q2dd = q2.copy()
            for A, f in zip(p["As"], p["fs"]):
                a, b, c = sine_components(A, f)
                q2 += a; q2d += b; q2dd += c
        elif kind == "chirp":
            A, f0, f1 = p["A"], p["f0"], p["f1"]
            f_inst = f0 + (f1 - f0) * T / TF
            phase  = 2 * np.pi * (f0 * T + 0.5 * (f1 - f0) * T**2 / TF)
            wi    = 2 * np.pi * f_inst
            widot = 2 * np.pi * (f1 - f0) / TF
            q2   = A * np.sin(phase)
            q2d  = A * wi * np.cos(phase)
            q2dd = -A * wi*wi * np.sin(phase) + A * widot * np.cos(phase)
        elif kind == "ramp_sine":
            A, f = p["A"], p["f"]
            w = 2 * np.pi * f
            env = T / TF
            q2   = A * env * np.sin(w*T)
            q2d  = (A / TF) * np.sin(w*T) + A * env * w * np.cos(w*T)
            q2dd = 2 * (A/TF) * w * np.cos(w*T) - A * env * w*w * np.sin(w*T)
        elif kind == "steps":
            levels = p["levels"]
            n_seg = len(levels) - 1
            seg_len = TF / n_seg
            q2 = np.zeros_like(T); q2d = q2.copy(); q2dd = q2.copy()
            for i in range(n_seg):
                t0 = i * seg_len
                tf_seg = t0 + seg_len * 0.7
                a, b, c = minjerk(T, t0, tf_seg, levels[i], levels[i+1])
                inseg = (T >= t0) & (T < (i+1)*seg_len)
                q2[inseg]   = a[inseg]
                q2d[inseg]  = b[inseg]
                q2dd[inseg] = c[inseg]
            q2[T >= TF] = levels[-1]
        else:
            raise ValueError(kind)
        paths.append(save_traj(idx, q1, q2, q1d, q2d, q1dd, q2dd))
        idx += 1
    return paths


# ---------------------------------------------------------------------------
# Category 6: Initial conditions only (last 25)
# A diverse 5x5 grid of starting (q1, q2) positions with zero initial
# velocity. Saved as initial_conditions.csv with columns q1, q2, q1_dot, q2_dot
# so you can load it with pd.read_csv or np.loadtxt.
# ---------------------------------------------------------------------------
def make_category6_initial_conditions():
    q1_vals = np.linspace(-np.pi, np.pi, 5)         # 5 angles in [-pi, pi]
    q2_vals = np.linspace(-np.pi, np.pi, 5)         # 5 angles in [-pi, pi]
    rows = []
    for q1 in q1_vals:
        for q2 in q2_vals:
            rows.append([float(q1), float(q2), 0.0, 0.0])
    df = pd.DataFrame(rows, columns=["q1", "q2", "q1_dot", "q2_dot"])
    path = os.path.join(OUT_DIR, "initial_conditions.csv")
    df.to_csv(path, index=False)
    return path, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Category 1: both-sine     (traj01-traj25)")
    p1 = make_category1_both_sine()
    print(f"   wrote {len(p1)} files")

    print("Category 2: swing-up      (traj26-traj40)")
    p2 = make_category2_swing_up()
    print(f"   wrote {len(p2)} files")

    print("Category 3: full circle   (traj41-traj50)")
    p3 = make_category3_full_circle()
    print(f"   wrote {len(p3)} files")

    print("Category 4: only q1       (traj51-traj60)")
    p4 = make_category4_q1_only()
    print(f"   wrote {len(p4)} files")

    print("Category 5: only q2       (traj61-traj75)")
    p5 = make_category5_q2_only()
    print(f"   wrote {len(p5)} files")

    print("Category 6: initial conds (25 rows in initial_conditions.csv)")
    ic_path, ic_df = make_category6_initial_conditions()
    print(f"   wrote {ic_path} ({len(ic_df)} rows)")

    return p1 + p2 + p3 + p4 + p5, ic_path


if __name__ == "__main__":
    paths, ic_path = generate_all()
    print(f"\nTotal trajectory files: {len(paths)}")
    print(f"Initial conditions:     {ic_path}")
