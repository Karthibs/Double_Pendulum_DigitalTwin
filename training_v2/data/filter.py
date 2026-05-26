"""
Filtering raw measured CSVs into clean (q, qd, qdd, tau) training tuples.

Output CSVs have:
time, pos1, pos2, vel1, vel2, acc1, acc2, tau1, tau2
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter


# ---------------------------------------------------------------------------
# filtering methods
# ---------------------------------------------------------------------------
def butterworth(x, order=3, Wn=0.02):
    """Zero-phase Butterworth low-pass via filtfilt."""
    b, a = butter(order, Wn)
    return filtfilt(b, a, x)


def savgol_smooth_and_deriv(x, t, window_pos=51, window_vel=51, polyorder=3):
    """Savitzky-Golay smoothing + analytical derivatives."""
    dt = float(t[1] - t[0])
    q  = savgol_filter(x, window_pos, polyorder, mode="interp")
    qd = savgol_filter(x, window_vel, polyorder, deriv=1, delta=dt, mode="interp")
    qdd = savgol_filter(x, window_vel, polyorder, deriv=2, delta=dt, mode="interp")
    return q, qd, qdd


def filter_one_csv(csv_path,
                   method="butterworth",
                   Wn_pos=0.05, Wn_vel=0.02, Wn_acc=0.02, Wn_tau=0.05,
                   order=3,
                   savgol_window=51, savgol_poly=3,
                   edge_trim=100):
    df = pd.read_csv(csv_path)

    # Accept either schema (raw measured or already saved)
    pos_cols = ("pos_meas1", "pos_meas2") if "pos_meas1" in df else ("pos1", "pos2")
    vel_cols = ("vel_meas1", "vel_meas2") if "vel_meas1" in df else ("vel1", "vel2")
    tau_cols = ("tau_meas1", "tau_meas2") if "tau_meas1" in df else ("tau1", "tau2")

    t   = df["time"].to_numpy()
    q1, q2 = df[pos_cols[0]].to_numpy(), df[pos_cols[1]].to_numpy()
    v1, v2 = df[vel_cols[0]].to_numpy(), df[vel_cols[1]].to_numpy()
    u1, u2 = df[tau_cols[0]].to_numpy(), df[tau_cols[1]].to_numpy()

    if method == "savgol":
        q1f, qd1, qdd1 = savgol_smooth_and_deriv(q1, t, savgol_window, savgol_window, savgol_poly)
        q2f, qd2, qdd2 = savgol_smooth_and_deriv(q2, t, savgol_window, savgol_window, savgol_poly)
        u1f = savgol_filter(u1, savgol_window, savgol_poly, mode="interp")
        u2f = savgol_filter(u2, savgol_window, savgol_poly, mode="interp")
    elif method == "butterworth":
        q1f = butterworth(q1, order, Wn_pos)
        q2f = butterworth(q2, order, Wn_pos)
        qd1 = butterworth(v1, order, Wn_vel)
        qd2 = butterworth(v2, order, Wn_vel)
        u1f = butterworth(u1, order, Wn_tau)
        u2f = butterworth(u2, order, Wn_tau)
        # acceleration from filtered velocity, then re-smooth
        qdd1 = butterworth(np.gradient(qd1, t), order, Wn_acc)
        qdd2 = butterworth(np.gradient(qd2, t), order, Wn_acc)
    else:
        raise ValueError(f"Unknown filtering method: {method}")

    out = pd.DataFrame({
        "time": t,
        "pos1": q1f, "pos2": q2f,
        "vel1": qd1, "vel2": qd2,
        "acc1": qdd1, "acc2": qdd2,
        "tau1": u1f, "tau2": u2f,
    })

    if edge_trim > 0 and len(out) > 2 * edge_trim:
        out = out.iloc[edge_trim:-edge_trim].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# directory processing
# ---------------------------------------------------------------------------
def filter_directory(raw_dir, out_dir, pattern="*_data*.csv", **filter_kwargs):
    """Filter every CSV matching `pattern` in `raw_dir`, write to `out_dir`.

    Returns the list of output file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    out_paths = []
    for f in files:
        name = os.path.basename(f)
        out_path = os.path.join(out_dir, name)
        df = filter_one_csv(f, **filter_kwargs)
        df.to_csv(out_path, index=False)
        out_paths.append(out_path)
        print(f"  {name:<35s}  {len(df):>6d} rows  -> {out_path}")
    print(f"\nFiltered {len(files)} files into {out_dir}")
    return out_paths
