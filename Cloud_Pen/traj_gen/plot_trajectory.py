"""
Plot function for the trajectory CSV files produced by
`generate_trajectories.py`.

Usage
-----
    from plot_trajectory import plot_trajectory
    plot_trajectory("trjs/traj01.csv")
    plot_trajectory("trjs/traj32.csv", save_to="traj32.png")

Or from the command line:
    python plot_trajectory.py trjs/traj01.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_trajectory(csv_path, save_to=None, show=True, title=None):
    """Plot a single trajectory CSV.

    Expected columns: time, pos1, pos2, vel1, vel2, acc1, acc2.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    save_to : str or None
        If given, save the figure to this path (PNG/PDF/SVG inferred).
    show : bool
        If True, call plt.show().
    title : str or None
        Override the figure title. Default: file name.
    """
    df = pd.read_csv(csv_path)
    t   = df["time"].to_numpy()
    q1  = df["pos1"].to_numpy(); q2  = df["pos2"].to_numpy()
    q1d = df["vel1"].to_numpy(); q2d = df["vel2"].to_numpy()
    q1dd = df["acc1"].to_numpy(); q2dd = df["acc2"].to_numpy()

    if title is None:
        title = os.path.basename(csv_path)

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=14)

    # Row 0: positions
    axes[0, 0].plot(t, q1, color="C0")
    axes[0, 0].set_ylabel("q1 [rad]")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline( np.pi, color="gray", ls=":", lw=0.7)
    axes[0, 0].axhline(-np.pi, color="gray", ls=":", lw=0.7)

    axes[0, 1].plot(t, q2, color="C1")
    axes[0, 1].set_ylabel("q2 [rad]")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline( np.pi, color="gray", ls=":", lw=0.7)
    axes[0, 1].axhline(-np.pi, color="gray", ls=":", lw=0.7)

    # Row 1: velocities, with hardware limits as dashed red lines
    axes[1, 0].plot(t, q1d, color="C0")
    axes[1, 0].set_ylabel(r"$\dot q_1$ [rad/s]")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline( 40, color="red", ls="--", lw=0.7, label="hw limit")
    axes[1, 0].axhline(-40, color="red", ls="--", lw=0.7)
    axes[1, 0].axhline( 30, color="orange", ls=":", lw=0.7, label="hw avg")
    axes[1, 0].axhline(-30, color="orange", ls=":", lw=0.7)
    axes[1, 0].legend(fontsize=8, loc="best")

    axes[1, 1].plot(t, q2d, color="C1")
    axes[1, 1].set_ylabel(r"$\dot q_2$ [rad/s]")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline( 50, color="red", ls="--", lw=0.7, label="hw limit")
    axes[1, 1].axhline(-50, color="red", ls="--", lw=0.7)
    axes[1, 1].axhline( 40, color="orange", ls=":", lw=0.7, label="hw avg")
    axes[1, 1].axhline(-40, color="orange", ls=":", lw=0.7)
    axes[1, 1].legend(fontsize=8, loc="best")

    # Row 2: accelerations
    axes[2, 0].plot(t, q1dd, color="C0")
    axes[2, 0].set_ylabel(r"$\ddot q_1$ [rad/s$^2$]")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(t, q2dd, color="C1")
    axes[2, 1].set_ylabel(r"$\ddot q_2$ [rad/s$^2$]")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].grid(True, alpha=0.3)

    # Annotate max values + limit-violation warning
    summary = (
        f"max |q1|   = {np.max(np.abs(q1)):.2f} rad,   "
        f"max |q2|   = {np.max(np.abs(q2)):.2f} rad\n"
        f"max |q1d|  = {np.max(np.abs(q1d)):.2f} rad/s  (hw 40),   "
        f"max |q2d|  = {np.max(np.abs(q2d)):.2f} rad/s  (hw 50)\n"
        f"max |q1dd| = {np.max(np.abs(q1dd)):.1f} rad/s^2,  "
        f"max |q2dd| = {np.max(np.abs(q2dd)):.1f} rad/s^2"
    )
    fig.text(0.5, -0.02, summary, ha="center", fontsize=9,
             family="monospace",
             bbox=dict(facecolor="lightyellow", edgecolor="gray", pad=4))

    # Velocity safety check
    over_q1 = np.max(np.abs(q1d)) > 40.0
    over_q2 = np.max(np.abs(q2d)) > 50.0
    if over_q1 or over_q2:
        fig.text(0.5, 1.00, "!! REFERENCE EXCEEDS HARDWARE VELOCITY LIMIT !!",
                 ha="center", color="red", fontsize=12, weight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_to is not None:
        plt.savefig(save_to, bbox_inches="tight", dpi=110)
    if show:
        plt.show()
    plt.close(fig)


def plot_phase_space(csv_path, save_to=None, show=True):
    """Auxiliary: q1 vs q2 and q1d vs q2d for one trajectory."""
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["pos1"], df["pos2"], lw=0.8)
    axes[0].set_xlabel("q1 [rad]"); axes[0].set_ylabel("q2 [rad]")
    axes[0].set_title("config space"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["vel1"], df["vel2"], lw=0.8, color="C2")
    axes[1].set_xlabel(r"$\dot q_1$"); axes[1].set_ylabel(r"$\dot q_2$")
    axes[1].set_title("velocity space"); axes[1].grid(True, alpha=0.3)
    fig.suptitle(os.path.basename(csv_path))
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, bbox_inches="tight", dpi=110)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_trajectory.py <path_to_csv> [save_path]")
        sys.exit(1)
    csv = sys.argv[1]
    save = sys.argv[2] if len(sys.argv) > 2 else None
    plot_trajectory(csv, save_to=save, show=(save is None))
