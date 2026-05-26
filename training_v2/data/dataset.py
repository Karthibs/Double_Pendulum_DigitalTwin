"""
PyTorch dataset built from a directory of filtered trajectory CSVs.

Splits are made BY TRAJECTORY, not by random sample. This is important for
chaotic systems: nearby samples within one trajectory are highly correlated,
so random per-sample splits leak information from train into val/test and
massively over-estimate generalization.
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class DoublePendulumDataset(Dataset):
    """Concatenated (q, qd, qdd, tau) samples from a list of filtered CSVs."""

    def __init__(self, csv_paths, stride=1, device="cpu"):
        Qs, QDs, QDDs, TAUs, traj_ids = [], [], [], [], []
        for i, path in enumerate(csv_paths):
            df = pd.read_csv(path)
            if stride > 1:
                df = df.iloc[::stride].reset_index(drop=True)
            Qs.append(df[["pos1", "pos2"]].to_numpy())
            QDs.append(df[["vel1", "vel2"]].to_numpy())
            QDDs.append(df[["acc1", "acc2"]].to_numpy())
            TAUs.append(df[["tau1", "tau2"]].to_numpy())
            traj_ids += [i] * len(df)

        self.q   = torch.tensor(np.concatenate(Qs),   dtype=torch.float32, device=device)
        self.qd  = torch.tensor(np.concatenate(QDs),  dtype=torch.float32, device=device)
        self.qdd = torch.tensor(np.concatenate(QDDs), dtype=torch.float32, device=device)
        self.tau = torch.tensor(np.concatenate(TAUs), dtype=torch.float32, device=device)
        self.traj_ids = torch.tensor(traj_ids, dtype=torch.long, device=device)

        self.files = list(csv_paths)
        self.n_traj = len(csv_paths)

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return self.q[idx], self.qd[idx], self.qdd[idx], self.tau[idx]


def split_csv_paths(filtered_dir, pattern="*_data*.csv",
                    val_frac=0.15, test_frac=0.15, seed=0):
    """Trajectory-level random split. Returns (train, val, test) lists of paths."""
    files = sorted(glob.glob(os.path.join(filtered_dir, pattern)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n_test = max(1, int(test_frac * len(files)))
    n_val  = max(1, int(val_frac  * len(files)))
    test_idx  = perm[:n_test]
    val_idx   = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return ([files[i] for i in sorted(train_idx)],
            [files[i] for i in sorted(val_idx)],
            [files[i] for i in sorted(test_idx)])


def save_split(splits, save_path):
    train, val, test = splits
    with open(save_path, "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f, indent=2)


def load_split(path):
    with open(path) as f:
        d = json.load(f)
    return d["train"], d["val"], d["test"]
