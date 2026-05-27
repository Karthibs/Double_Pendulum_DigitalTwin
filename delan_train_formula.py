"""
delan_train_formula.py
"""

import argparse
import time
import numpy as np
import torch

from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork
from deep_lagrangian_networks.replay_memory import PyTorchReplayMemory
from deep_lagrangian_networks.utils import load_dataset, init_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, default=[True])
    parser.add_argument("-i", nargs=1, type=int, default=[0])
    parser.add_argument("-s", nargs=1, type=int, default=[42])
    parser.add_argument("-r", nargs=1, type=int, default=[1])
    parser.add_argument("-l", nargs=1, type=int, default=[0])
    parser.add_argument("-m", nargs=1, type=int, default=[1])
    seed, cuda, render, load_model, save_model = init_env(parser.parse_args())

    device = "cuda" if cuda and torch.cuda.is_available() else "cpu"
    n_dof = 2

    train_data, test_data, divider, dt_mean = load_dataset()
    _, train_qp, train_qv, train_qa, _, _, train_tau = train_data

    hyper = {
        'n_width': 64, 'n_depth': 2, 'diagonal_epsilon': 0.01,
        'activation': 'SoftPlus',
        'b_init': 1.e-4, 'b_diag_init': 1.e-3,
        'w_init': 'xavier_normal',
        'gain_hidden': np.sqrt(2.), 'gain_output': 0.1,
        'n_minibatch': 512, 'learning_rate': 5.e-4,
        'weight_decay': 1.e-5, 'max_epoch': 300,
        'friction_type': 'formula',
        'init_tau_c': 0.05,
        'init_tau_s': 0.05,
        'init_nu':    0.5,
        'init_d':     0.05,
        'init_eps':   0.05,
    }

    model = DeepLagrangianNetwork(n_dof, **hyper).to(device)
    optim = torch.optim.Adam(model.parameters(),
                             lr=hyper['learning_rate'],
                             weight_decay=hyper['weight_decay'],
                             amsgrad=True)

    mem_dim = ((n_dof,), (n_dof,), (n_dof,), (n_dof,))
    mem = PyTorchReplayMemory(train_qp.shape[0], hyper['n_minibatch'], mem_dim, cuda)
    mem.add_samples([train_qp, train_qv, train_qa, train_tau])

    lambda_power = 1.e-2
    warmup_energy_epochs = 100

    t0 = time.perf_counter()
    print("\n################################################")
    print("Training DeLaN with FORMULA friction (route B):")

    for epoch in range(1, hyper['max_epoch'] + 1):
        lam_E = lambda_power if epoch > warmup_energy_epochs else 0.0
        agg = {'inv': 0., 'dE': 0., 'pdiss': 0., 'tot': 0., 'n': 0}

        for q, qd, qdd, tau in mem:
            optim.zero_grad()
            tau_hat, dEdt_hat, tau_f, p_diss = model(q, qd, qdd)

            err_inv = torch.sum((tau_hat - tau) ** 2, dim=1)
            l_inv = torch.mean(err_inv)

            P_ctrl = (qd * tau).sum(dim=1)
            l_dE = torch.mean((dEdt_hat - (P_ctrl - p_diss.squeeze(-1))) ** 2)

            loss = l_inv + lam_E * l_dE
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            agg['inv']   += l_inv.item()
            agg['dE']    += l_dE.item()
            agg['pdiss'] += p_diss.mean().item()
            agg['tot']   += loss.item()
            agg['n']     += 1

        if epoch == 1 or epoch % 50 == 0:
            n = agg['n']
            print(f"Epoch {epoch:05d} | t={time.perf_counter()-t0:6.1f}s "
                  f"| loss={agg['tot']/n:.3e} | inv={agg['inv']/n:.3e} "
                  f"| dEdt={agg['dE']/n:.3e} | mean P_diss={agg['pdiss']/n:.3e}")
            tc, ts, nu, d, eps = model.friction_formula.params()
            print(f"   τ_c={tc.detach().cpu().numpy()} "
                  f"τ_s={ts.detach().cpu().numpy()} "
                  f"ν={nu.detach().cpu().numpy()} "
                  f"d={d.detach().cpu().numpy()} "
                  f"ε={eps.detach().cpu().numpy()}")

    if save_model:
        ckpt_path = "data/delan_model_formula.torch"
        torch.save({
            "epoch": hyper['max_epoch'],
            "hyper": hyper,
            "state_dict": model.state_dict(),
        }, ckpt_path)
        print(f"\nsaved -> {ckpt_path}")


if __name__ == "__main__":
    main()
