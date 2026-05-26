
import numpy as np
import torch


def compare_gravity(
    models,
    test_configs=None,
    analytical_params=None,
    print_diffs=True,
):
    """Print learned gravity vectors from multiple models alongside analytical"""
    if test_configs is None:
        test_configs = [
            (0., 0.),
            (np.pi / 2, 0.),
            (np.pi / 2, -np.pi / 2),
            (0., np.pi / 2),
            (np.pi / 4, np.pi / 4),
            (np.pi, np.pi),
            (2*np.pi/3, -2*np.pi/3),
        ]

    if analytical_params is None:
        analytical_params = dict(
            m1=0.131, m2=0.064,
            L1=0.05, L_com1=0.046, L_com2=0.048,
            g=9.81,
        )

    m1 = analytical_params['m1']; m2 = analytical_params['m2']
    L1 = analytical_params['L1']
    Lc1 = analytical_params['L_com1']; Lc2 = analytical_params['L_com2']
    g_val = analytical_params['g']

    def analytical(q1, q2):
        g1 = (m1 * Lc1 + m2 * L1) * g_val * np.sin(q1) + m2 * Lc2 * g_val * np.sin(q1 + q2)
        g2 = m2 * Lc2 * g_val * np.sin(q1 + q2)
        return np.array([g1, g2])

    results = {label: [] for label in models}
    results['analytical'] = []

    # ---- Header ----
    labels = list(models.keys())
    col_width = 22
    header = f"  {'q = (q1, q2)':>16s}  | " + f"{'analytical':>{col_width}s} | " \
             + " | ".join(f"{lab:>{col_width}s}" for lab in labels)
    print(header)
    print("-" * len(header))

    # ---- One row per config ----
    for q1, q2 in test_configs:
        q = torch.tensor([[q1, q2]], dtype=torch.float32)
        qd = torch.zeros_like(q)
        zeros = torch.zeros_like(q)

        ga = analytical(q1, q2)
        results['analytical'].append(ga)
        ga_str = f"({ga[0]:+.4f}, {ga[1]:+.4f})"

        learned_strs = []
        for label, model in models.items():
            model.eval()
            with torch.no_grad():
                gl = model._dyn_model(q, qd, zeros)[3].squeeze().cpu().numpy()
            results[label].append(gl)
            learned_strs.append(f"({gl[0]:+.4f}, {gl[1]:+.4f})")

        config_str = f"({q1:+.2f}, {q2:+.2f})"
        row = f"  {config_str:>16s}  | {ga_str:>{col_width}s} | " \
              + " | ".join(f"{s:>{col_width}s}" for s in learned_strs)
        print(row)

    # ---- Per-model error summary ----
    if print_diffs:
        print()
        print("  Mean Euclidean error vs analytical across the tested configs:")
        for label in labels:
            diffs = [np.linalg.norm(results[label][i] - results['analytical'][i])
                     for i in range(len(test_configs))]
            mean = np.mean(diffs); worst = np.max(diffs); worst_i = int(np.argmax(diffs))
            wq1, wq2 = test_configs[worst_i]
            print(f"    {label:25s}  mean={mean:.4f} Nm  worst={worst:.4f} Nm at q=({wq1:+.2f}, {wq2:+.2f})")

    return results


if __name__ == '__main__':
    # Example - replace with your loaded models
    import sys
    sys.path.insert(0, '..')
    from training_v2.model import DeepLagrangianNetwork

    # ----- helper to load a checkpoint -----
    def load_model(ckpt_path, **model_kwargs):
        defaults = dict(n_dof=2, learn_friction=True,
                        n_width=64, n_depth=2, activation='SoftPlus',
                        diagonal_epsilon=1e-3, b_diag_init=0.5)
        defaults.update(model_kwargs)
        m = DeepLagrangianNetwork(**defaults)
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        m.load_state_dict(ck['model']); m.eval()
        return m

    model_main = load_model('training_v2/checkpoints/delan_best.pt')

    compare_gravity(
        models={
            'main':       model_main,
            # 'savgol':     model_savgol
        },
    )