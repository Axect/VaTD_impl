import torch
import wandb
import numpy as np

from util import run
from config import RunConfig, OptimizeConfig

import argparse


def create_adjacency_matrix(L, d=2):
    """
    Create adjacency matrix for a d-dimensional hypercube lattice with periodic boundary conditions.

    Args:
        L: lattice size (L^d total sites)
        d: dimension (default 2)

    Returns:
        adjacency matrix of shape (L^d, L^d)
    """
    N_site = L**d
    Adj = np.zeros((N_site, N_site), dtype=np.float32)

    def index2coord(idx, L, d):
        coord = []
        for _ in range(d):
            coord.append(idx % L)
            idx //= L
        return coord[::-1]

    def coord2index(coord, L):
        idx = coord[0]
        for i in range(1, len(coord)):
            idx = idx * L + coord[i]
        return idx

    # Build adjacency matrix
    for i in range(N_site):
        coord = index2coord(i, L, d)
        for dim in range(d):
            # Move in positive direction
            new_coord = coord.copy()
            new_coord[dim] = (new_coord[dim] + 1) % L  # periodic boundary
            j = coord2index(new_coord, L)
            Adj[i, j] = 1.0
            Adj[j, i] = 1.0

    return Adj


def create_ising_energy_fn(L, d=2, device="cpu"):
    """
    Create Ising energy function for a d-dimensional lattice.

    Args:
        L: lattice size
        d: dimension (default 2)
        device: torch device

    Returns:
        energy function that takes samples (B, 1, H, W) and returns (B, 1)
    """
    # Create adjacency matrix and convert to torch
    Adj = create_adjacency_matrix(L, d)
    N = (
        torch.from_numpy(Adj).float().to(device) * -1
    )  # multiply by -1 like VanillaIsing

    def energy_fn(samples):
        """
        Calculate Ising energy for batch of samples.

        Args:
            samples: (B, 1, H, W) tensor with values in {-1, 1}

        Returns:
            (B, 1) energy values
        """
        # Flatten spatial dimensions: (B, 1, H, W) -> (B, H*W)
        flat_samples = samples.flatten(-2)  # (B, 1, H*W)

        # Energy calculation: E = sum_ij J_ij * s_i * s_j / 2
        # = (s @ N) * s / 2 where N is adjacency matrix * -J
        energy = ((flat_samples @ N) * flat_samples).sum([-2, -1]).unsqueeze(-1) / 2

        return energy

    return energy_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_config", type=str, required=True, help="Path to the YAML config file"
    )
    parser.add_argument(
        "--optimize_config", type=str, help="Path to the optimization YAML config file"
    )
    parser.add_argument(
        "--device", type=str, help="Device to run on (e.g. 'cuda:0' or 'cpu')"
    )
    args = parser.parse_args()

    # Run Config
    base_config = RunConfig.from_yaml(args.run_config)

    # Device
    if args.device:
        base_config.device = args.device

    # Create Ising energy function
    # Get lattice size from model config
    model_config = base_config.gen_config().get("model_config", {})
    L = model_config.get("size", 16)
    if isinstance(L, (list, tuple)):
        L = L[0]  # assume square lattice

    energy_fn = create_ising_energy_fn(L=L, d=2, device=base_config.device)

    # Import exact partition function for validation
    from vatd_exact_partition import logZ as exact_logZ, CRITICAL_TEMPERATURE
    from util import generate_fixed_betas

    # Get model config for training beta range (from config)
    config = base_config.gen_config()
    net_config = config.get("net_config", {})
    train_beta_min = net_config.get("beta_min", 0.2)
    train_beta_max = net_config.get("beta_max", 1.0)
    num_beta = net_config.get("num_beta", 8)

    # Fixed validation beta range (for extrapolation testing)
    # Validation uses wider range than training to test generalization
    val_beta_min = 0.1
    val_beta_max = 2.0
    val_num_beta = 8

    # Generate fixed validation betas (wider range for extrapolation)
    fixed_val_betas = generate_fixed_betas(val_beta_min, val_beta_max, val_num_beta)

    # Compute exact logZ for each validation beta
    print(
        f"\nComputing exact partition function for {val_num_beta} validation temperatures..."
    )
    print(f"Training beta range: [{train_beta_min:.3f}, {train_beta_max:.3f}]")
    print(
        f"Validation beta range: [{val_beta_min:.3f}, {val_beta_max:.3f}] (wider for extrapolation)"
    )
    print(
        f"Critical temperature: Tc = {CRITICAL_TEMPERATURE:.6f} (βc = {1.0/CRITICAL_TEMPERATURE:.6f})"
    )

    exact_logz_values = []
    for i, beta in enumerate(fixed_val_betas):
        exact_logz = exact_logZ(n=L, j=1.0, beta=beta.item())
        exact_logz_values.append(exact_logz.item())
        T = 1.0 / beta.item()
        print(
            f"  β_{i} = {beta.item():.4f} (T = {T:.4f}): log Z = {exact_logz.item():.6f}"
        )

    # Attach to energy_fn as attributes (for trainer access)
    energy_fn.exact_logz_values = exact_logz_values
    energy_fn.fixed_val_betas = fixed_val_betas.tolist()
    energy_fn.lattice_size = L
    energy_fn.critical_temperature = CRITICAL_TEMPERATURE
    print("Exact partition function values attached to energy_fn\n")

    # Run
    if args.optimize_config:
        optimize_config = OptimizeConfig.from_yaml(args.optimize_config)
        pruner = optimize_config.create_pruner()

        def objective(trial, base_config, optimize_config, energy_fn):
            params = optimize_config.suggest_params(trial)

            config = base_config.gen_config()
            config["project"] = f"{base_config.project}_Opt"
            for category, category_params in params.items():
                config[category].update(category_params)

            run_config = RunConfig(**config)
            group_name = run_config.gen_group_name()
            group_name += f"[{trial.number}]"

            trial.set_user_attr("group_name", group_name)

            return run(run_config, energy_fn, group_name, trial=trial, pruner=pruner)

        study = optimize_config.create_study(project=f"{base_config.project}_Opt")
        study.optimize(
            lambda trial: objective(trial, base_config, optimize_config, energy_fn),
            n_trials=optimize_config.trials,
        )

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print(
            f"  Path: runs/{base_config.project}_Opt/{trial.user_attrs['group_name']}"
        )

    else:
        run(base_config, energy_fn)


if __name__ == "__main__":
    main()
