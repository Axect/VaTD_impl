import torch
import wandb
import numpy as np
import math

from config import RunConfig, OptimizeConfig

import argparse


def is_dfm_model(net_name: str) -> bool:
    """Check if the model is a Discrete Flow Matching model."""
    net_lower = net_name.lower()
    return "dfm" in net_lower or "flowmatch" in net_lower or "discreteflowmatcher" in net_lower


def is_xy_model(net_name: str) -> bool:
    """Check if the model is for XY model (continuous angles)."""
    net_lower = net_name.lower()
    return "xy" in net_lower or "continuous" in net_lower


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
    Uses O(N) sparse computation instead of O(N²) adjacency matrix.

    Args:
        L: lattice size
        d: dimension (default 2)
        device: torch device

    Returns:
        energy function that takes samples (B, 1, H, W) and returns (B, 1)
    """

    def energy_fn(samples):
        """
        Calculate Ising energy for batch of samples using sparse nearest-neighbor computation.
        O(N) complexity instead of O(N²) with adjacency matrix.

        Args:
            samples: (B, 1, H, W) tensor with values in {-1, 1}

        Returns:
            (B, 1) energy values

        Energy: E = -J * sum_{<i,j>} s_i * s_j  (J=1)
        With periodic boundary conditions, each bond counted once.
        """
        # Count each bond once: right and down neighbors only (periodic BCs via roll)
        # This avoids double-counting that would require /2
        right_neighbor = torch.roll(samples, shifts=-1, dims=-1)
        down_neighbor = torch.roll(samples, shifts=-1, dims=-2)

        # E = -J * sum(s_i * s_j) for each bond, J=1
        energy = -(samples * right_neighbor + samples * down_neighbor)
        energy = energy.sum(dim=[-1, -2, -3]).unsqueeze(-1)

        return energy

    return energy_fn


def create_xy_energy_fn(L, d=2, device="cpu", J=1.0):
    """
    Create XY energy function for a d-dimensional lattice.
    Uses O(N) sparse computation with periodic boundary conditions.

    XY Model: Spins are continuous angles theta in [-pi, pi]
    Energy: E = -J * sum_{<i,j>} cos(theta_i - theta_j)

    Args:
        L: lattice size
        d: dimension (default 2)
        device: torch device
        J: coupling constant (default 1.0)

    Returns:
        energy function that takes samples (B, 1, H, W) and returns (B, 1)
    """

    def energy_fn(samples):
        """
        Calculate XY energy for batch of samples.

        Args:
            samples: (B, 1, H, W) tensor with angles in [-pi, pi]

        Returns:
            (B, 1) energy values

        Energy: E = -J * sum_{<i,j>} cos(theta_i - theta_j)
        With periodic boundary conditions, each bond counted once.
        """
        # Angle differences with periodic boundary conditions
        diff_right = samples - torch.roll(samples, shifts=-1, dims=-1)
        diff_down = samples - torch.roll(samples, shifts=-1, dims=-2)

        # E = -J * sum cos(theta_i - theta_j) for each bond
        energy = -J * (torch.cos(diff_right) + torch.cos(diff_down))
        energy = energy.sum(dim=[-1, -2, -3]).unsqueeze(-1)

        return energy

    # Attach metadata
    energy_fn.model_type = "xy"
    energy_fn.lattice_size = L
    energy_fn.J = J

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

    # Get lattice size from model config
    model_config = base_config.gen_config().get("model_config", {})
    L = model_config.get("size", 16)
    if isinstance(L, (list, tuple)):
        L = L[0]  # assume square lattice

    # Get model config for training beta range (from config)
    config = base_config.gen_config()
    net_config = config.get("net_config", {})
    train_beta_min = net_config.get("beta_min", 0.2)
    train_beta_max = net_config.get("beta_max", 1.0)
    num_beta = net_config.get("num_beta", 8)
    fix_first = net_config.get("fix_first", None)

    # Detect model type (XY vs Ising)
    use_xy = is_xy_model(base_config.net)

    if use_xy:
        # XY Model: continuous angles
        J = net_config.get("J", 1.0)
        energy_fn = create_xy_energy_fn(L=L, d=2, device=base_config.device, J=J)

        from util import generate_fixed_betas

        # Fixed validation beta range for XY model
        # BKT transition at T ~ 0.89 (beta ~ 1.12)
        val_beta_min = 0.5
        val_beta_max = 2.0
        val_num_beta = 8

        fixed_val_betas = generate_fixed_betas(val_beta_min, val_beta_max, val_num_beta)

        print(f"\n[XY Model] No exact partition function available (BKT transition)")
        print(f"Training beta range: [{train_beta_min:.3f}, {train_beta_max:.3f}]")
        print(f"Validation beta range: [{val_beta_min:.3f}, {val_beta_max:.3f}]")
        print(f"BKT transition: T_BKT ~ 0.89 (β_BKT ~ 1.12)")

        # Try to load MCMC reference data if available
        import os
        ref_path = f"refs/xy_L{L}.pt"
        if os.path.exists(ref_path):
            mcmc_ref = torch.load(ref_path)
            energy_fn.mcmc_reference = mcmc_ref
            print(f"Loaded MCMC reference data from {ref_path}")
        else:
            energy_fn.mcmc_reference = None
            print(f"No MCMC reference data at {ref_path} (run generate_xy_reference.py)")

        # Attach validation betas (no exact logZ for XY)
        energy_fn.fixed_val_betas = fixed_val_betas.tolist()
        energy_fn.exact_logz_values = None  # No exact solution for XY
        print()

    else:
        # Ising Model: discrete spins
        energy_fn = create_ising_energy_fn(L=L, d=2, device=base_config.device)

        # Import exact partition function for validation
        from vatd_exact_partition import logZ as exact_logZ, CRITICAL_TEMPERATURE
        from util import generate_fixed_betas

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
            # If the model fixes the first spin, adjust exact logZ to the same conditional partition
            if fix_first is not None:
                exact_logz = exact_logz - math.log(2.0)
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

    # Select run function based on model type
    use_dfm = is_dfm_model(base_config.net)
    if use_dfm:
        from util_dfm import run as run_fn
        print(f"[DFM] Using Discrete Flow Matching trainer for {base_config.net}")
    else:
        from util import run as run_fn
        print(f"[PixelCNN] Using standard trainer for {base_config.net}")

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

            return run_fn(run_config, energy_fn, group_name, trial=trial, pruner=pruner)

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
        run_fn(base_config, energy_fn)


if __name__ == "__main__":
    main()
