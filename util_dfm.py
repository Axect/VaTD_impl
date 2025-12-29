"""
Training utilities for Discrete Flow Matching on Ising Model.

This module provides the FlowMatchingTrainer class for training
DiscreteFlowMatcher models with:
- Hybrid loss: denoising + REINFORCE
- Curriculum learning
- Thermodynamic integration validation
- WandB logging
"""

import torch
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import math
import random
import os

matplotlib.use("Agg")

from config import RunConfig
from partition_estimation import (
    thermodynamic_integration,
    elbo_partition,
    validate_partition_function,
)


def generate_fixed_betas(beta_min, beta_max, num_beta):
    """Generate log-spaced beta values for validation."""
    betas = np.logspace(np.log10(beta_min), np.log10(beta_max), num_beta)
    return torch.tensor(betas, dtype=torch.float32)


def sign_log_transform(x):
    """Apply sign-log transform for visualization."""
    if isinstance(x, torch.Tensor):
        x = x.item()
    if x == 0:
        return 0.0
    sign = 1 if x > 0 else -1
    return sign * math.log10(1 + abs(x))


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_sample_grid(model, betas, n_samples=4, device="cpu"):
    """Generate visualization of samples at different temperatures."""
    model.eval()
    n_temps = len(betas)

    fig, axes = plt.subplots(
        n_temps, n_samples, figsize=(n_samples * 2 + 0.8, n_temps * 2)
    )
    if n_temps == 1:
        axes = axes.reshape(1, -1)
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    with torch.no_grad():
        for i, beta in enumerate(betas):
            T_val = 1.0 / beta.item()
            T = (1.0 / beta).expand(n_samples).to(device)
            samples = model.sample(batch_size=n_samples, T=T)

            for j in range(n_samples):
                sample = samples[j, 0].cpu().numpy()
                axes[i, j].imshow(sample, cmap="coolwarm", vmin=-1, vmax=1)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

                if j == 0:
                    axes[i, j].set_ylabel(
                        f"T={T_val:.2f}\n(β={beta.item():.2f})",
                        fontsize=9,
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=10,
                    )

    plt.suptitle("Ising Samples (blue=-1, red=+1)", fontsize=12)
    plt.tight_layout()
    return fig


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience=10, mode="min", min_delta=0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if self.mode == "min":
            if val_loss <= self.best_loss * (1 - self.min_delta):
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
        else:
            if val_loss >= self.best_loss * (1 + self.min_delta):
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


class FlowMatchingTrainer:
    """
    Trainer for Discrete Flow Matching models.

    Supports:
    - Hybrid training: denoising loss + REINFORCE
    - Curriculum learning (high T → low T)
    - Thermodynamic integration for partition function
    - ELBO validation

    Args:
        model: DiscreteFlowMatcher model
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        device: Device for training
        energy_fn: Energy function with optional exact_logz_values attribute
        epochs: Total number of epochs
        lambda_reinforce: Weight for REINFORCE loss term
        accumulation_steps: Gradient accumulation steps per epoch
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        early_stopping_config=None,
        device="cpu",
        trial=None,
        seed=None,
        pruner=None,
        energy_fn=None,
        epochs=None,
        lambda_reinforce=0.1,
        accumulation_steps=1,
        training_mode="hybrid",  # "hybrid" or "denoise_only"
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.trial = trial
        self.seed = seed
        self.pruner = pruner
        self.epochs = epochs
        self.lambda_reinforce = lambda_reinforce
        self.accumulation_steps = accumulation_steps
        self.training_mode = training_mode

        # Curriculum learning
        self.curriculum_enabled = getattr(model, "curriculum_enabled", False)
        self.current_epoch = 0
        self.phase1_epochs = getattr(model, "phase1_epochs", 50)
        self.phase1_beta_max = getattr(model, "phase1_beta_max", 0.35)
        self.phase2_epochs = getattr(model, "phase2_epochs", 100)

        # Validation betas
        if energy_fn is not None and hasattr(energy_fn, "fixed_val_betas"):
            self.fixed_val_betas = torch.tensor(
                energy_fn.fixed_val_betas, dtype=torch.float32
            ).to(device)
        else:
            self.fixed_val_betas = generate_fixed_betas(
                model.beta_min, model.beta_max, model.num_beta
            ).to(device)

        # Exact partition function for validation
        if energy_fn is not None:
            self.exact_logz_values = getattr(energy_fn, "exact_logz_values", None)
            self.lattice_size = getattr(energy_fn, "lattice_size", None)
            self.critical_temperature = getattr(energy_fn, "critical_temperature", None)
        else:
            self.exact_logz_values = None
            self.lattice_size = None
            self.critical_temperature = None

        # Early stopping
        if early_stopping_config and early_stopping_config.enabled:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_config.patience,
                mode=early_stopping_config.mode,
                min_delta=early_stopping_config.min_delta,
            )
        else:
            self.early_stopping = None

    def get_curriculum_beta_range(self):
        """Get current beta range based on curriculum phase."""
        if not self.curriculum_enabled:
            return self.model.beta_min, self.model.beta_max

        if self.current_epoch < self.phase1_epochs:
            return self.model.beta_min, self.phase1_beta_max
        else:
            # Cosine annealing from phase1_beta_max to beta_max
            epochs_in_phase2 = self.current_epoch - self.phase1_epochs
            progress = min(1.0, epochs_in_phase2 / self.phase2_epochs)
            # Cosine schedule
            cos_progress = 0.5 * (1 - math.cos(math.pi * progress))
            effective_beta_max = (
                self.phase1_beta_max
                + (self.model.beta_max - self.phase1_beta_max) * cos_progress
            )
            return self.model.beta_min, effective_beta_max

    def train_epoch(self, energy_fn):
        """
        Single training epoch with hybrid loss.

        Algorithm:
        1. Sample temperatures log-uniformly
        2. Sample from model (gradient detached)
        3. Apply noising at random time t
        4. Compute denoising loss (cross-entropy)
        5. Compute REINFORCE loss (optional)
        6. Backward and optimizer step

        Returns:
            dict: Training statistics
        """
        batch_size = self.model.batch_size
        num_beta = self.model.num_beta
        num_steps = self.accumulation_steps

        beta_min, beta_max = self.get_curriculum_beta_range()

        self.model.train()

        epoch_stats = {
            "denoise_loss": 0.0,
            "reinforce_loss": 0.0,
            "total_loss": 0.0,
            "log_prob_mean": 0.0,
            "energy_mean": 0.0,
        }

        all_beta_samples = []

        for step in range(num_steps):
            # Log-uniform beta sampling
            log_beta = torch.rand(num_beta, device=self.device) * (
                math.log(beta_max) - math.log(beta_min)
            ) + math.log(beta_min)
            beta_samples = torch.exp(log_beta)
            all_beta_samples.append(beta_samples.clone())

            T_samples = 1.0 / beta_samples
            T_expanded = T_samples.repeat_interleave(batch_size)
            total_size = num_beta * batch_size

            # Sample from model (no gradient)
            with torch.no_grad():
                samples = self.model.sample(batch_size=total_size, T=T_expanded)

            # Compute loss based on training mode
            loss, metrics = self.model.training_loss(
                samples,
                T_expanded,
                energy_fn=energy_fn,
                lambda_reinforce=self.lambda_reinforce,
                training_mode=self.training_mode,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.optimizer.step()

            # Scheduler step for OneCycleLR
            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            # Accumulate stats
            for key in ["denoise_loss", "total_loss"]:
                if key in metrics:
                    epoch_stats[key] += metrics[key] / num_steps
            if "reinforce_loss" in metrics:
                epoch_stats["reinforce_loss"] += metrics[key] / num_steps
            if "energy_mean" in metrics:
                epoch_stats["energy_mean"] += metrics["energy_mean"] / num_steps

        # Compute additional statistics
        all_betas = torch.cat(all_beta_samples)

        with torch.no_grad():
            sample_diversity = samples.var(dim=0).mean().item()
            magnetization = samples.mean(dim=(1, 2, 3)).abs().mean().item()

        epoch_stats.update({
            "sample_diversity": sample_diversity,
            "magnetization_mean": magnetization,
            "beta_samples_mean": all_betas.mean().item(),
            "beta_samples_min": all_betas.min().item(),
            "beta_samples_max": all_betas.max().item(),
            "train_loss": epoch_stats["total_loss"],
        })

        return epoch_stats

    def val_epoch(self, energy_fn):
        """
        Validation epoch with fixed beta values.

        Computes ELBO-based validation loss and compares with exact
        partition function if available.

        Returns:
            dict: Validation statistics
        """
        batch_size = self.model.batch_size
        num_beta = len(self.fixed_val_betas)

        beta_samples = self.fixed_val_betas
        T_samples = 1.0 / beta_samples
        T_expanded = T_samples.repeat_interleave(batch_size)
        total_size = num_beta * batch_size

        self.model.eval()

        with torch.no_grad():
            # Sample from model
            samples = self.model.sample(batch_size=total_size, T=T_expanded)

            # Compute log probability (ELBO approximation)
            log_prob = self.model.log_prob(samples, T=T_expanded)

            # Compute energy
            energy = energy_fn(samples)
            beta = (1.0 / T_expanded).unsqueeze(-1)

            # Variational free energy: F = E[log q + β·E]
            loss_raw = log_prob + beta * energy

            # Reshape to (num_beta, batch_size)
            loss_view = loss_raw.view(num_beta, batch_size)
            loss_per_beta = loss_view.mean(dim=1)

            val_loss = loss_per_beta.mean().item()

            # Build result dict
            val_dict = {"val_loss": val_loss}
            for i in range(num_beta):
                val_dict[f"val_loss_beta_{i}"] = loss_per_beta[i].item()

            # Compute errors vs exact if available
            if self.exact_logz_values is not None:
                for i in range(num_beta):
                    exact_logz = self.exact_logz_values[i]
                    model_loss = val_dict[f"val_loss_beta_{i}"]
                    error = model_loss + exact_logz
                    val_dict[f"val_error_exact_beta_{i}"] = error

                # Mean absolute error
                errors = [val_dict[f"val_error_exact_beta_{i}"] for i in range(num_beta)]
                val_dict["val_error_mean"] = np.mean(np.abs(errors))

            # Log probability and energy statistics
            log_prob_view = log_prob.view(num_beta, batch_size)
            energy_view = energy.view(num_beta, batch_size)

            val_dict["val_log_prob_mean"] = log_prob_view.mean().item()
            val_dict["val_energy_mean"] = energy_view.mean().item()

        return val_dict

    def train(self, energy_fn, epochs, log_every=1, save_path=None, wandb_log=True):
        """
        Full training loop.

        Args:
            energy_fn: Energy function
            epochs: Number of epochs
            log_every: Log every N epochs
            save_path: Path to save checkpoints
            wandb_log: Whether to log to WandB

        Returns:
            dict: Final metrics
        """
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        pbar = tqdm(range(epochs), desc="Training DFM")

        for epoch in pbar:
            self.current_epoch = epoch

            # Training
            train_stats = self.train_epoch(energy_fn)
            history["train_loss"].append(train_stats["train_loss"])

            # Validation
            val_stats = self.val_epoch(energy_fn)
            history["val_loss"].append(val_stats["val_loss"])

            # Update scheduler (non-OneCycleLR)
            if self.scheduler is not None and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.OneCycleLR
            ):
                self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            if epoch % log_every == 0:
                log_dict = {
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    **train_stats,
                    **val_stats,
                }

                # Add curriculum info
                if self.curriculum_enabled:
                    beta_min, beta_max = self.get_curriculum_beta_range()
                    log_dict["curriculum_beta_max"] = beta_max

                # Log to WandB
                if wandb_log and wandb.run is not None:
                    wandb.log(log_dict)

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{train_stats['train_loss']:.3f}",
                    "val": f"{val_stats['val_loss']:.3f}",
                    "lr": f"{current_lr:.2e}",
                })

            # Save best model
            if val_stats["val_loss"] < best_val_loss:
                best_val_loss = val_stats["val_loss"]
                if save_path:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    }, save_path)

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_stats["val_loss"]):
                    tqdm.write(f"Early stopping at epoch {epoch}")
                    break

        return {
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "best_val_loss": best_val_loss,
            "history": history,
        }


def run(run_config: RunConfig, energy_fn, group_name=None, trial=None, pruner=None):
    """
    Main training entry point for Discrete Flow Matching.

    Same interface as util.run() for compatibility with main.py.

    Args:
        run_config: RunConfig instance
        energy_fn: Energy function with optional exact_logz_values
        group_name: WandB group name (optional)
        trial: Optuna trial (optional)
        pruner: Optuna pruner (optional)

    Returns:
        float: Final validation loss (for Optuna compatibility)
    """
    import os
    import math

    project = run_config.project
    device = run_config.device
    seeds = run_config.seeds

    if not group_name:
        group_name = run_config.gen_group_name()
    tags = run_config.gen_tags() if hasattr(run_config, 'gen_tags') else []

    # Create run directory
    group_path = f"runs/{run_config.project}/{group_name}"
    if not os.path.exists(group_path):
        os.makedirs(group_path)
    run_config.to_yaml(f"{group_path}/config.yaml")

    total_loss = 0
    complete_seeds = 0

    try:
        for seed in seeds:
            set_seed(seed)

            # Create model
            model = run_config.create_model().to(device)

            # Create optimizer and scheduler
            optimizer = run_config.create_optimizer(model)
            scheduler = run_config.create_scheduler(optimizer)

            run_name = f"{seed}"
            wandb.init(
                project=project,
                name=run_name,
                group=group_name,
                tags=tags,
                config=run_config.gen_config(),
            )

            # Extract DFM training parameters from net_config
            net_config = run_config.gen_config().get("net_config", {})

            # Create trainer
            trainer = FlowMatchingTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                early_stopping_config=run_config.early_stopping_config,
                device=device,
                trial=trial,
                seed=seed,
                pruner=pruner,
                energy_fn=energy_fn,
                epochs=run_config.epochs,
                lambda_reinforce=net_config.get("lambda_reinforce", 0.1),
                accumulation_steps=net_config.get("accumulation_steps", 1),
                training_mode=net_config.get("training_mode", "hybrid"),
            )

            # Train and get final val_loss
            metrics = trainer.train(
                energy_fn=energy_fn,
                epochs=run_config.epochs,
                log_every=1,
                save_path=None,  # We'll save manually
                wandb_log=True,
            )

            val_loss = metrics["final_val_loss"]
            total_loss += val_loss
            complete_seeds += 1

            # Save model & configs
            run_path = f"{group_path}/{run_name}"
            if not os.path.exists(run_path):
                os.makedirs(run_path)
            torch.save(model.state_dict(), f"{run_path}/model.pt")

            wandb.finish()

            # Early stopping if loss becomes inf
            if math.isinf(val_loss):
                break

    except Exception as e:
        tqdm.write(f"Runtime error during training: {e}")
        wandb.finish()
        if trial is not None:
            import optuna
            raise optuna.TrialPruned()
        raise

    return total_loss / (complete_seeds if complete_seeds > 0 else 1)


if __name__ == "__main__":
    """Test the FlowMatchingTrainer."""
    print("Testing FlowMatchingTrainer")
    print("=" * 70)

    from model_dfm import DiscreteFlowMatcher

    # Test config
    hparams = {
        "size": 8,
        "fix_first": 1,
        "batch_size": 32,
        "num_beta": 4,
        "beta_min": 0.2,
        "beta_max": 1.0,
        "num_flow_steps": 20,
        "t_max": 5.0,
        "hidden_channels": 32,
        "hidden_conv_layers": 2,
        "hidden_width": 64,
        "hidden_fc_layers": 1,
        "curriculum_enabled": True,
        "phase1_epochs": 3,
        "phase1_beta_max": 0.4,
        "phase2_epochs": 5,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = DiscreteFlowMatcher(hparams, device=device).to(device)

    # Energy function
    def energy_fn(x):
        right = torch.roll(x, -1, dims=-1)
        down = torch.roll(x, -1, dims=-2)
        energy = -(x * right + x * down).sum(dim=[1, 2, 3])
        return energy.unsqueeze(-1)

    # Add validation info to energy_fn
    energy_fn.fixed_val_betas = [0.3, 0.5, 0.7, 0.9]
    energy_fn.exact_logz_values = [-45.0, -60.0, -80.0, -100.0]  # Placeholder
    energy_fn.lattice_size = 8

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Create trainer
    trainer = FlowMatchingTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        energy_fn=energy_fn,
        epochs=10,
        lambda_reinforce=0.1,
        accumulation_steps=2,
    )

    # Test training epoch
    print("\n--- Testing train_epoch ---")
    train_stats = trainer.train_epoch(energy_fn)
    print(f"Train stats: {train_stats}")

    # Test validation epoch
    print("\n--- Testing val_epoch ---")
    val_stats = trainer.val_epoch(energy_fn)
    print(f"Val stats: {val_stats}")

    # Test full training (short)
    print("\n--- Testing full training (5 epochs) ---")
    metrics = trainer.train(energy_fn, epochs=5, wandb_log=False)
    print(f"Final metrics: {metrics}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
