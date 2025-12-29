import torch
import torch.nn.functional as F
import numpy as np
import beaupy
from rich.console import Console
import wandb
import optuna
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server

from config import RunConfig
from ising import swendsen_wang_update  # Import Swendsen-Wang

import random
import os
import math


def generate_fixed_betas(beta_min, beta_max, num_beta):
    """
    Generate log-spaced beta values for fixed validation.

    Args:
        beta_min (float): Minimum beta (1/T_max)
        beta_max (float): Maximum beta (1/T_min)
        num_beta (int): Number of beta values

    Returns:
        torch.Tensor: shape (num_beta,) with log-spaced beta values

    Example:
        >>> generate_fixed_betas(0.1, 2.0, 8)
        tensor([0.1000, 0.1468, 0.2154, 0.3162, 0.4642, 0.6813, 1.0000, 1.4678])
    """
    betas = np.logspace(np.log10(beta_min), np.log10(beta_max), num_beta)
    return torch.tensor(betas, dtype=torch.float32)


def sign_log_transform(x):
    """
    Apply sign-log transform: sign(x) * log10(1 + |x|)

    This transform preserves sign while compressing the magnitude using log scale.
    Useful for visualizing negative losses on log-scale plots.

    Args:
        x (float or torch.Tensor): Input value(s)

    Returns:
        float: Transformed value

    Examples:
        >>> sign_log_transform(99)  # log10(100) = 2
        2.0
        >>> sign_log_transform(-99)  # -log10(100) = -2
        -2.0
        >>> sign_log_transform(0)
        0.0
    """
    if isinstance(x, torch.Tensor):
        x = x.item()

    if x == 0:
        return 0.0

    sign = 1 if x > 0 else -1
    return sign * math.log10(1 + abs(x))


def create_sample_grid(model, betas, n_samples=4, device="cpu"):
    """
    Generate and visualize Ising model samples at different temperatures.

    Args:
        model: DiscretePixelCNN model
        betas: tensor of beta values (1/T)
        n_samples: number of samples per temperature
        device: torch device

    Returns:
        matplotlib figure
    """
    model.eval()
    n_temps = len(betas)

    # Extra width for temperature labels
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
                sample = samples[j, 0].cpu().numpy()  # (H, W)
                axes[i, j].imshow(sample, cmap="coolwarm", vmin=-1, vmax=1)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

                # Add temperature label on the left of first column
                if j == 0:
                    axes[i, j].set_ylabel(
                        f"T={T_val:.2f}\n(β={beta.item():.2f})",
                        fontsize=9,
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=10,
                    )

    plt.suptitle("Ising Samples (blue=−1, red=+1)", fontsize=12)
    plt.tight_layout()
    return fig


def set_seed(seed: int):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
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
        else:  # mode == "max"
            if val_loss >= self.best_loss * (1 + self.min_delta):
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


def predict_final_loss(losses, max_epochs):
    if len(losses) < 10:
        return -np.log10(losses[-1])
    try:
        # Convert to numpy array
        y = np.array(losses)
        t = np.arange(len(y))

        # Fit a linear model to the log of the losses
        y_transformed = np.log(y)
        K, log_A = np.polyfit(t, y_transformed, 1)
        A = np.exp(log_A)

        # Predict final loss
        predicted_loss = -np.log10(A * np.exp(K * max_epochs))

        if np.isfinite(predicted_loss):
            return predicted_loss

    except Exception as e:
        tqdm.write(f"Error in loss prediction: {e}")

    return -np.log10(losses[-1])


class Trainer:
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
        # v0.11 parameters
        accumulation_steps=1,       # Optimizer steps per epoch (for accumulated mode)
        training_mode="standard",   # "standard" or "accumulated"
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.trial = trial
        self.seed = seed
        self.pruner = pruner
        self.epochs = epochs

        # v0.11: Training mode settings
        self.accumulation_steps = accumulation_steps
        self.training_mode = training_mode

        # Hybrid Training / MCMC Correction Settings
        # Checks if model hparams has mcmc_enabled, else defaults to False
        self.mcmc_enabled = getattr(model, "hparams", {}).get("mcmc_enabled", False)
        self.mcmc_freq = getattr(model, "hparams", {}).get("mcmc_freq", 5)
        self.mcmc_steps = getattr(model, "hparams", {}).get("mcmc_steps", 10)
        self.mcmc_weight = getattr(model, "hparams", {}).get("mcmc_weight", 1.0)

        # Curriculum learning settings
        # 2-Phase curriculum: High temp only → Gradual expansion to full range
        self.curriculum_enabled = getattr(model, "curriculum_enabled", False)
        self.current_epoch = 0

        # Phase 1: High temperature only (disordered phase)
        self.phase1_epochs = getattr(model, "phase1_epochs", 50)
        self.phase1_beta_max = getattr(model, "phase1_beta_max", 0.35)

        # Phase 2: Gradual expansion to full range
        self.phase2_epochs = getattr(model, "phase2_epochs", 100)

        # Pre-compute fixed betas for validation (from energy_fn)
        # Validation uses fixed wider range (0.1-2.0) for extrapolation testing
        # Training uses model's beta range from config
        if energy_fn is not None and hasattr(energy_fn, "fixed_val_betas"):
            self.fixed_val_betas = torch.tensor(
                energy_fn.fixed_val_betas, dtype=torch.float32
            ).to(device)
        else:
            # Fallback: use model's beta range if energy_fn doesn't provide validation betas
            self.fixed_val_betas = generate_fixed_betas(
                model.beta_min, model.beta_max, model.num_beta
            ).to(device)

        # Store exact partition function values for validation (if available)
        if energy_fn is not None:
            self.exact_logz_values = getattr(energy_fn, "exact_logz_values", None)
            self.lattice_size = getattr(energy_fn, "lattice_size", None)
            self.critical_temperature = getattr(energy_fn, "critical_temperature", None)
        else:
            self.exact_logz_values = None
            self.lattice_size = None
            self.critical_temperature = None

        if early_stopping_config and early_stopping_config.enabled:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_config.patience,
                mode=early_stopping_config.mode,
                min_delta=early_stopping_config.min_delta,
            )
        else:
            self.early_stopping = None

    def get_curriculum_phase(self):
        """
        Determine current curriculum phase based on epoch.

        Returns:
            int: Phase number (1 or 2)
        """
        if self.current_epoch < self.phase1_epochs:
            return 1
        else:
            return 2

    def get_curriculum_beta_range(self):
        """
        Get current beta range based on 2-phase curriculum learning schedule.

        Phase 1: High temp only (β ∈ [beta_min, phase1_beta_max])
        Phase 2: Gradual expansion to full range, then stay at full range

        Returns:
            tuple: (beta_min, effective_beta_max)
        """
        if not self.curriculum_enabled:
            return self.model.beta_min, self.model.beta_max

        phase = self.get_curriculum_phase()

        if phase == 1:
            # Phase 1: High temperature only
            return self.model.beta_min, self.phase1_beta_max

        else:
            # Phase 2: Gradual expansion from phase1_beta_max to full beta_max
            phase2_start = self.phase1_epochs
            phase2_progress = (self.current_epoch - phase2_start) / self.phase2_epochs
            phase2_progress = min(1.0, phase2_progress)
            # Cosine annealing for smooth transition
            phase2_progress = 0.5 * (1 - math.cos(math.pi * phase2_progress))

            effective_beta_max = (
                self.phase1_beta_max
                + phase2_progress * (self.model.beta_max - self.phase1_beta_max)
            )
            return self.model.beta_min, effective_beta_max

    def step(self, batch_size, T):
        # Sample from PixelCNN (no gradient tracking needed for sampling)
        with torch.no_grad():
            samples = self.model.sample(batch_size=batch_size, T=T)
        log_prob = self.model.log_prob(samples, T=T)
        return log_prob, samples

    def train_epoch(self, energy_fn):
        batch_size = self.model.batch_size
        num_beta = self.model.num_beta

        # Get curriculum-adjusted beta range
        beta_min, beta_max = self.get_curriculum_beta_range()

        # Log-uniform sampling over current beta range
        log_beta_samples = torch.rand(num_beta, device=self.device) * (
            math.log(beta_max) - math.log(beta_min)
        ) + math.log(beta_min)
        beta_samples = torch.exp(log_beta_samples)

        T_samples = 1.0 / beta_samples
        T_expanded = T_samples.repeat_interleave(batch_size)
        total_size = num_beta * batch_size

        self.model.train()
        # ScheduleFree Optimizer or SPlus
        if any(
            keyword in self.optimizer.__class__.__name__
            for keyword in ["ScheduleFree", "SPlus"]
        ):
            self.optimizer.train()

        log_prob, samples = self.step(batch_size=total_size, T=T_expanded)

        # PixelCNN outputs discrete samples
        energy = energy_fn(samples)

        # Reshape for per-beta computation
        log_prob_view = log_prob.view(num_beta, batch_size)
        energy_view = energy.view(num_beta, batch_size)
        beta_expanded = (1.0 / T_expanded).view(num_beta, batch_size)

        self.optimizer.zero_grad()

        # VaTD Gradient Computation:
        # Free Energy: F = E_q[log q(x|T) + β·E(x)]
        #
        # Correct gradient (via Leibniz rule / REINFORCE):
        # ∇_θ F = E_q[(log q + β·E) · ∇_θ log q]
        #
        # Both log q and β·E terms use REINFORCE with (log q + β·E) as weight
        beta_energy = beta_expanded * energy_view

        # Full REINFORCE weight: log q + β·E
        reinforce_weight = log_prob_view.detach() + beta_energy

        # RLOO: Leave-One-Out baseline for variance reduction
        # For each sample, baseline is mean of all OTHER samples
        # Shape: reinforce_weight is (num_beta, batch_size)
        sum_weight = reinforce_weight.sum(dim=1, keepdim=True)  # (num_beta, 1)

        # Leave-one-out baseline: (sum - current_sample) / (N - 1)
        loo_baseline = (sum_weight - reinforce_weight) / (batch_size - 1)

        # Advantage with LOO baseline (must be detached for REINFORCE)
        advantage = (reinforce_weight - loo_baseline).detach()

        # REINFORCE loss: gradient is E[(log q + β·E - baseline) · ∇log q]
        loss = torch.mean(advantage * log_prob_view)
        loss.backward()

        # Log the actual Free Energy (not the surrogate loss)
        train_loss = (log_prob_view + beta_expanded * energy_view).mean().item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Compute statistics for debugging (detect mode collapse)
        with torch.no_grad():
            log_prob_mean = log_prob_view.mean().item()
            log_prob_std = log_prob_view.std().item()
            energy_mean = energy_view.mean().item()
            energy_std = energy_view.std().item()

            # Sample diversity: variance across samples (low = potential collapse)
            # samples shape: (total_size, 1, H, W), values in {-1, 1}
            sample_diversity = samples.var(dim=0).mean().item()

            # Magnetization: mean spin value (|m| close to 1 = ordered/collapsed)
            magnetization = samples.mean(dim=(1, 2, 3))  # per-sample magnetization
            magnetization_mean = magnetization.abs().mean().item()

        stats = {
            "train_loss": train_loss,
            "log_prob_mean": log_prob_mean,
            "log_prob_std": log_prob_std,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "sample_diversity": sample_diversity,
            "magnetization_mean": magnetization_mean,
            "beta_samples_mean": beta_samples.mean().item(),
            "beta_samples_min": beta_samples.min().item(),
            "beta_samples_max": beta_samples.max().item(),
        }

        return stats

    def train_epoch_accumulated(self, energy_fn):
        """
        Training epoch with multiple optimizer steps (refs/vatd style).

        Each step:
        1. Sample from PixelCNN
        2. Compute REINFORCE loss with RLOO baseline
        3. Backward + optimizer.step()

        This is simpler and faster than nested loops.
        accumulation_steps = number of optimizer steps per epoch (like refs/vatd's 19)

        Returns:
            dict: Training statistics
        """
        batch_size = self.model.batch_size
        num_beta = self.model.num_beta
        num_steps = self.accumulation_steps

        # Get curriculum-adjusted beta range
        beta_min, beta_max = self.get_curriculum_beta_range()

        self.model.train()
        # ScheduleFree Optimizer or SPlus support
        if any(
            keyword in self.optimizer.__class__.__name__
            for keyword in ["ScheduleFree", "SPlus"]
        ):
            self.optimizer.train()

        # Epoch-level accumulators for statistics
        epoch_loss = 0.0
        epoch_log_prob = 0.0
        epoch_energy = 0.0
        all_beta_samples = []
        last_samples = None

        # Multiple optimizer steps per epoch (like refs/vatd)
        for step in range(num_steps):
            # Log-uniform sampling over current beta range
            log_beta = torch.rand(num_beta, device=self.device) * (
                math.log(beta_max) - math.log(beta_min)
            ) + math.log(beta_min)
            beta_samples = torch.exp(log_beta)

            all_beta_samples.append(beta_samples.clone())

            T_samples = 1.0 / beta_samples
            T_expanded = T_samples.repeat_interleave(batch_size)
            total_size = num_beta * batch_size

            # Sample (no gradient needed)
            with torch.no_grad():
                samples = self.model.sample(batch_size=total_size, T=T_expanded)
                last_samples = samples

            # Log probability (needs gradient for REINFORCE)
            log_prob = self.model.log_prob(samples, T=T_expanded)

            # Energy (no gradient needed)
            with torch.no_grad():
                energy = energy_fn(samples)

            # Reshape for per-beta computation
            log_prob_view = log_prob.view(num_beta, batch_size)
            energy_view = energy.view(num_beta, batch_size)
            beta_expanded = (1.0 / T_expanded).view(num_beta, batch_size)
            beta_energy = beta_expanded * energy_view

            # REINFORCE weight: log q + beta * E
            reinforce_weight = log_prob_view.detach() + beta_energy

            # RLOO: Leave-One-Out baseline for variance reduction
            sum_weight = reinforce_weight.sum(dim=1, keepdim=True)
            loo_baseline = (sum_weight - reinforce_weight) / (batch_size - 1)
            advantage = (reinforce_weight - loo_baseline).detach()

            # REINFORCE loss
            self.optimizer.zero_grad()
            loss = (advantage * log_prob_view).mean()
            loss.backward()

            # Gradient clipping and optimizer step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Step scheduler for OneCycleLR (needs per-step updates)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            # Accumulate statistics
            with torch.no_grad():
                actual_loss = (log_prob_view + beta_energy).mean().item()
                epoch_loss += actual_loss / num_steps
                epoch_log_prob += log_prob_view.mean().item() / num_steps
                epoch_energy += energy_view.mean().item() / num_steps

        # Compute final statistics
        all_betas = torch.cat(all_beta_samples)

        with torch.no_grad():
            if last_samples is not None:
                sample_diversity = last_samples.var(dim=0).mean().item()
                magnetization = last_samples.mean(dim=(1, 2, 3)).abs().mean().item()
            else:
                sample_diversity = 0.0
                magnetization = 0.0

        stats = {
            "train_loss": epoch_loss,
            "log_prob_mean": epoch_log_prob,
            "log_prob_std": 0.0,
            "energy_mean": epoch_energy,
            "energy_std": 0.0,
            "sample_diversity": sample_diversity,
            "magnetization_mean": magnetization,
            "beta_samples_mean": all_betas.mean().item(),
            "beta_samples_min": all_betas.min().item(),
            "beta_samples_max": all_betas.max().item(),
            "optimizer_steps_per_epoch": num_steps,
        }

        return stats

    def train_epoch_sequential(self, energy_fn):
        """
        Hybrid training: Parallel sampling + Sequential backward per temperature.
        Now supports MCMC Correction (Kalman Filter style) if enabled.

        This matches refs/vatd behavior where each temperature gets its own
        forward/backward pass, preventing crosstalk between temperature groups.

        Key insight: Sampling (255 autoregressive steps) is expensive,
        but log_prob computation is relatively cheap.

        Algorithm:
        1. Sample ALL temperatures in parallel (fast - one pass)
        2. For each temperature separately:
           - Compute log_prob (forward pass with single T)
           - Compute RLOO baseline within this T group
           - Backward (gradients accumulate)
        3. (Optional) MCMC Correction:
           - Every K steps, run Swendsen-Wang on current samples
           - Compute MLE loss on these MCMC samples
           - Backward (Supervised Learning signal)
        4. optimizer.step() after all T processed

        Returns:
            dict: Training statistics
        """
        batch_size = self.model.batch_size
        num_beta = self.model.num_beta
        num_steps = self.accumulation_steps

        # Get curriculum-adjusted beta range
        beta_min, beta_max = self.get_curriculum_beta_range()

        self.model.train()
        if any(
            keyword in self.optimizer.__class__.__name__
            for keyword in ["ScheduleFree", "SPlus"]
        ):
            self.optimizer.train()

        # Epoch-level accumulators
        epoch_loss = 0.0
        epoch_log_prob = 0.0
        epoch_energy = 0.0
        mcmc_loss_acc = 0.0
        all_beta_samples = []
        last_samples = None

        for step in range(num_steps):
            # Log-uniform sampling over current beta range
            log_beta = torch.rand(num_beta, device=self.device) * (
                math.log(beta_max) - math.log(beta_min)
            ) + math.log(beta_min)
            beta_samples = torch.exp(log_beta)
            all_beta_samples.append(beta_samples.clone())

            T_samples = 1.0 / beta_samples
            T_expanded = T_samples.repeat_interleave(batch_size)
            total_size = num_beta * batch_size

            # Step 1: Parallel sampling (expensive part - done once)
            with torch.no_grad():
                samples = self.model.sample(batch_size=total_size, T=T_expanded)
                last_samples = samples
                # Pre-compute energy for all samples
                energy_all = energy_fn(samples)

            # Step 2: Sequential backward per temperature
            self.optimizer.zero_grad()
            step_loss = 0.0
            step_log_prob = 0.0
            step_energy = 0.0

            for i in range(num_beta):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size

                # Extract samples and energy for this temperature
                samples_i = samples[start_idx:end_idx]
                energy_i = energy_all[start_idx:end_idx]
                T_i = T_samples[i].expand(batch_size)
                beta_i = beta_samples[i]

                # Forward pass with single temperature (no crosstalk)
                log_prob_i = self.model.log_prob(samples_i, T=T_i)

                # REINFORCE weight: log q + beta * E
                beta_energy_i = beta_i * energy_i
                reinforce_weight_i = log_prob_i.detach() + beta_energy_i

                # RLOO baseline within this temperature group
                sum_weight = reinforce_weight_i.sum()
                loo_baseline_i = (sum_weight - reinforce_weight_i) / (batch_size - 1)
                advantage_i = (reinforce_weight_i - loo_baseline_i).detach()

                # REINFORCE loss for this temperature
                loss_i = (advantage_i * log_prob_i).mean()
                loss_i.backward()  # Gradients accumulate

                # Accumulate statistics
                with torch.no_grad():
                    step_loss += (log_prob_i + beta_energy_i).mean().item() / num_beta
                    step_log_prob += log_prob_i.mean().item() / num_beta
                    step_energy += energy_i.mean().item() / num_beta

            # Step 3: MCMC Correction (Kalman Filter Update)
            # Only run periodically
            if self.mcmc_enabled and self.current_epoch % self.mcmc_freq == 0:
                with torch.no_grad():
                    # Run Swendsen-Wang to equilibrate samples
                    # Start from current model samples (warm start)
                    mcmc_samples = swendsen_wang_update(
                        samples, T_expanded, n_sweeps=self.mcmc_steps, fix_first=(self.model.fix_first is not None)
                    )

                # Teacher Forcing: Maximize likelihood of MCMC samples
                # Treat MCMC samples as "ground truth"
                log_prob_mcmc = self.model.log_prob(mcmc_samples, T=T_expanded)
                loss_mle = -log_prob_mcmc.mean()

                # Backward MLE loss with weight
                (loss_mle * self.mcmc_weight).backward()

                mcmc_loss_acc += loss_mle.item() / num_steps

            # Gradient clipping and optimizer step (after all T processed)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Step scheduler for OneCycleLR (needs per-step updates)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            # Accumulate epoch statistics
            epoch_loss += step_loss / num_steps
            epoch_log_prob += step_log_prob / num_steps
            epoch_energy += step_energy / num_steps

        # Compute final statistics
        all_betas = torch.cat(all_beta_samples)

        with torch.no_grad():
            if last_samples is not None:
                sample_diversity = last_samples.var(dim=0).mean().item()
                magnetization = last_samples.mean(dim=(1, 2, 3)).abs().mean().item()
            else:
                sample_diversity = 0.0
                magnetization = 0.0

        stats = {
            "train_loss": epoch_loss,
            "log_prob_mean": epoch_log_prob,
            "log_prob_std": 0.0,
            "energy_mean": epoch_energy,
            "energy_std": 0.0,
            "sample_diversity": sample_diversity,
            "magnetization_mean": magnetization,
            "beta_samples_mean": all_betas.mean().item(),
            "beta_samples_min": all_betas.min().item(),
            "beta_samples_max": all_betas.max().item(),
            "optimizer_steps_per_epoch": num_steps,
        }
        
        if self.mcmc_enabled:
             stats["mcmc_loss"] = mcmc_loss_acc

        return stats

    def val_epoch(self, energy_fn):
        """
        Validation epoch with fixed beta values.

        Returns:
            dict: Contains 'val_loss' (mean) and individual beta losses
                  {
                      'val_loss': float,
                      'val_loss_beta_0': float,
                      'val_loss_beta_1': float,
                      ...
                      'val_loss_beta_7': float
                  }
        """
        batch_size = self.model.batch_size
        # Use validation beta count (may differ from training num_beta)
        num_beta = len(self.fixed_val_betas)

        # Use fixed betas instead of random sampling
        beta_samples = self.fixed_val_betas  # shape: (num_beta,)
        T_samples = 1.0 / beta_samples
        T_expanded = T_samples.repeat_interleave(batch_size)
        total_size = num_beta * batch_size

        self.model.eval()
        # ScheduleFree Optimizer or SPlus
        if any(
            keyword in self.optimizer.__class__.__name__
            for keyword in ["ScheduleFree", "SPlus"]
        ):
            self.optimizer.eval()

        with torch.no_grad():
            log_prob, samples = self.step(batch_size=total_size, T=T_expanded)

            # PixelCNN outputs discrete samples
            energy = energy_fn(samples)
            beta = (1.0 / T_expanded).unsqueeze(-1)  # (total_size, 1)
            loss_raw = log_prob + beta * energy

            # Reshape to separate beta dimensions: (num_beta, batch_size)
            loss_view = loss_raw.view(num_beta, batch_size)

            # Compute mean loss per beta
            loss_per_beta = loss_view.mean(dim=1)  # shape: (num_beta,)

            # Compute overall mean
            val_loss = loss_per_beta.mean().item()

            # Create result dictionary with individual beta losses
            val_dict = {"val_loss": val_loss}
            for i in range(num_beta):
                val_dict[f"val_loss_beta_{i}"] = loss_per_beta[i].item()

            # Compute exact errors if available
            if self.exact_logz_values is not None:
                for i in range(num_beta):
                    # Exact log partition function (negative of exact loss)
                    exact_logz = self.exact_logz_values[i]

                    # Model loss approximates: -log Z
                    # Error = model_loss - (-exact_logz) = model_loss + exact_logz
                    model_loss = val_dict[f"val_loss_beta_{i}"]
                    error_vs_exact = model_loss + exact_logz
                    val_dict[f"val_error_exact_beta_{i}"] = error_vs_exact
                    val_dict[f"val_exact_logz_beta_{i}"] = exact_logz

        return val_dict

    def train(self, energy_fn, epochs):
        val_loss = 0
        val_losses = []
        initial_train_loss = None
        plateau_counter = 0
        last_val_loss = None

        # Log curriculum settings at start
        if self.curriculum_enabled:
            tqdm.write(f"[Curriculum] 2-Phase enabled:")
            tqdm.write(f"  Phase 1: epochs 0-{self.phase1_epochs}, beta_max={self.phase1_beta_max:.3f} (high temp)")
            tqdm.write(f"  Phase 2: epochs {self.phase1_epochs}+, gradual expansion to full range")

        # Log training mode at start
        if self.training_mode == "accumulated":
            tqdm.write(f"[v0.11] Accumulated training mode enabled:")
            tqdm.write(f"  optimizer_steps_per_epoch={self.accumulation_steps}")
            tqdm.write(f"  total_optimizer_steps={self.accumulation_steps * self.epochs}")
            tqdm.write(f"  baseline=RLOO")
        elif self.training_mode == "sequential":
            tqdm.write(f"[v0.13] Sequential training mode enabled:")
            tqdm.write(f"  Parallel sampling + Sequential backward per temperature")
            tqdm.write(f"  optimizer_steps_per_epoch={self.accumulation_steps}")
            tqdm.write(f"  baseline=RLOO (per temperature)")

        for epoch in tqdm(range(epochs), desc="Overall Progress"):
            # Update current epoch for curriculum learning
            self.current_epoch = epoch

            # Select training method based on training_mode
            if self.training_mode == "sequential":
                # v0.13: Parallel sampling + Sequential backward per T
                train_stats = self.train_epoch_sequential(energy_fn)
            elif self.training_mode == "accumulated":
                train_stats = self.train_epoch_accumulated(energy_fn)
            else:
                # Standard REINFORCE training (backward compatible)
                train_stats = self.train_epoch(energy_fn)
            train_loss = train_stats["train_loss"]

            val_dict = self.val_epoch(energy_fn)
            val_loss = val_dict["val_loss"]
            val_losses.append(val_loss)

            # Track initial train_loss for divergence detection
            if initial_train_loss is None:
                initial_train_loss = train_loss

            # Early stopping if loss becomes NaN or Inf
            if math.isnan(train_loss) or math.isnan(val_loss):
                tqdm.write("Early stopping due to NaN loss")
                val_loss = math.inf
                if self.trial is not None:
                    raise optuna.TrialPruned()
                break

            if math.isinf(train_loss) or math.isinf(val_loss):
                tqdm.write("Early stopping due to Inf loss")
                val_loss = math.inf
                if self.trial is not None:
                    raise optuna.TrialPruned()
                break

            # Divergence detection: train_loss sign changed (negative -> positive)
            if initial_train_loss < 0 and train_loss > 0:
                tqdm.write(f"Divergence detected: train_loss sign flipped ({train_loss:.2f})")
                val_loss = math.inf
                if self.trial is not None:
                    raise optuna.TrialPruned()
                break

            # Plateau detection using relative error (for at least 10 epochs)
            if last_val_loss is not None and epoch >= 10:
                rel_change = abs(val_loss - last_val_loss) / (abs(last_val_loss) + 1e-8)
                if rel_change < 1e-5:
                    plateau_counter += 1
                else:
                    plateau_counter = 0

                if plateau_counter >= 10:
                    tqdm.write(f"Plateau detected for {plateau_counter} epochs")
                    if self.trial is not None:
                        raise optuna.TrialPruned()
                    break

            last_val_loss = val_loss

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    tqdm.write(f"Early stopping triggered at epoch {epoch}")
                    break

            # Get current curriculum beta range and phase for logging
            curr_beta_min, curr_beta_max = self.get_curriculum_beta_range()
            curr_phase = self.get_curriculum_phase() if self.curriculum_enabled else 0

            # Use validation beta count (may differ from training num_beta)
            val_num_beta = len(self.fixed_val_betas)

            # ========== Organized wandb logging by section ==========
            log_dict = {}

            # --- train/ section ---
            log_dict["train/loss"] = train_loss
            log_dict["train/loss_signlog"] = sign_log_transform(train_loss)
            log_dict["train/log_prob_mean"] = train_stats["log_prob_mean"]
            log_dict["train/log_prob_std"] = train_stats["log_prob_std"]
            log_dict["train/energy_mean"] = train_stats["energy_mean"]
            log_dict["train/energy_std"] = train_stats["energy_std"]
            log_dict["train/lr"] = self.optimizer.param_groups[0]["lr"]

            # --- val/ section ---
            log_dict["val/loss"] = val_loss
            log_dict["val/loss_signlog"] = sign_log_transform(val_loss)
            for i in range(val_num_beta):
                log_dict[f"val/loss_beta_{i}"] = val_dict[f"val_loss_beta_{i}"]
                log_dict[f"val/loss_beta_{i}_signlog"] = sign_log_transform(
                    val_dict[f"val_loss_beta_{i}"]
                )
                log_dict[f"val/beta_{i}"] = self.fixed_val_betas[i].item()

            # --- curriculum/ section ---
            log_dict["curriculum/phase"] = curr_phase
            log_dict["curriculum/beta_min"] = curr_beta_min
            log_dict["curriculum/beta_max"] = curr_beta_max
            log_dict["curriculum/T_min"] = 1.0 / curr_beta_max
            log_dict["curriculum/T_max"] = 1.0 / curr_beta_min

            # --- diversity/ section ---
            log_dict["diversity/sample_diversity"] = train_stats["sample_diversity"]
            log_dict["diversity/magnetization_mean"] = train_stats["magnetization_mean"]

            # --- sampling/ section ---
            log_dict["sampling/beta_samples_mean"] = train_stats["beta_samples_mean"]
            log_dict["sampling/beta_samples_min"] = train_stats["beta_samples_min"]
            log_dict["sampling/beta_samples_max"] = train_stats["beta_samples_max"]

            # --- exact/ section (exact error metrics) ---
            if self.exact_logz_values is not None:
                for i in range(val_num_beta):
                    if f"val_error_exact_beta_{i}" in val_dict:
                        log_dict[f"exact/error_beta_{i}"] = val_dict[f"val_error_exact_beta_{i}"]
                        log_dict[f"exact/error_beta_{i}_signlog"] = sign_log_transform(
                            val_dict[f"val_error_exact_beta_{i}"]
                        )
                    if f"val_exact_logz_beta_{i}" in val_dict:
                        log_dict[f"exact/logz_beta_{i}"] = val_dict[f"val_exact_logz_beta_{i}"]

                # Mean absolute error across all betas
                errors = [val_dict[f"val_error_exact_beta_{i}"] for i in range(val_num_beta)]
                log_dict["exact/error_mean"] = sum(errors) / len(errors)
                log_dict["exact/error_abs_mean"] = sum(abs(e) for e in errors) / len(errors)
                log_dict["exact/error_abs_mean_signlog"] = sign_log_transform(
                    log_dict["exact/error_abs_mean"]
                )

            # --- compare/ section: val_loss vs exact comparison ---
            # val_loss ≈ -ln(Z), so compare val_loss_signlog with sign_log_transform(-exact_logz)
            if self.exact_logz_values is not None:
                # Group 1: beta 0-3 (low beta / high temperature)
                for i in range(min(4, val_num_beta)):
                    exact_logz = val_dict.get(f"val_exact_logz_beta_{i}", None)
                    if exact_logz is not None:
                        # exact: -ln(Z) = -exact_logz, apply sign_log_transform
                        log_dict[f"compare_0_3/exact_beta_{i}"] = sign_log_transform(-exact_logz)
                        log_dict[f"compare_0_3/val_beta_{i}"] = sign_log_transform(
                            val_dict[f"val_loss_beta_{i}"]
                        )

                # Group 2: beta 4-7 (high beta / low temperature)
                for i in range(4, min(8, val_num_beta)):
                    exact_logz = val_dict.get(f"val_exact_logz_beta_{i}", None)
                    if exact_logz is not None:
                        log_dict[f"compare_4_7/exact_beta_{i}"] = sign_log_transform(-exact_logz)
                        log_dict[f"compare_4_7/val_beta_{i}"] = sign_log_transform(
                            val_dict[f"val_loss_beta_{i}"]
                        )

            if epoch >= 10:
                log_dict["predicted_final_loss"] = predict_final_loss(
                    val_losses, epochs
                )

            # Pruning check
            if (
                self.pruner is not None
                and self.trial is not None
                and self.seed is not None
            ):
                self.pruner.report(
                    trial_id=self.trial.number,
                    seed=self.seed,
                    epoch=epoch,
                    value=val_loss,
                )
                if self.pruner.should_prune():
                    raise optuna.TrialPruned()

            # Scheduler stepping:
            # - ReduceLROnPlateau: requires metrics, step per epoch
            # - OneCycleLR: stepped per optimizer step (inside train_epoch_sequential)
            # - Others: step per epoch
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                pass  # Already stepped inside train_epoch_sequential
            else:
                self.scheduler.step()

            # Log sample visualization periodically (every 10 epochs or at start/end)
            if epoch % 10 == 0 or epoch == epochs - 1:
                try:
                    # Use a subset of validation betas for visualization
                    vis_betas = self.fixed_val_betas[::2]  # Every other beta
                    fig = create_sample_grid(
                        self.model, vis_betas, n_samples=4, device=self.device
                    )
                    log_dict["samples"] = wandb.Image(fig)
                    plt.close(fig)
                except Exception as e:
                    tqdm.write(f"Sample visualization failed: {e}")

            wandb.log(log_dict)
            if epoch % 10 == 0 or epoch == epochs - 1:
                lr = self.optimizer.param_groups[0]["lr"]
                tqdm.write(
                    f"epoch: {epoch}, train_loss: {train_loss:.4e}, val_loss: {val_loss:.4e}, lr: {lr:.4e}"
                )

        return val_loss


def run(run_config: RunConfig, energy_fn, group_name=None, trial=None, pruner=None):
    project = run_config.project
    device = run_config.device
    seeds = run_config.seeds
    if not group_name:
        group_name = run_config.gen_group_name()
    tags = run_config.gen_tags()

    group_path = f"runs/{run_config.project}/{group_name}"
    if not os.path.exists(group_path):
        os.makedirs(group_path)
    run_config.to_yaml(f"{group_path}/config.yaml")

    # Register trial at the beginning if pruner exists
    if pruner is not None and trial is not None and hasattr(pruner, "register_trial"):
        pruner.register_trial(trial.number)

    total_loss = 0
    complete_seeds = 0
    try:
        for seed in seeds:
            set_seed(seed)

            model = run_config.create_model().to(device)
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

            # Extract v0.11 training parameters from net_config
            net_config = run_config.gen_config().get("net_config", {})
            accumulation_steps = net_config.get("accumulation_steps", 1)
            training_mode = net_config.get("training_mode", "standard")

            trainer = Trainer(
                model,
                optimizer,
                scheduler,
                early_stopping_config=run_config.early_stopping_config,
                device=device,
                trial=trial,
                seed=seed,
                pruner=pruner,
                energy_fn=energy_fn,
                epochs=run_config.epochs,
                # v0.11 parameters
                accumulation_steps=accumulation_steps,
                training_mode=training_mode,
            )

            val_loss = trainer.train(energy_fn, epochs=run_config.epochs)
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

    except optuna.TrialPruned:
        wandb.finish()
        raise
    except Exception as e:
        tqdm.write(f"Runtime error during training: {e}")
        wandb.finish()
        raise optuna.TrialPruned()
    finally:
        # Call trial_finished only once after all seeds are done
        if (
            pruner is not None
            and trial is not None
            and hasattr(pruner, "complete_trial")
        ):
            pruner.complete_trial(trial.number)

    return total_loss / (complete_seeds if complete_seeds > 0 else 1)


# ┌──────────────────────────────────────────────────────────┐
#  For Analyze
# └──────────────────────────────────────────────────────────┘
def select_project():
    runs_path = "runs/"
    projects = [
        d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))
    ]
    projects.sort()
    if not projects:
        raise ValueError(f"No projects found in {runs_path}")

    selected_project = beaupy.select(projects)
    return selected_project


def select_group(project):
    runs_path = f"runs/{project}"
    groups = [
        d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))
    ]
    groups.sort()
    if not groups:
        raise ValueError(f"No run groups found in {runs_path}")

    selected_group = beaupy.select(groups)
    return selected_group


def select_seed(project, group_name):
    group_path = f"runs/{project}/{group_name}"
    seeds = [
        d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))
    ]
    seeds.sort()
    if not seeds:
        raise ValueError(f"No seeds found in {group_path}")

    selected_seed = beaupy.select(seeds)
    return selected_seed


def select_device():
    devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    selected_device = beaupy.select(devices)
    return selected_device


def load_model(project, group_name, seed, weights_only=True):
    """
    Load a trained model and its configuration.

    Args:
        project (str): The name of the project.
        group_name (str): The name of the run group.
        seed (str): The seed of the specific run.
        weights_only (bool, optional): If True, only load the model weights without loading the entire pickle file.
                                       This can be faster and use less memory. Defaults to True.

    Returns:
        tuple: A tuple containing the loaded model and its configuration.

    Raises:
        FileNotFoundError: If the config or model file is not found.

    Example usage:
        # Load full model
        model, config = load_model("MyProject", "experiment1", "seed42")

        # Load only weights (faster and uses less memory)
        model, config = load_model("MyProject", "experiment1", "seed42", weights_only=True)
    """
    config_path = f"runs/{project}/{group_name}/config.yaml"
    model_path = f"runs/{project}/{group_name}/{seed}/model.pt"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found for {project}/{group_name}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found for {project}/{group_name}/{seed}"
        )

    config = RunConfig.from_yaml(config_path)
    model = config.create_model()

    # Use weights_only option in torch.load
    state_dict = torch.load(model_path, map_location="cpu", weights_only=weights_only)
    model.load_state_dict(state_dict)

    return model, config


def load_study(project, study_name):
    """
    Load the best study from an optimization run.

    Args:
        project (str): The name of the project.
        study_name (str): The name of the study.

    Returns:
        optuna.Study: The loaded study object.
    """
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{project}.db")
    return study


def load_best_model(project, study_name, weights_only=True):
    """
    Load the best model and its configuration from an optimization study.

    Args:
        project (str): The name of the project.
        study_name (str): The name of the study.

    Returns:
        tuple: A tuple containing the loaded model, its configuration, and the best trial number.
    """
    study = load_study(project, study_name)
    best_trial = study.best_trial
    project_name = project
    group_name = best_trial.user_attrs["group_name"]

    # Select Seed
    seed = select_seed(project_name, group_name)
    best_model, best_config = load_model(
        project_name, group_name, seed, weights_only=weights_only
    )

    return best_model, best_config
