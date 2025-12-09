import torch
import torch.nn.functional as F
import numpy as np
import beaupy
from rich.console import Console
import wandb
import optuna
from tqdm import tqdm

from config import RunConfig, GumbelConfig

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
        gumbel_config=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.trial = trial
        self.seed = seed
        self.pruner = pruner
        self.gumbel_config = gumbel_config if gumbel_config else GumbelConfig()

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

    def step(self, batch_size, T):
        # Sample from PixelCNN (no gradient tracking needed for sampling)
        with torch.no_grad():
            samples = self.model.sample(batch_size=batch_size, T=T)
        log_prob = self.model.log_prob(samples, T=T)
        return log_prob, samples

    def train_epoch(self, energy_fn):
        batch_size = self.model.batch_size
        beta_min = self.model.beta_min
        beta_max = self.model.beta_max
        num_beta = self.model.num_beta
        # Log-uniform sampling for beta (matches validation's log-spaced distribution)
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

        # Separate gradient computation:
        # Free Energy: F = E_q[log q(x|T) + β·E(x)]
        #
        # Part 1: log q term - direct gradient (differentiable)
        # ∇_θ E_q[log q] can be computed directly since q depends on θ
        log_prob_loss = torch.mean(log_prob_view)

        # Part 2: β·E term - REINFORCE (score function estimator)
        # ∇_θ E_q[β·E] = E_q[∇_θ log q · β·E] since E is independent of θ
        beta_energy = beta_expanded * energy_view

        # Baseline for variance reduction (mean per beta)
        baseline_energy = beta_energy.mean(dim=1, keepdim=True).detach()

        # Advantage MUST be detached to act as fixed reward signal
        advantage = (beta_energy - baseline_energy).detach()

        # REINFORCE gradient for energy term
        energy_reinforce_loss = torch.mean(advantage * log_prob_view)

        # Total loss (combines both gradient estimates)
        total_loss = log_prob_loss + energy_reinforce_loss
        total_loss.backward()

        # Log the actual Free Energy (not the surrogate loss)
        train_loss = (log_prob_view + beta_expanded * energy_view).mean().item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return train_loss

    def train_epoch_gumbel(self, energy_fn, temperature=1.0, hard=True):
        """
        Training epoch using Gumbel-Softmax for gradient estimation.

        This method uses a hybrid approach:
        1. Sample discrete configurations using regular PixelCNN sampling
        2. Compute logits for these samples
        3. Apply Gumbel-Softmax to create soft reconstructions with gradients
        4. Use soft samples for energy computation (enables backprop)

        This combines the benefits of:
        - Discrete sampling for valid configurations
        - Soft relaxation for gradient flow

        Args:
            energy_fn: Energy function to evaluate samples
            temperature (float): Gumbel-Softmax temperature (tau)
                - Recommended: anneal from 1.0 to 0.1
                - Lower → closer to discrete, higher → more uniform
            hard (bool): Use straight-through estimator
                - True: discrete forward, soft backward (recommended)
                - False: fully continuous (more biased)

        Returns:
            float: Training loss (Free Energy)
        """
        batch_size = self.model.batch_size
        beta_min = self.model.beta_min
        beta_max = self.model.beta_max
        num_beta = self.model.num_beta

        # Log-uniform sampling for beta
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

        # Step 1: Sample discrete configurations
        with torch.no_grad():
            samples_discrete = self.model.sample(batch_size=total_size, T=T_expanded)

        # Compute log_prob with gradient for entropy term (REINFORCE)
        log_prob = self.model.log_prob(samples_discrete, T=T_expanded)

        # Step 2: Get logits for the discrete samples (with gradient tracking)
        # Convert back to {0,1} space for model input
        samples_01 = self.model.reverse_mapping(samples_discrete)

        # Add temperature conditioning if needed
        if T_expanded is not None:
            T_cond = T_expanded.to(self.device)
            if T_cond.dim() == 1:
                T_cond = T_cond.unsqueeze(1)
            import einops
            T_spatial = einops.repeat(
                T_cond, "b c -> b c h w",
                h=samples_01.shape[2], w=samples_01.shape[3]
            )
            samples_input = torch.cat([samples_01, T_spatial], dim=1)
        else:
            samples_input = samples_01

        # Get logits with gradients
        logits = self.model.masked_conv.forward(samples_input)  # (B, Cat, C, H, W)

        # Step 3: Apply Gumbel-Softmax to logits
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(logits) + 1e-20
        ) + 1e-20)

        # Soft probabilities with temperature
        soft_probs = F.softmax((logits + gumbel_noise) / temperature, dim=1)

        if hard:
            # Straight-through estimator
            # Forward: use discrete samples
            # Backward: gradients from soft_probs
            hard_samples = F.one_hot(
                samples_01.long(), num_classes=self.model.category
            ).permute(0, 4, 1, 2, 3).float()  # (B, Cat, C, H, W)

            # Straight-through: detach soft, add back with gradient
            soft_samples = hard_samples - soft_probs.detach() + soft_probs
        else:
            # Pure soft samples
            soft_samples = soft_probs

        # Step 4: Convert soft samples to scalar values for energy computation
        # For binary case: value = 0 * p(0) + 1 * p(1) = p(1)
        if hard:
            # Use discrete samples but with gradient connection to soft_probs
            samples_01_soft = samples_01 - samples_01.detach() + soft_samples[:, 1, :, :, :]
        else:
            # Weighted average
            samples_01_soft = soft_samples[:, 1, :, :, :]

        # Convert to {-1, +1} space
        samples_soft = self.model.mapping(samples_01_soft)

        # Step 5: Compute energy with gradient flow
        energy = energy_fn(samples_soft)

        # Reshape for per-beta computation
        log_prob_view = log_prob.view(num_beta, batch_size)
        energy_view = energy.view(num_beta, batch_size)
        beta_expanded_view = (1.0 / T_expanded).view(num_beta, batch_size)

        self.optimizer.zero_grad()

        # Hybrid gradient computation:
        # Free Energy: F = E_q[log q(x) + β·E(x)]
        #
        # Part 1: Entropy term - REINFORCE gradient
        # ∇_θ E_q[log q(x)] computed via score function with gradient
        log_prob_loss = torch.mean(log_prob_view)

        # Part 2: Energy term - Gumbel-Softmax gradient (lower variance)
        # Direct gradient through soft samples
        beta_energy = beta_expanded_view * energy_view
        energy_loss = torch.mean(beta_energy)

        # Combined loss
        # Note: Both terms contribute gradients now, preventing vanishing gradients at low beta
        total_loss = log_prob_loss + energy_loss
        total_loss.backward()

        # Log the actual Free Energy
        train_loss = (log_prob_view + beta_expanded_view * energy_view).mean().item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return train_loss

    def train_epoch_flow(self, energy_fn):
        """
        Training epoch for flow models.

        Flow models support reparameterized sampling, so REINFORCE is not needed.
        Gradients flow directly through sampling via the reparameterization trick.

        Args:
            energy_fn: Energy function (samples -> energy)

        Returns:
            Average loss value
        """
        batch_size = self.model.batch_size
        beta_min = self.model.beta_min
        beta_max = self.model.beta_max
        num_beta = self.model.num_beta

        # Log-uniform beta sampling
        log_beta_samples = torch.rand(num_beta, device=self.device) * (
            math.log(beta_max) - math.log(beta_min)
        ) + math.log(beta_min)
        beta_samples = torch.exp(log_beta_samples)
        T_samples = 1.0 / beta_samples
        T_expanded = T_samples.repeat_interleave(batch_size)
        total_size = num_beta * batch_size

        self.model.train()

        # Generate continuous samples (enables gradient flow)
        samples_continuous = self.model.sample_continuous(
            batch_size=total_size,
            T=T_expanded
        )

        # DEBUG: Check continuous samples
        if torch.any(samples_continuous.abs() > 10.0):
            print(f"WARNING: Extreme continuous samples! Max: {samples_continuous.abs().max().item():.2e}")
            print(f"  Range: [{samples_continuous.min().item():.2e}, {samples_continuous.max().item():.2e}]")

        # Discrete samples (for energy computation validation)
        samples_discrete = self.model.dequantizer.quantize(samples_continuous)

        # Compute log probability
        log_prob = self.model.log_prob(samples_continuous, T=T_expanded)

        # DEBUG: Check log_prob
        if torch.any(log_prob.abs() > 1000.0):
            print(f"WARNING: Extreme log_prob! Range: [{log_prob.min().item():.2e}, {log_prob.max().item():.2e}]")

        # Compute energy on continuous samples for gradient flow
        # Ising energy E = (s^T A s)/2 is bilinear, so it works with continuous values
        # This allows gradients to flow through the flow model
        energy = energy_fn(samples_continuous)

        beta = (1.0 / T_expanded).view(-1, 1)

        # DEBUG: Check energies
        if torch.any(energy.abs() > 1000.0):
            print(f"WARNING: Extreme energy! Range: [{energy.min().item():.2e}, {energy.max().item():.2e}]")

        # Free energy loss: F = E_p[log p(x) + β·E(x)]
        # This minimizes the variational free energy
        loss = (log_prob + beta * energy).mean()

        # DEBUG: Check final loss
        if torch.isnan(loss) or torch.isinf(loss) or loss.abs() > 1e10:
            print(f"CRITICAL: Loss explosion detected! loss = {loss.item():.2e}")
            print(f"  log_prob: mean={log_prob.mean().item():.2e}, std={log_prob.std().item():.2e}")
            print(f"  beta: mean={beta.mean().item():.2e}, max={beta.max().item():.2e}")
            print(f"  energy_for_grad: mean={energy_for_grad.mean().item():.2e}, max={energy_for_grad.max().item():.2e}")
            print(f"  beta * energy: mean={(beta * energy_for_grad).mean().item():.2e}, max={(beta * energy_for_grad).max().item():.2e}")

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def train_epoch_discrete_flow(self, energy_fn):
        """
        Training epoch for discrete flow models using Straight-Through Estimator.

        Unlike continuous flows, discrete flows use STE to enable gradient flow
        through discrete sampling and energy computation.

        Args:
            energy_fn: Energy function (samples -> energy)

        Returns:
            Average loss value
        """
        batch_size = self.model.batch_size
        beta_min = self.model.beta_min
        beta_max = self.model.beta_max
        num_beta = self.model.num_beta

        # Log-uniform beta sampling (same as continuous flow)
        log_beta_samples = torch.rand(num_beta, device=self.device) * (
            math.log(beta_max) - math.log(beta_min)
        ) + math.log(beta_min)
        beta_samples = torch.exp(log_beta_samples)
        T_samples = 1.0 / beta_samples
        T_expanded = T_samples.repeat_interleave(batch_size)
        total_size = num_beta * batch_size

        self.model.train()

        # Generate discrete samples with gradient tracking (via STE)
        # Note: sample() internally uses STE in BinaryCouplingLayer
        samples_discrete = self.model.sample(
            batch_size=total_size,
            T=T_expanded
        )  # (total_size, 1, H, W) in {-1, +1}

        # Compute log probability with gradients
        log_prob = self.model.log_prob(samples_discrete, T=T_expanded)

        # Compute energy on discrete samples
        # STE allows gradients to flow back through the discrete samples
        energy = energy_fn(samples_discrete)

        beta = (1.0 / T_expanded).view(-1, 1)

        # Free energy loss: F = E_p[log p(x) + β·E(x)]
        # Both terms have gradients thanks to STE
        loss = (log_prob + beta * energy).mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

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

        for epoch in tqdm(range(epochs), desc="Overall Progress"):
            # Detect model type
            is_discrete_flow = 'DiscreteFlow' in self.model.__class__.__name__
            is_continuous_flow = 'CheckerboardFlow' in self.model.__class__.__name__

            # Choose training method based on model type and configuration
            if is_discrete_flow:
                train_loss = self.train_epoch_discrete_flow(energy_fn)
            elif is_continuous_flow:
                train_loss = self.train_epoch_flow(energy_fn)
            elif self.gumbel_config.use_gumbel:
                # Gumbel-Softmax with temperature annealing
                temperature = self.gumbel_config.get_temperature(epoch, epochs)
                train_loss = self.train_epoch_gumbel(
                    energy_fn,
                    temperature=temperature,
                    hard=self.gumbel_config.hard
                )
            else:
                # Standard REINFORCE
                train_loss = self.train_epoch(energy_fn)

            val_dict = self.val_epoch(energy_fn)
            val_loss = val_dict["val_loss"]
            val_losses.append(val_loss)

            # Early stopping if loss becomes NaN
            if math.isnan(train_loss) or math.isnan(val_loss):
                tqdm.write("Early stopping due to NaN loss")
                val_loss = math.inf
                break

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    tqdm.write(f"Early stopping triggered at epoch {epoch}")
                    break

            log_dict = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            # Add Gumbel-Softmax temperature if using Gumbel
            if self.gumbel_config.use_gumbel:
                temperature = self.gumbel_config.get_temperature(epoch, epochs)
                log_dict["gumbel_temperature"] = temperature

            # Add individual beta losses and beta values
            # Use validation beta count (may differ from training num_beta)
            val_num_beta = len(self.fixed_val_betas)
            for i in range(val_num_beta):
                log_dict[f"val_loss_beta_{i}"] = val_dict[f"val_loss_beta_{i}"]
                log_dict[f"val_beta_{i}"] = self.fixed_val_betas[i].item()

            # Add sign-log transforms
            log_dict["train_loss_signlog"] = sign_log_transform(train_loss)
            log_dict["val_loss_signlog"] = sign_log_transform(val_loss)
            for i in range(val_num_beta):
                loss_val = val_dict[f"val_loss_beta_{i}"]
                log_dict[f"val_loss_beta_{i}_signlog"] = sign_log_transform(loss_val)

            # Add exact error metrics
            if self.exact_logz_values is not None:
                for i in range(val_num_beta):
                    if f"val_error_exact_beta_{i}" in val_dict:
                        log_dict[f"val_error_exact_beta_{i}"] = val_dict[
                            f"val_error_exact_beta_{i}"
                        ]
                        log_dict[f"val_error_exact_beta_{i}_signlog"] = (
                            sign_log_transform(val_dict[f"val_error_exact_beta_{i}"])
                        )
                    if f"val_exact_logz_beta_{i}" in val_dict:
                        log_dict[f"val_exact_logz_beta_{i}"] = val_dict[
                            f"val_exact_logz_beta_{i}"
                        ]

                # Add mean absolute error across all betas
                errors = [
                    val_dict[f"val_error_exact_beta_{i}"] for i in range(val_num_beta)
                ]
                log_dict["val_error_exact_mean"] = sum(errors) / len(errors)
                log_dict["val_error_exact_abs_mean"] = sum(
                    abs(e) for e in errors
                ) / len(errors)
                log_dict["val_error_exact_abs_mean_signlog"] = sign_log_transform(
                    log_dict["val_error_exact_abs_mean"]
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

            # ReduceLROnPlateau requires metrics, other schedulers don't
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
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
                gumbel_config=run_config.gumbel_config,
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
