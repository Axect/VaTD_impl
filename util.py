import torch
import torch.nn.functional as F
import numpy as np
import beaupy
from rich.console import Console
import wandb
import optuna

from config import RunConfig

import random
import os
import math
import contextlib


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
        print(f"Error in loss prediction: {e}")

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
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.trial = trial
        self.seed = seed
        self.pruner = pruner

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
        # Allow differentiable sampling for flow-based models (e.g., RealNVP)
        use_no_grad = not getattr(self.model, "differentiable_sampling", False)
        with torch.no_grad() if use_no_grad else contextlib.nullcontext():
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

        # Straight-Through Estimator for flow-based models (RealNVP)
        # Forward: use discrete spins for energy calculation
        # Backward: gradients flow through continuous samples
        if getattr(self.model, "differentiable_sampling", False):
            # RealNVP outputs continuous values in (-1, 1)
            # Discretize for energy but keep gradient path
            samples_discrete = torch.sign(samples)
            samples_discrete[samples_discrete == 0] = 1  # handle rare zero case
            # Straight-through: forward uses discrete, backward uses continuous
            samples_for_energy = samples_discrete.detach() + samples - samples.detach()
        else:
            # PixelCNN already outputs discrete samples
            samples_for_energy = samples

        energy = energy_fn(samples_for_energy)
        beta = (1.0 / T_expanded).unsqueeze(-1)  # (total_size, 1)
        loss_raw = log_prob + beta * energy
        loss_view = loss_raw.view(num_beta, batch_size)
        log_prob_view = log_prob.view(num_beta, batch_size)

        self.optimizer.zero_grad()

        # Branching for Differentiable (Reparameterization) vs Non-Differentiable (REINFORCE)
        if getattr(self.model, "differentiable_sampling", False):
            # Reparameterization Trick (Pathwise Derivative)
            # Directly minimize Free Energy: L = E[ log q + beta * E ]
            # This has much lower variance than REINFORCE for differentiable models (e.g. RealNVP)
            loss = loss_raw.mean()
            loss.backward()
            train_loss = loss.item()
        else:
            # REINFORCE (Score Function Estimator)
            # Gradient: E[ (log q + beta * E - baseline) * grad(log q) ]

            # Baseline for variance reduction (mean per beta)
            baselines = loss_view.mean(dim=1, keepdim=True).detach()

            # Advantage must be detached to act as fixed reward signal
            # This prevents double-counting gradients of log_prob
            advantage = (loss_view - baselines).detach()

            loss_REINFORCE = torch.mean(advantage * log_prob_view)
            loss_REINFORCE.backward()

            # Log the actual Free Energy, not the surrogate loss, to be consistent with val_loss
            train_loss = loss_view.mean().item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return train_loss

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

            # Discretize samples for energy calculation if needed
            if getattr(self.model, "differentiable_sampling", False):
                # RealNVP outputs continuous values, discretize for energy
                samples_discrete = torch.sign(samples)
                samples_discrete[samples_discrete == 0] = 1
                samples_for_energy = samples_discrete
            else:
                samples_for_energy = samples

            energy = energy_fn(samples_for_energy)
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
                    # Model loss approximates: -log Z + beta * <E>
                    # At equilibrium, this should equal -log Z
                    # Error = model_loss - (-exact_logz) = model_loss + exact_logz
                    model_loss = val_dict[f"val_loss_beta_{i}"]
                    error_vs_exact = model_loss + exact_logz
                    val_dict[f"val_error_exact_beta_{i}"] = error_vs_exact
                    val_dict[f"val_exact_logz_beta_{i}"] = exact_logz

        return val_dict

    def train(self, energy_fn, epochs):
        val_loss = 0
        val_losses = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(energy_fn)
            val_dict = self.val_epoch(energy_fn)
            val_loss = val_dict["val_loss"]
            val_losses.append(val_loss)

            # Early stopping if loss becomes NaN
            if math.isnan(train_loss) or math.isnan(val_loss):
                print("Early stopping due to NaN loss")
                val_loss = math.inf
                break

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            log_dict = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }

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

            self.scheduler.step()
            wandb.log(log_dict)
            if epoch % 10 == 0 or epoch == epochs - 1:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
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
        print(f"Runtime error during training: {e}")
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
