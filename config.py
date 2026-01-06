from dataclasses import dataclass, asdict, field
import optuna
import yaml
import importlib
import hashlib
import json


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 10
    mode: str = "min"  # "min" or "max"
    min_delta: float = 0.0001


@dataclass
class RunConfig:
    project: str
    device: str
    seeds: list[int]
    net: str
    optimizer: str
    scheduler: str
    epochs: int
    batch_size: int
    net_config: dict[str, int]
    optimizer_config: dict[str, int | float]
    scheduler_config: dict[str, int | float]
    early_stopping_config: EarlyStoppingConfig = field(
        default_factory=lambda: EarlyStoppingConfig()
    )

    def __post_init__(self):
        if isinstance(self.early_stopping_config, dict):
            self.early_stopping_config = EarlyStoppingConfig(
                **self.early_stopping_config
            )

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return cls(**config)

    def to_yaml(self, path: str):
        with open(path, "w") as file:
            yaml.dump(asdict(self), file, sort_keys=False)

    def create_model(self):
        module_name, class_name = self.net.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class(self.net_config, device=self.device)

    def create_optimizer(self, model):
        module_name, class_name = self.optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        optimizer_class = getattr(module, class_name)
        return optimizer_class(model.parameters(), **self.optimizer_config)

    def create_scheduler(self, optimizer):
        scheduler_module, scheduler_class_name = self.scheduler.rsplit(".", 1)
        scheduler_module = importlib.import_module(scheduler_module)
        scheduler_class = getattr(scheduler_module, scheduler_class_name)

        # Special handling for OneCycleLR which requires total_steps
        if scheduler_class_name == "OneCycleLR":
            scheduler_config = self.scheduler_config.copy()
            # Calculate total_steps from epochs and accumulation_steps
            accumulation_steps = self.net_config.get("accumulation_steps", 1)
            total_steps = self.epochs * accumulation_steps
            scheduler_config["total_steps"] = total_steps
            return scheduler_class(optimizer, **scheduler_config)

        return scheduler_class(optimizer, **self.scheduler_config)

    def gen_group_name(self):
        """
        Generate a concise group name with key params + hash for uniqueness.

        Format: {ModelName}_lr{lr}_e{epochs}_{hash}

        The hash is computed from the entire config (net_config, optimizer_config,
        scheduler_config, etc.) ensuring uniqueness for any configuration change
        including curriculum learning phase parameters.
        """
        # Model name
        model_name = self.net.split(".")[-1]

        # Key params: lr and epochs
        lr = self.optimizer_config.get("lr", 0)
        lr_str = f"{lr:.0e}".replace("+", "").replace("0", "")  # e.g., "5e-3"

        # Compute hash from full config for uniqueness
        config_hash = self._compute_config_hash()

        name = f"{model_name}_lr{lr_str}_e{self.epochs}_{config_hash}"
        return name

    def _compute_config_hash(self, length=6):
        """
        Compute a short hash from the full config for uniqueness.

        Includes all config values: net_config (with phase info), optimizer_config,
        scheduler_config, early_stopping_config, etc.

        Args:
            length: Number of characters for the hash (default: 6)

        Returns:
            str: Short hash string (hex)
        """
        # Collect all config values that affect training behavior
        hash_dict = {
            "net": self.net,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "batch_size": self.batch_size,
            "net_config": self.net_config,
            "optimizer_config": self.optimizer_config,
            "scheduler_config": self.scheduler_config,
            "early_stopping_config": asdict(self.early_stopping_config),
        }

        # Convert to sorted JSON string for deterministic hashing
        config_str = json.dumps(hash_dict, sort_keys=True, default=str)

        # Compute SHA256 hash and take first `length` characters
        hash_bytes = hashlib.sha256(config_str.encode()).hexdigest()
        return hash_bytes[:length]

    def gen_tags(self):
        """
        Generate concise tags for wandb.

        Tags must be <= 64 characters each.
        Format: [ModelName, key_param=value (scalars only), Optimizer, Scheduler, hash]
        """
        model_name = self.net.split(".")[-1]
        optimizer_name = self.optimizer.split(".")[-1]
        scheduler_name = self.scheduler.split(".")[-1]
        config_hash = self._compute_config_hash(length=8)

        tags = [model_name]

        # Only include scalar values (skip dicts, lists)
        for k, v in self.net_config.items():
            if isinstance(v, (int, float, bool, str)) and not isinstance(v, bool):
                tag = f"{k[:12]}={v}"  # Truncate key to 12 chars
                if len(tag) <= 64:
                    tags.append(tag)
            elif isinstance(v, bool):
                if v:  # Only add if True
                    tags.append(k[:20])

        tags.extend([optimizer_name, scheduler_name, f"cfg:{config_hash}"])
        return tags

    def gen_config(self):
        return asdict(self)


def default_run_config():
    return RunConfig(
        project="PyTorch_Template",
        device="cpu",
        seeds=[42],
        net="MLP",
        optimizer="torch.optim.adamw.AdamW",
        scheduler="torch.optim.lr_scheduler.CosineAnnealingLR",
        epochs=50,
        batch_size=256,
        net_config={
            "nodes": 128,
            "layers": 4,
        },
        optimizer_config={
            "lr": 1e-3,
        },
        scheduler_config={
            "T_max": 50,
            "eta_min": 1e-5,
        },
    )


@dataclass
class OptimizeConfig:
    study_name: str
    trials: int
    seed: int
    metric: str
    direction: str
    sampler: dict = field(default_factory=dict)
    pruner: dict = field(default_factory=dict)
    search_space: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return cls(**config)

    def to_yaml(self, path):
        with open(path, "w") as file:
            yaml.dump(asdict(self), file, sort_keys=False)

    def _create_sampler(self):
        module_name, class_name = self.sampler["name"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        sampler_class = getattr(module, class_name)
        sampler_kwargs = self.sampler.get("kwargs", {})
        if class_name == "GridSampler":
            sampler_kwargs["search_space"] = self.grid_search_space()
        return sampler_class(**sampler_kwargs)

    def create_pruner(self):
        if not self.pruner:
            return None
        module_name, class_name = self.pruner["name"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        pruner_class = getattr(module, class_name)
        pruner_kwargs = self.pruner.get("kwargs", {})
        return pruner_class(**pruner_kwargs)

    def create_study(self, project):
        sampler = self._create_sampler()
        study = {
            "study_name": self.study_name,
            "storage": f"sqlite:///{project}.db",
            "sampler": sampler,
            "direction": self.direction,
            "load_if_exists": True,
        }
        return optuna.create_study(**study)

    def suggest_params(self, trial):
        params = {}
        for category, config in self.search_space.items():
            params[category] = {}
            for param, param_config in config.items():
                if param_config["type"] == "int":
                    params[category][param] = trial.suggest_int(
                        f"{category}_{param}",
                        param_config["min"],
                        param_config["max"],
                        step=param_config.get("step", 1),
                    )
                elif param_config["type"] == "float":
                    if param_config.get("log", False):
                        params[category][param] = trial.suggest_float(
                            f"{category}_{param}",
                            param_config["min"],
                            param_config["max"],
                            log=True,
                        )
                    else:
                        params[category][param] = trial.suggest_float(
                            f"{category}_{param}",
                            param_config["min"],
                            param_config["max"],
                        )
                elif param_config["type"] == "categorical":
                    params[category][param] = trial.suggest_categorical(
                        f"{category}_{param}", param_config["choices"]
                    )
        return params

    def grid_search_space(self):
        params = {}
        for category, config in self.search_space.items():
            for param, param_config in config.items():
                if param_config["type"] == "categorical":
                    params[f"{category}_{param}"] = param_config["choices"]
                else:
                    raise ValueError(
                        f"Unsupported grid search space type: {param_config['type']}"
                    )
        return params


def abbreviate(s: str):
    return "".join([w for w in s if w.isupper()])
