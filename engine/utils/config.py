from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

METRIC_PATTERN = re.compile(r"^(?:(?P<mode>\w+)_)?(?P<metric>\w+)@(?P<k>\d+)$")


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RuntimeConfig(ImmutableModel):
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    enable_amp: bool = True
    compile_model: bool = True
    seed: int = 2026
    show_progress: bool = False
    n_workers: Annotated[int, Field(ge=0)] = 0


class TrainingConfig(ImmutableModel):
    epochs: Annotated[int, Field(gt=0)] = 35
    learning_rate: Annotated[float, Field(gt=0.0)] = 0.0025
    weight_decay: Annotated[float, Field(ge=0.0)] = 1e-07
    train_batch_size: Annotated[int, Field(gt=0)] = 256
    eval_batch_size: Annotated[int, Field(gt=0)] = 256
    stride: Annotated[int, Field(gt=0)] = 150
    early_stopping: Annotated[int, Field(gt=0)] = 5
    lr_decay_factor: Annotated[float, Field(gt=0.0, le=1.0)] = 0.96


class ModelConfig(ImmutableModel):
    model_type: Literal["SASRecNext", "SASRec"] = "SASRecNext"
    tied_weights: bool = True
    d_model: Annotated[int, Field(gt=0)] = 128
    n_heads: Annotated[int, Field(gt=0)] = 4
    n_layers: Annotated[int, Field(gt=0)] = 3
    max_seq_len: Annotated[int, Field(gt=0)] = 200
    attn_dropout: Annotated[float, Field(ge=0.0, le=1.0)] = 0
    ffn_dim: Annotated[int, Field(gt=0)] = 384
    ffn_dropout: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    embedding_dropout: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1

    @model_validator(mode="after")
    def _validate_heads(self) -> ModelConfig:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        return self


class EvaluationConfig(ImmutableModel):
    mode: Literal["uni100", "both", "full"] = "full"
    top_k: tuple[int, ...] = (10, 20, 50, 200)
    metrics: tuple[str, ...] = ("recall", "ndcg", "mrr")
    valid_metric: str = "mrr@10"
    max_seq_len: Annotated[int, Field(gt=0)] = 200
    eval_set: Literal["val", "test", "both"] = "test"

    @model_validator(mode="after")
    def _validate_eval(self) -> EvaluationConfig:
        if any(k <= 0 for k in self.top_k):
            raise ValueError(f"All top_k values must be > 0: {self.top_k}")

        allowed_metrics = {"recall", "ndcg", "mrr"}
        configured_metrics = {m.lower() for m in self.metrics}

        # Check invalid metrics using set difference
        if invalid := configured_metrics - allowed_metrics:
            raise ValueError(f"Invalid metrics: {invalid}. Supported: {allowed_metrics}")

        # Parse valid_metric in one stroke, mode prefix is now optional
        match = METRIC_PATTERN.match(self.valid_metric.lower())
        if not match:
            raise ValueError(f"valid_metric '{self.valid_metric}' must follow '[mode_]metric@k' format")

        parsed_mode, parsed_metric, parsed_k = match.groups()

        # If user didn't specify a mode, default it intelligently
        if parsed_mode is None:
            parsed_mode = "full" if self.mode == "both" else self.mode

        allowed_modes = {"uni100", "both", "full"}
        if parsed_mode not in allowed_modes:
            raise ValueError(f"Invalid mode in valid_metric: '{parsed_mode}'. Supported: {allowed_modes}")

        # Verify it doesn't contradict the evaluation mode
        if parsed_mode != self.mode and self.mode != "both":
            raise ValueError(f"valid_metric mode '{parsed_mode}' contradicts evaluation mode '{self.mode}'")

        if parsed_metric not in configured_metrics:
            raise ValueError(f"valid_metric '{parsed_metric}' must be one of {configured_metrics}")

        if self.mode == "both":
            # Evaluator keeps the prefix when mode is "both"
            expected_key = f"{parsed_mode}_{parsed_metric}@{parsed_k}"
        else:
            # Evaluator strips the prefix when mode is not "both"
            expected_key = f"{parsed_metric}@{parsed_k}"

        object.__setattr__(self, "valid_metric", expected_key)

        if int(parsed_k) not in self.top_k:
            raise ValueError(f"valid_metric @{parsed_k} is not in top_k: {self.top_k}")
        return self


class DataConfig(ImmutableModel):
    dataset: Literal["ml-1m", "ml-10m"] = "ml-10m"
    raw_data_dir: str = ""
    processed_data_dir: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""
    train_seq_mode: Literal["sliding_window", "recent_only"] = "sliding_window"

    @model_validator(mode="before")
    @classmethod
    def _set_default_paths(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        dataset = data.get("dataset", "ml-10m")
        if not data.get("raw_data_dir"):
            data["raw_data_dir"] = f"data/raw/{dataset}"
        if not data.get("processed_data_dir"):
            data["processed_data_dir"] = f"data/processed/{dataset}"
        if not data.get("checkpoint_dir"):
            data["checkpoint_dir"] = f"checkpoints/{dataset}"
        if not data.get("log_dir"):
            data["log_dir"] = f"logs/{dataset}"
        return data


class AssetsConfig(ImmutableModel):
    base_url: str = "https://github.com/mohiey-507/SASRecNext-Engine/releases/download"
    release_tag: str = "v1.0-ml10m"
    files: list[str] = Field(
        default_factory=lambda: [
            "sasrecnext_tied_best_model.pt",
        ]
    )


class Config(ImmutableModel):
    runtime: RuntimeConfig = Field(default=RuntimeConfig())
    training: TrainingConfig = Field(default=TrainingConfig())
    model: ModelConfig = Field(default=ModelConfig())
    evaluation: EvaluationConfig = Field(default=EvaluationConfig())
    data: DataConfig = Field(default=DataConfig())
    assets: AssetsConfig = Field(default=AssetsConfig())

    @model_validator(mode="before")
    @classmethod
    def _set_default_eval_max_seq_len(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        model = data.get("model", {})
        evaluation = data.get("evaluation", {})

        if isinstance(evaluation, dict) and "max_seq_len" not in evaluation:
            evaluation["max_seq_len"] = model.get("max_seq_len", 200) if isinstance(model, dict) else 200
            data["evaluation"] = evaluation

        return data


def load_config(path: Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load configuration from a YAML file, falling back to defaults for missing fields."""
    if path is None:
        raw: dict[str, Any] = {}
    else:
        raw = yaml.safe_load(path.read_text()) or {}

    if overrides:
        for override in overrides:
            if "=" not in override:
                continue
            key_path, value_str = override.split("=", 1)
            keys = key_path.split(".")

            # Simple conversion for boolean and numbers since Pydantic does the rest
            val: Any = value_str
            if value_str.lower() == "true":
                val = True
            elif value_str.lower() == "false":
                val = False
            elif value_str.isdigit():
                val = int(value_str)

            # Traverse nested dict
            d = raw
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val
    # Dynamic Directory Naming based on config file
    if path is not None:
        dataset = raw.get("data", {}).get("dataset", "ml-10m")
        config_name = path.stem

        data_dict = raw.setdefault("data", {})

        if not data_dict.get("checkpoint_dir"):
            data_dict["checkpoint_dir"] = f"checkpoints/{dataset}/{config_name}"

        if not data_dict.get("log_dir"):
            data_dict["log_dir"] = f"logs/{dataset}/{config_name}"

    return Config(**raw)
