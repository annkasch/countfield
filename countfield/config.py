"""Run configuration (YAML -> pydantic). Relative paths resolve against the YAML file."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    path: Path = Field(description="npz written by countfield.data.save_groups")
    observations: list[str] = Field(min_length=1, description="groups pooled as observed points")
    evaluation: list[str] = Field(default_factory=list, description="held-out groups to score")
    merge_decimals: int = Field(
        6,
        ge=0,
        description="points whose positions agree after rounding to this many "
        "decimals (data units) are merged by summing k and n",
    )

    @model_validator(mode="after")
    def _disjoint(self) -> DataConfig:
        both = set(self.observations) & set(self.evaluation)
        if both:
            raise ValueError(f"groups both observed and evaluated: {sorted(both)}")
        return self


class ModelConfig(_Strict):
    neighbours: int = Field(32, ge=1)
    hidden: int = Field(64, ge=4)
    heads: int = Field(4, ge=1)
    quadrature_nodes: int = Field(20, ge=4)

    @model_validator(mode="after")
    def _heads_divide(self) -> ModelConfig:
        if self.hidden % self.heads:
            raise ValueError(f"hidden ({self.hidden}) must be divisible by heads ({self.heads})")
        return self


class TrainingConfig(_Strict):
    seed: int = 0
    steps: int = Field(5000, ge=1)
    batch_size: int = Field(256, ge=1)
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    holdout_fraction: float = Field(
        0.1, gt=0, lt=1, description="share of observed points held out for checkpoint selection"
    )
    eval_every: int = Field(100, ge=1)


class EvaluationConfig(_Strict):
    latent_grid: int = Field(2048, ge=16, description="latent quantiles for predictive pmfs")


class RunConfig(_Strict):
    data: DataConfig
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    output_directory: Path


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    config = RunConfig.model_validate(raw)
    base = path.resolve().parent
    data = config.data.model_copy(update={"path": (base / config.data.path).resolve()})
    return config.model_copy(
        update={"data": data, "output_directory": (base / config.output_directory).resolve()}
    )
