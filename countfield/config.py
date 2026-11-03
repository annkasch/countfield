"""Run configuration (YAML -> pydantic). Relative paths resolve against the YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    path: Path = Field(
        description="count-set npz (countfield.data.save_groups) or photon npz "
        "(countfield.photons.save_photon_groups; required by point_transformer)"
    )
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


class LocalLatentConfig(_Strict):
    """Local-latent NP on counts: K nearest observed points -> mu, sigma (leave-one-out)."""

    kind: Literal["local_latent"] = "local_latent"
    neighbours: int = Field(32, ge=1)
    hidden: int = Field(64, ge=4)
    heads: int = Field(4, ge=1)
    quadrature_nodes: int = Field(20, ge=4)
    batch_size: int = Field(256, ge=1, description="query points per step")

    @model_validator(mode="after")
    def _heads_divide(self) -> LocalLatentConfig:
        if self.hidden % self.heads:
            raise ValueError(f"hidden ({self.hidden}) must be divisible by heads ({self.heads})")
        return self


class PointTransformerConfig(_Strict):
    """Point Transformer U-Net over the point cloud; point features = photon sets (masked)."""

    kind: Literal["point_transformer"] = "point_transformer"
    dims: list[int] = Field([32, 64, 64], min_length=1, description="channels per level")
    blocks: int = Field(1, ge=1, description="PT blocks per level (encoder and decoder)")
    neighbours: int = Field(16, ge=2)
    downsample_ratio: int = Field(4, ge=2)
    photon_hidden: int = Field(64, ge=4)
    quadrature_nodes: int = Field(20, ge=4)
    mask_fraction: float = Field(
        0.15, gt=0, lt=1, description="share of training points masked (targets) per step"
    )
    misses_per_point: int | None = Field(
        256,
        ge=1,
        description="missed photons sampled per visible point and step (with "
        "replacement, weighted to an unbiased sum); null = all",
    )
    eval_misses_per_point: int | None = Field(
        None, ge=1, description="as misses_per_point for holdout/evaluation passes; null = all"
    )


ModelConfig = Annotated[LocalLatentConfig | PointTransformerConfig, Field(discriminator="kind")]


class TrainingConfig(_Strict):
    seed: int = 0
    steps: int = Field(5000, ge=1)
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    holdout_fraction: float = Field(
        0.1, gt=0, lt=1, description="share of observed points held out for checkpoint selection"
    )
    eval_every: int = Field(100, ge=1)


class EvaluationConfig(_Strict):
    latent_grid: int = Field(2048, ge=16, description="latent quantiles for predictive pmfs")
    coverage_ylim: tuple[float, float] | None = Field(
        None, description="fixed y range of the coverage plots (k/n); null = from the data"
    )

    @model_validator(mode="after")
    def _ordered_ylim(self) -> EvaluationConfig:
        if self.coverage_ylim is not None and not self.coverage_ylim[0] < self.coverage_ylim[1]:
            raise ValueError(f"coverage_ylim must be (low, high), got {self.coverage_ylim}")
        return self


class RunConfig(_Strict):
    data: DataConfig
    model: ModelConfig = Field(default_factory=LocalLatentConfig)
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    output_directory: Path

    @model_validator(mode="before")
    @classmethod
    def _defaults(cls, raw):
        """Model kind defaults to local_latent; training.batch_size moved to the model."""
        if not isinstance(raw, dict):
            return raw
        raw = dict(raw)
        model = dict(raw.get("model") or {})
        model.setdefault("kind", "local_latent")
        training = dict(raw.get("training") or {})
        if "batch_size" in training and model["kind"] == "local_latent":
            model.setdefault("batch_size", training.pop("batch_size"))
        raw["model"], raw["training"] = model, training
        return raw


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    config = RunConfig.model_validate(raw)
    base = path.resolve().parent
    data = config.data.model_copy(update={"path": (base / config.data.path).resolve()})
    return config.model_copy(
        update={"data": data, "output_directory": (base / config.output_directory).resolve()}
    )
