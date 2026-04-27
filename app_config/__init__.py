"""
Загрузка конфига из yaml + переопределение через env переменные.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class ClearMLConfig:
    project_name: str
    dataset_name: str
    dataset_version: str
    queue_name: str
    serving_name: str
    endpoint: str


@dataclass
class ExperimentConfig:
    name: str
    max_features: int
    ngram_max: int
    C: float
    max_iter: int
    solver: str
    random_seed: int


@dataclass
class TrainingConfig:
    experiments: List[ExperimentConfig]


@dataclass
class ServingConfig:
    host: str
    port: int
    endpoint: str
    health_endpoint: str
    timeout: int


@dataclass
class UIConfig:
    port: int
    default_serving_url: str
    request_timeout: int


@dataclass
class AppConfig:
    clearml: ClearMLConfig
    training: TrainingConfig
    serving: ServingConfig
    ui: UIConfig


def load_config() -> AppConfig:
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)

    # ClearML config
    clearml_cfg = ClearMLConfig(**raw["clearml"])

    # Experiments
    experiments = [
        ExperimentConfig(**exp)
        for exp in raw["training"]["experiments"]
    ]
    training_cfg = TrainingConfig(experiments=experiments)

    # Serving — env переменные имеют приоритет
    serving_raw = raw["serving"]
    serving_cfg = ServingConfig(
        host=os.getenv("SERVING_HOST", serving_raw["host"]),
        port=int(os.getenv("SERVING_PORT", serving_raw["port"])),
        endpoint=serving_raw["endpoint"],
        health_endpoint=serving_raw["health_endpoint"],
        timeout=serving_raw["timeout"],
    )

    # UI
    ui_raw = raw["ui"]
    ui_cfg = UIConfig(
        port=int(os.getenv("UI_PORT", ui_raw["port"])),
        default_serving_url=os.getenv(
            "SERVING_URL", ui_raw["default_serving_url"]
        ),
        request_timeout=ui_raw["request_timeout"],
    )

    return AppConfig(
        clearml=clearml_cfg,
        training=training_cfg,
        serving=serving_cfg,
        ui=ui_cfg,
    )


cfg = load_config()
