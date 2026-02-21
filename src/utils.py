import os
import json
import joblib
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict

def set_seed(seed: int) -> None:
    np.random.seed(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_joblib(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)

def load_joblib(path: str) -> Any:
    return joblib.load(path)

@dataclass
class Paths:
    model_path: str = "models/artifacts/pipeline.joblib"
    metrics_path: str = "models/reports/metrics.json"
    figures_dir: str = "reports/figures"
    model_card_path: str = "reports/model_card.md"
    surrogate_path: str = "models/artifacts/surrogate.joblib"