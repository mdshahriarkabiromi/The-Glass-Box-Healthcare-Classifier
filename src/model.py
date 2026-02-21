from dataclasses import dataclass
from typing import Literal
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

@dataclass
class ModelConfig:
    use_calibration: bool
    calibration_method: Literal["sigmoid", "isotonic"]
    passthrough: bool
    n_jobs: int
    random_state: int

def build_model(cfg: ModelConfig):
    base_estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            n_jobs=cfg.n_jobs, random_state=cfg.random_state,
            class_weight="balanced"
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=2,
            n_jobs=cfg.n_jobs, random_state=cfg.random_state,
            class_weight="balanced"
        )),
        ("hgb", HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=400,
            random_state=cfg.random_state
        )),
        ("svm", SVC(
            kernel="rbf", C=2.0, gamma="scale",
            probability=True, class_weight="balanced",
            random_state=cfg.random_state
        )),
    ]

    meta = LogisticRegression(max_iter=4000, class_weight="balanced")

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        passthrough=cfg.passthrough,
        n_jobs=cfg.n_jobs,
        stack_method="predict_proba"
    )

    if cfg.use_calibration:
        return CalibratedClassifierCV(stack, method=cfg.calibration_method, cv=5)
    return stack