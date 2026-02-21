import argparse
import yaml
from sklearn.pipeline import Pipeline

from src.utils import Paths, set_seed, save_joblib, save_json
from src.data import load_csv, split_data, SplitConfig
from src.features import infer_feature_types, build_preprocessor
from src.model import build_model, ModelConfig
from src.evaluate import evaluate_binary, save_diagnostics_plots

def pick_threshold(y_val, p_val, strategy: str, min_recall: float) -> float:
    thresholds = [i / 100 for i in range(5, 96)]
    best_t = 0.5
    best_score = -1.0

    for t in thresholds:
        m = evaluate_binary(y_val, p_val, threshold=t)
        if strategy == "f1":
            score = m["f1"]
            if score > best_score:
                best_score, best_t = score, t
        elif strategy == "recall_at_least":
            # prioritize: meet recall constraint, then maximize precision
            if m["recall"] >= min_recall:
                score = m["precision"]
                if score > best_score:
                    best_score, best_t = score, t
        else:
            raise ValueError("Unknown threshold strategy.")
    return best_t

def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["split"]["random_state"]
    set_seed(seed)

    df = load_csv(cfg["data"]["path"])

    split_cfg = SplitConfig(
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        stratify=cfg["split"]["stratify"],
        random_state=seed
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target=cfg["data"]["target"],
        cfg=split_cfg,
        id_column=cfg["data"].get("id_column", None),
    )

    num_cols, cat_cols = infer_feature_types(X_train)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    model_cfg = ModelConfig(
        use_calibration=cfg["model"]["use_calibration"],
        calibration_method=cfg["model"]["calibration_method"],
        passthrough=cfg["model"]["stack"]["passthrough"],
        n_jobs=cfg["training"]["n_jobs"],
        random_state=seed
    )

    model = build_model(model_cfg)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    p_val = pipeline.predict_proba(X_val)[:, 1]
    t = pick_threshold(
        y_val, p_val,
        strategy=cfg["threshold"]["strategy"],
        min_recall=float(cfg["threshold"]["min_recall"])
    )

    p_test = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary(y_test, p_test, threshold=t)
    metrics["threshold_strategy"] = cfg["threshold"]["strategy"]

    paths = Paths()
    save_joblib(pipeline, paths.model_path)
    save_json(metrics, paths.metrics_path)
    save_diagnostics_plots(y_test, p_test, paths.figures_dir, threshold=t)

    print("✅ Saved model:", paths.model_path)
    print("✅ Saved metrics:", paths.metrics_path)
    print("✅ Saved figures:", paths.figures_dir)
    print("Chosen threshold:", t)
    print("Test metrics:", {k: metrics[k] for k in ["roc_auc", "pr_auc", "precision", "recall", "f1", "brier"]})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    main(args.config)