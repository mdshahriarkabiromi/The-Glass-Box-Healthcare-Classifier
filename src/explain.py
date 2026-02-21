import argparse
import yaml
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lime.lime_tabular import LimeTabularExplainer

from src.utils import Paths, load_joblib, save_joblib, ensure_dir
from src.features import infer_feature_types, build_preprocessor

def get_feature_names(pipeline: Pipeline) -> np.ndarray:
    pre = pipeline.named_steps["preprocess"]
    return pre.get_feature_names_out()

def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = Paths()
    pipeline = load_joblib(paths.model_path)

    df = pd.read_csv(cfg["data"]["path"])
    if cfg["data"].get("id_column") in df.columns:
        df = df.drop(columns=[cfg["data"]["id_column"]])

    target = cfg["data"]["target"]
    X = df.drop(columns=[target])

    ensure_dir(paths.figures_dir)

    # ---------------------------------------------------------
    # 1) SHAP via surrogate glass-box on transformed feature space
    # ---------------------------------------------------------
    if cfg["explain"]["surrogate"]["enabled"]:
        # Use the fitted preprocessor to transform into model feature space
        pre = pipeline.named_steps["preprocess"]
        X_t = pre.transform(X)
        feature_names = get_feature_names(pipeline)

        # Soft labels from real ensemble
        y_soft = pipeline.predict_proba(X)[:, 1]
        y_bin = (y_soft >= 0.5).astype(int)

        surrogate = LogisticRegression(max_iter=6000, class_weight="balanced")
        surrogate.fit(X_t, y_bin)
        save_joblib(surrogate, paths.surrogate_path)

        # SHAP
        bg_n = min(int(cfg["explain"]["background_samples"]), X_t.shape[0])
        ex_n = min(int(cfg["explain"]["shap_samples"]), X_t.shape[0])

        rng = np.random.default_rng(42)
        bg_idx = rng.choice(X_t.shape[0], size=bg_n, replace=False)
        ex_idx = rng.choice(X_t.shape[0], size=ex_n, replace=False)

        background = X_t[bg_idx]
        explain_X = X_t[ex_idx]

        explainer = shap.LinearExplainer(surrogate, background, feature_perturbation="interventional")
        shap_values = explainer(explain_X)

        shap.summary_plot(shap_values, features=explain_X, feature_names=feature_names, show=False, max_display=20)
        plt.title("SHAP Summary (Glass-Box Surrogate)")
        plt.savefig(f"{paths.figures_dir}/shap_beeswarm_surrogate.png", bbox_inches="tight")
        plt.close()

        shap.summary_plot(shap_values, features=explain_X, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
        plt.title("SHAP Global Importance (Surrogate)")
        plt.savefig(f"{paths.figures_dir}/shap_bar_surrogate.png", bbox_inches="tight")
        plt.close()

        # Local waterfall for one patient
        i = 0
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[i],
            base_values=explainer.expected_value,
            data=explain_X[i],
            feature_names=feature_names
        ), max_display=15, show=False)
        plt.title("SHAP Waterfall (Surrogate) — Example Patient")
        plt.savefig(f"{paths.figures_dir}/shap_waterfall_surrogate_patient0.png", bbox_inches="tight")
        plt.close()

        print("✅ Saved SHAP figures (surrogate) in:", paths.figures_dir)

    # ---------------------------------------------------------
    # 2) LIME local explanation against REAL pipeline
    # ---------------------------------------------------------
    pre = pipeline.named_steps["preprocess"]
    X_t_full = pre.transform(X)
    feature_names = get_feature_names(pipeline)

    lime_explainer = LimeTabularExplainer(
        training_data=np.array(X_t_full),
        feature_names=list(feature_names),
        class_names=["no_stroke", "stroke"],
        mode="classification"
    )

    # predict_fn expects transformed inputs -> we call internal model directly
    model = pipeline.named_steps["model"]
    def predict_proba_transformed(Xt):
        return model.predict_proba(Xt)

    exp = lime_explainer.explain_instance(
        data_row=np.array(X_t_full[0]),
        predict_fn=predict_proba_transformed,
        num_features=int(cfg["explain"]["lime_num_features"])
    )

    fig = exp.as_pyplot_figure()
    fig.suptitle("LIME Local Explanation (Real Ensemble) — Patient 0")
    fig.savefig(f"{paths.figures_dir}/lime_patient0.png", bbox_inches="tight")
    plt.close(fig)

    print("✅ Saved LIME figure in:", paths.figures_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    main(args.config)