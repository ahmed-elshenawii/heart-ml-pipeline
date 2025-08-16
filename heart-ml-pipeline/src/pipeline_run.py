import os
import json
import joblib
import pandas as pd

from .download_data import load_or_download
from .data_prep import basic_cleaning, prepare_splits, pca_fit_transform
from .train_supervised import fit_and_select_best
from .train_unsupervised import kmeans_sweep
from .evaluate import save_json

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
META_PATH = os.path.join(MODELS_DIR, "model_meta.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "outputs", "supervised_results.json")
CLUSTER_PATH = os.path.join(BASE_DIR, "outputs", "unsupervised_results.json")

def main():
    # 1) Load/download data
    df = load_or_download()

    # 2) Clean
    df = basic_cleaning(df)

    # 3) Splits
    d = prepare_splits(df)
    X, y = d["X"], d["y"]
    X_train, X_test, y_train, y_test = d["X_train"], d["X_test"], d["y_train"], d["y_test"]

    # 4) PCA visualization (2D) for the full X
    pca, scaler, X_pca = pca_fit_transform(X, n_components=2)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c=y)
    plt.title("PCA (2 components) - Colored by Target")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "pca_2d.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 5) Supervised with HPO
    best, all_results, meta = fit_and_select_best(X_train, y_train, X_test, y_test, PLOTS_DIR, X.columns)

    # 6) Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best["best_estimator"], BEST_MODEL_PATH)
    meta.update({"best_model_name": best["name"], "best_params": best["best_params"]})
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    save_json(all_results, REPORT_PATH)

    # 7) Unsupervised (KMeans sweep)
    best_k, summary = kmeans_sweep(X, PLOTS_DIR, k_min=2, k_max=6)
    save_json({"best": best_k, "summary": summary}, CLUSTER_PATH)

    print("=== PIPELINE COMPLETE ===")
    print(f"Best supervised model: {best['name']} | test AUC={best['test_auc']:.3f}")
    print(f"Saved model to: {BEST_MODEL_PATH}")
    print(f"Reports saved to: {REPORT_PATH} and {CLUSTER_PATH}")
    print(f"Plots saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
