import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

RANDOM_STATE = 42
CV_SPLITS = 5

MODELS_PARAM_GRID = {
    "logreg": (
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_STATE))]),
        {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
        },
    ),
    "svc": (
        Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
        {
            "clf__C": [0.5, 1.0, 2.0],
            "clf__gamma": ["scale", 0.01, 0.1],
        },
    ),
    "rf": (
        Pipeline([("clf", RandomForestClassifier(random_state=RANDOM_STATE))]),
        {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5],
        },
    ),
}

def fit_and_select_best(X_train, y_train, X_test, y_test, plots_dir: str, feature_names):
    os.makedirs(plots_dir, exist_ok=True)
    results = []
    best_overall = None

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for name, (pipe, grid) in MODELS_PARAM_GRID.items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        y_proba = gs.predict_proba(X_test)[:, 1]
        y_pred = gs.predict(X_test)
        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # ROC curve
        RocCurveDisplay.from_estimator(gs.best_estimator_, X_test, y_test)
        plt.title(f"ROC Curve - {name}")
        plt.savefig(os.path.join(plots_dir, f"roc_{name}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Confusion matrix heatmap
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.savefig(os.path.join(plots_dir, f"cm_{name}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        results.append({
            "name": name,
            "best_params": gs.best_params_,
            "cv_best_score_roc_auc": gs.best_score_,
            "test_auc": float(auc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "best_estimator": gs.best_estimator_,
        })

        if (best_overall is None) or (auc > best_overall["test_auc"]):
            best_overall = results[-1]

    # Persist the best model (pipeline) with feature names meta
    meta = {"feature_names": list(feature_names)}
    return best_overall, results, meta
