import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

RANDOM_STATE = 42

def kmeans_sweep(X: pd.DataFrame, plots_dir: str, k_min: int = 2, k_max: int = 6):
    os.makedirs(plots_dir, exist_ok=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best = None
    summary = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        db = davies_bouldin_score(Xs, labels)
        summary.append({"k": k, "silhouette": float(sil), "davies_bouldin": float(db)})

        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        Xp = pca.fit_transform(Xs)
        plt.figure()
        plt.scatter(Xp[:, 0], Xp[:, 1], c=labels, alpha=0.8)
        plt.title(f"KMeans k={k} (sil={sil:.3f}, DB={db:.3f})")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.savefig(os.path.join(plots_dir, f"kmeans_k{k}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        if (best is None) or (sil > best["silhouette"]):
            best = {"k": k, "silhouette": float(sil), "davies_bouldin": float(db)}

    return best, summary
