import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

COLORES_CLUSTER = ["#E05C5C", "#4C72B0", "#55A868"]
COLORES_CLASE   = {"GALAXY": "#4C72B0", "STAR": "#55A868", "QSO": "#E05C5C"}

def ejecutar_clustering(X, df):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_cluster = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, labels_cluster)
    inercia   = float(kmeans.inertia_)
    print(f"      Silhouette: {sil_score:.4f}  Inercia: {inercia:.2f}")

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_
    clases_reales = df["class"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i in range(3):
        mask = labels_cluster == i
        axes[0].scatter(X_2d[mask,0], X_2d[mask,1], c=COLORES_CLUSTER[i],
                        s=15, alpha=0.6, label=f"Cluster {i+1} (n={mask.sum()})")
    centroides_2d = pca.transform(kmeans.cluster_centers_)
    axes[0].scatter(centroides_2d[:,0], centroides_2d[:,1], c="black",
                    marker="X", s=120, zorder=5, label="Centroides")
    axes[0].set_title(f"Clusters KMeans (k=3)\nVar. explicada: {var_exp.sum():.1%}",
                      fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({var_exp[0]:.1%})"); axes[0].set_ylabel(f"PC2 ({var_exp[1]:.1%})")
    axes[0].legend(fontsize=8)

    le = LabelEncoder()
    le.fit(clases_reales)
    for idx, clase in enumerate(le.classes_):
        mask = clases_reales == clase
        axes[1].scatter(X_2d[mask,0], X_2d[mask,1],
                        c=COLORES_CLASE.get(clase, f"C{idx}"),
                        s=15, alpha=0.6, label=f"{clase} (n={mask.sum()})")
    axes[1].set_title("Clases Reales SDSS\nPCA 2D", fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({var_exp[0]:.1%})"); axes[1].set_ylabel(f"PC2 ({var_exp[1]:.1%})")
    axes[1].legend(fontsize=8)
    fig.suptitle("KMeans vs Clases Reales — SDSS", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/clustering_comparacion_pca.png", dpi=150)
    plt.close()

    tabla = pd.crosstab(pd.Series(labels_cluster, name="Cluster"),
                        pd.Series(clases_reales, name="Clase Real"))
    tabla.index = [f"Cluster {i+1}" for i in tabla.index]
    fig2, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(tabla, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Distribución de Clases por Cluster", fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/clustering_heatmap_clases.png", dpi=150)
    plt.close()

    inercias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_ for k in range(1,9)]
    fig3, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1,9), inercias, "o-", color="#4C72B0", linewidth=2, markersize=7)
    ax.axvline(3, color="#E05C5C", linestyle="--", label="k=3 seleccionado")
    ax.set_xlabel("Clusters (k)"); ax.set_ylabel("Inercia (WCSS)")
    ax.set_title("Método del Codo", fontweight="bold"); ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/clustering_elbow.png", dpi=150)
    plt.close()

    metricas = {
        "modelo": "KMeans", "k": 3,
        "silhouette_score": round(float(sil_score), 4),
        "inercia_wcss": round(inercia, 2),
        "distribucion_clusters": {f"cluster_{i+1}": int((labels_cluster==i).sum()) for i in range(3)},
    }
    with open("outputs/metricas_clustering.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print("      Guardado: outputs/metricas_clustering.json")
    return metricas
