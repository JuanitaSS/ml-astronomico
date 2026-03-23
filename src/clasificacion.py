import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def ejecutar_clasificacion(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"      Accuracy: {acc:.4f}")

    try:
        clases = np.load("outputs/label_encoder_classes.npy", allow_pickle=True).tolist()
    except FileNotFoundError:
        clases = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clases, yticklabels=clases, ax=ax)
    ax.set_title("Matriz de Confusión — KNN (k=5)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig("outputs/clasificacion_confusion_matrix.png", dpi=150)
    plt.close()

    fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    axes[0].bar([clases[i] for i in unique_test], counts_test, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Distribución Real", fontweight="bold"); axes[0].set_ylabel("Cantidad")
    axes[1].bar([clases[i] for i in unique_pred], counts_pred, color="#55A868", edgecolor="white")
    axes[1].set_title("Distribución Predicha", fontweight="bold"); axes[1].set_ylabel("Cantidad")
    fig2.suptitle("KNN — Comparación de Distribuciones", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/clasificacion_distribucion.png", dpi=150)
    plt.close()

    metricas = {
        "modelo": "KNN", "k": 5, "split": "70/30",
        "accuracy": round(float(acc), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": {k: v for k, v in report.items() if k != "accuracy"},
    }
    with open("outputs/metricas_clasificacion.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print("      Guardado: outputs/metricas_clasificacion.json")
    return metricas
