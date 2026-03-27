import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix

def ejecutar_regresion(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"      MSE: {mse:.6f}  RMSE: {rmse:.6f}  R²: {r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.4, s=12, color="#E05C5C")
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "k--", lw=1.5, label="Ideal")
    axes[0].set_xlabel("Redshift Real"); axes[0].set_ylabel("Redshift Predicho")
    axes[0].set_title(f"Predicho vs Real\nR² = {r2:.4f}", fontweight="bold"); axes[0].legend()
    residuos = y_test - y_pred
    axes[1].scatter(y_pred, residuos, alpha=0.4, s=12, color="#4C72B0")
    axes[1].axhline(0, color="black", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Redshift Predicho"); axes[1].set_ylabel("Residuos")
    axes[1].set_title(f"Residuos\nMSE = {mse:.6f}", fontweight="bold")
    fig.suptitle("Regresión Lineal — Redshift", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/regresion_diagnostico.png", dpi=150)
    plt.close()

    features = ["u", "g", "r", "i", "z"]
    coefs = model.coef_
    colors = ["#55A868" if c > 0 else "#E05C5C" for c in coefs]
    fig3, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(features, coefs, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Coeficientes del Modelo Lineal", fontweight="bold")
    for bar, val in zip(bars, coefs):
        ax.text(val + 0.001*np.sign(val), bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/regresion_coeficientes.png", dpi=150)
    plt.close()

    # Construir una matriz de confusión a partir de redshift discretizado en bins
    n_bins = 5
    edges = np.linspace(min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()), n_bins + 1)
    y_test_cat = np.clip(np.digitize(y_test, edges[1:-1]), 0, n_bins - 1)
    y_pred_cat = np.clip(np.digitize(y_pred, edges[1:-1]), 0, n_bins - 1)

    cm = confusion_matrix(y_test_cat, y_pred_cat)

    fig4, ax4 = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4,
                xticklabels=[f"bin_{i+1}" for i in range(n_bins)],
                yticklabels=[f"bin_{i+1}" for i in range(n_bins)])
    ax4.set_title("Matriz de Confusión (Redshift en bins)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Predicho"); ax4.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig("outputs/regresion_confusion_matrix.png", dpi=150)
    plt.close()

    metricas = {
        "modelo": "Regresión Lineal", "split": "70/30",
        "MSE": round(float(mse), 8), "RMSE": round(float(rmse), 8),
        "R2": round(float(r2), 6),
        "coeficientes": {f: round(float(c), 6) for f, c in zip(features, coefs)},
        "confusion_matrix_bins": cm.tolist(),
        "bins": [float(x) for x in edges.tolist()]
    }
    with open("outputs/metricas_regresion.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print("      Guardado: outputs/metricas_regresion.json")
    print("      Guardado: outputs/regresion_confusion_matrix.png")
    return metricas
