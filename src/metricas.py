import json
from datetime import datetime

def guardar_resumen_metricas(metricas_cls, metricas_reg, metricas_clust):
    resumen = {
        "pipeline": "ML Astronómico SDSS — IUE",
        "fecha_ejecucion": datetime.now().isoformat(),
        "clasificacion": {"modelo": metricas_cls.get("modelo"), "accuracy": metricas_cls.get("accuracy")},
        "regresion": {"modelo": metricas_reg.get("modelo"), "MSE": metricas_reg.get("MSE"), "R2": metricas_reg.get("R2")},
        "clustering": {"modelo": metricas_clust.get("modelo"), "silhouette_score": metricas_clust.get("silhouette_score")},
    }
    with open("outputs/resumen_metricas.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    lineas = [
        "=" * 55,
        "  RESUMEN DE MÉTRICAS — PIPELINE ML ASTRONÓMICO",
        "=" * 55,
        f"  Fecha: {resumen['fecha_ejecucion']}",
        "",
        "  CLASIFICACIÓN (KNN k=5)",
        f"    Accuracy:         {metricas_cls.get('accuracy', 'N/A')}",
        "",
        "  REGRESIÓN LINEAL (target: redshift)",
        f"    MSE:              {metricas_reg.get('MSE', 'N/A')}",
        f"    RMSE:             {metricas_reg.get('RMSE', 'N/A')}",
        f"    R²:               {metricas_reg.get('R2', 'N/A')}",
        "",
        "  CLUSTERING KMeans (k=3)",
        f"    Silhouette Score: {metricas_clust.get('silhouette_score', 'N/A')}",
        f"    Inercia (WCSS):   {metricas_clust.get('inercia_wcss', 'N/A')}",
        "=" * 55,
    ]
    with open("outputs/resumen_metricas.txt", "w") as f:
        f.write("\n".join(lineas))
    print("\n".join(lineas))
    print("\n      Guardado: outputs/resumen_metricas.json / .txt")
