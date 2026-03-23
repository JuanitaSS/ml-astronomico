import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

os.makedirs("outputs", exist_ok=True)

from preprocessing import cargar_datos, preprocesar
from clasificacion import ejecutar_clasificacion
from regresion import ejecutar_regresion
from clustering import ejecutar_clustering
from metricas import guardar_resumen_metricas

def main():
    print("=" * 60)
    print("  PIPELINE ML - DATOS ASTRONÓMICOS SDSS")
    print("=" * 60)

    print("\n[1/4] Cargando y preprocesando datos...")
    df = cargar_datos("sdss_sample.csv")
    X_cls, y_cls, X_reg, y_reg, X_clust, scaler = preprocesar(df)
    print(f"      Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")

    print("\n[2/4] Ejecutando clasificación (KNN k=5)...")
    metricas_cls = ejecutar_clasificacion(X_cls, y_cls)

    print("\n[3/4] Ejecutando regresión lineal (target: redshift)...")
    metricas_reg = ejecutar_regresion(X_reg, y_reg)

    print("\n[4/4] Ejecutando KMeans (k=3)...")
    metricas_clust = ejecutar_clustering(X_clust, df)

    guardar_resumen_metricas(metricas_cls, metricas_reg, metricas_clust)

    print("\n" + "=" * 60)
    print("  Pipeline completado. Resultados en outputs/")
    print("=" * 60)

if __name__ == "__main__":
    main()
