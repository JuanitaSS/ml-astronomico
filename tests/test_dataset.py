import sys, os, json
import pandas as pd
import numpy as np

DATASET_PATH        = "sdss_sample.csv"
COLUMNAS_REQUERIDAS = ["u", "g", "r", "i", "z", "redshift", "class"]
CLASES_VALIDAS      = {"GALAXY", "STAR", "QSO"}
RESULTS             = []

def check(nombre, condicion, detalle=""):
    estado = "PASS" if condicion else "FAIL"
    RESULTS.append({"test": nombre, "estado": estado, "detalle": detalle})
    print(f"  [{'✓' if condicion else '✗'}] {nombre:<45} {estado}  {detalle}")
    return condicion

def main():
    print("\n" + "=" * 60)
    print("  PRUEBAS BÁSICAS DEL DATASET SDSS")
    print("=" * 60)

    if not os.path.isfile(DATASET_PATH):
        check("Archivo sdss_sample.csv existe", False, "No encontrado")
        guardar_resultados()
        sys.exit(1)

    check("Archivo sdss_sample.csv existe", True)
    df = pd.read_csv(DATASET_PATH)

    df["class"] = df["class"].str.upper().str.strip()

    faltantes = set(COLUMNAS_REQUERIDAS) - set(df.columns)
    check("Columnas requeridas presentes", not faltantes,
          str(faltantes) if faltantes else "OK")
    check("Dataset tiene >= 100 filas", len(df) >= 100, f"n={len(df)}")

    for col in COLUMNAS_REQUERIDAS:
        nulos = df[col].isnull().sum()
        check(f"Sin nulos en '{col}'", nulos == 0, f"{nulos} nulos")

    for col in ["u","g","r","i","z","redshift"]:
        check(f"'{col}' es numérica", pd.api.types.is_numeric_dtype(df[col]),
              str(df[col].dtype))

    clases_en_datos = set(df["class"].unique())
    check("Clases válidas {GALAXY,STAR,QSO}",
          clases_en_datos.issubset(CLASES_VALIDAS), str(clases_en_datos))
    check("Al menos 2 clases presentes", df["class"].nunique() >= 2,
          f"{df['class'].nunique()} clases")

    for col in ["u","g","r","i","z"]:
        ok = df[col].between(-5, 35).all()
        check(f"'{col}' en rango [-5, 35]", ok,
              f"min={df[col].min():.2f} max={df[col].max():.2f}")

    check("Redshift >= -0.1", (df["redshift"] >= -0.1).all())

    guardar_resultados()
    passed = sum(1 for r in RESULTS if r["estado"] == "PASS")
    failed = len(RESULTS) - passed
    print(f"\n  Resultado: {passed}/{len(RESULTS)} pruebas pasadas")
    print("=" * 60 + "\n")
    if failed > 0:
        sys.exit(1)

def guardar_resultados():
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/test_results.json", "w") as f:
        json.dump({"tests": RESULTS}, f, indent=2)
    print(f"\n  Guardado: outputs/test_results.json")

if __name__ == "__main__":
    main()
