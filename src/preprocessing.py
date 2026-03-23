import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

FEATURES_FOTO = ["u", "g", "r", "i", "z"]
FEATURES_CLS  = ["u", "g", "r", "i", "z", "redshift"]
TARGET_CLS    = "class"
TARGET_REG    = "redshift"

def cargar_datos(ruta: str) -> pd.DataFrame:
    if not os.path.isfile(ruta):
        ruta_alternativa = os.path.join(os.path.dirname(__file__), "..", ruta)
        ruta = os.path.normpath(ruta_alternativa)

    df = pd.read_csv(ruta)

    df[TARGET_CLS] = df[TARGET_CLS].str.upper().str.strip()

    columnas_requeridas = set(FEATURES_FOTO + [TARGET_CLS, TARGET_REG])
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes: {faltantes}")

    df = df.dropna(subset=FEATURES_CLS + [TARGET_CLS])
    print(f"      Clases detectadas: {df[TARGET_CLS].unique().tolist()}")
    print(f"      Distribución:\n{df[TARGET_CLS].value_counts().to_string()}")
    return df

def preprocesar(df: pd.DataFrame):
    
    os.makedirs("outputs", exist_ok=True)

    scaler = StandardScaler()

    X_cls_raw = df[FEATURES_CLS].values
    le = LabelEncoder()
    y_cls = le.fit_transform(df[TARGET_CLS].values)
    X_cls = scaler.fit_transform(X_cls_raw)
    np.save("outputs/label_encoder_classes.npy", le.classes_)

    X_reg_raw = df[FEATURES_FOTO].values
    y_reg = df[TARGET_REG].values
    scaler_reg = StandardScaler()
    X_reg = scaler_reg.fit_transform(X_reg_raw)

    X_clust_raw = df[FEATURES_FOTO].values
    scaler_clust = StandardScaler()
    X_clust = scaler_clust.fit_transform(X_clust_raw)

    return X_cls, y_cls, X_reg, y_reg, X_clust, scaler
