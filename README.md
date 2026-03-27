# 🔭 Pipeline ML Astronómico SDSS
**Institución Universitaria de Envigado — Big Data**
## INTEGRANTES:
* Cristian David Ocampo Uribe
* Juanita Solórzano Salazar

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange?logo=scikit-learn)
![Jenkins](https://img.shields.io/badge/Jenkins-pipeline-D24939?logo=jenkins)

Pipeline reproducible de Machine Learning sobre datos astronómicos del **Sloan Digital Sky Survey (SDSS)**, con ejecución local, Docker y automatización Jenkins.

---

## 📁 Estructura del Proyecto
```
ml_astronomico/
├── src/
│   ├── main.py            # Orquestador del pipeline
│   ├── preprocessing.py   # Carga y preprocesamiento
│   ├── clasificacion.py   # KNN k=5
│   ├── regresion.py       # Regresión lineal
│   ├── clustering.py      # KMeans k=3
│   └── metricas.py        # Guardado de métricas
├── tests/
│   └── test_dataset.py    # Validación del dataset (24 pruebas)
├── outputs/               # Generado automáticamente
├── Dockerfile
├── Jenkinsfile
├── requirements.txt
└── sdss_sample.csv
```

---

## ⚙️ Ejecución Local
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python tests/test_dataset.py
python src/main.py
```

---

## 🐳 Ejecución con Docker
```bash
docker build -t ml-astronomico .
docker run --rm -v $(pwd)/outputs:/app/outputs ml-astronomico
```

---

## 📊 Resultados

| Modelo | Métrica | Valor |
|--------|---------|-------|
| KNN k=5 | Accuracy | 0.9933 |
| Regresión Lineal | R² | 0.5153 |
| Regresión Lineal | MSE | 0.2984 |
| KMeans k=3 | Silhouette | 0.5406 |

---

## 🤖 Pipeline Jenkins (7 etapas)

1. Checkout del repositorio
2. Instalación de dependencias
3. Validación del dataset (24 pruebas)
4. Ejecución del pipeline ML
5. Build de imagen Docker
6. Ejecución en Docker
7. Archivado de artefactos

---

## 📤 Artefactos generados en `outputs/`

- `resumen_metricas.json` / `.txt`
- `metricas_clasificacion.json`
- `metricas_regresion.json`
- `metricas_clustering.json`
- `test_results.json`
- `clasificacion_confusion_matrix.png`
- `clasificacion_distribucion.png`
- `regresion_diagnostico.png`
- `regresion_coeficientes.png`
- `clustering_comparacion_pca.png`
- `clustering_heatmap_clases.png`
- `clustering_elbow.png`
