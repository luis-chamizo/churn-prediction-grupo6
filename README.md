# Churn Prediction Inteligente – Telco Customer Churn

**Proyecto Final #1 – Minería de Datos (metodología CRISP-DM)**
**Grupo 6:** Luis Mario Chamizo Tiafi · Juan Diego Chamizo Tiafi

Sistema de predicción de abandono de clientes de telecomunicaciones que:
- compara **5 modelos de machine learning** (Regresión Logística, Random Forest, XGBoost, LightGBM, Red Neuronal MLP) y un **Stacking** adicional;
- usa **feature engineering RFM** adaptado a suscripciones y otras 16 variables derivadas;
- se evalúa con **Accuracy, Precision, Recall, F1, ROC-AUC y PR-AUC**, además de la curva de ganancia y un análisis costo-beneficio;
- explica cada predicción con **SHAP**;
- se despliega como una **aplicación web Streamlit** con predicción individual y por lotes, **dashboard de clientes en riesgo** y **recomendaciones de retención personalizadas** (con simulador "¿qué pasaría si…?").

## Estructura

```
Proyecto_Churn_Grupo6/
├── app/                         # Aplicación web (Streamlit)
│   ├── Inicio.py                #   página principal
│   ├── utils.py                 #   carga de modelos, SHAP, simulador
│   └── vistas/                  #   inicio, predicción individual, lotes, dashboard, desempeño
├── src/features.py              # Limpieza, FeatureEngineer (RFM), preprocesamiento, recomendaciones
├── notebooks/
│   ├── Churn_Prediction_CRISP_DM.ipynb   # Proceso completo CRISP-DM (ejecutado)
│   └── build_notebook.py                 # Script que genera el notebook
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset original (Kaggle)
│   └── clientes_scored.csv                    # Cartera puntuada (alimenta el dashboard)
├── models/                      # Pipelines entrenados (.joblib) + metadata.json
├── reports/                     # Métricas, validación cruzada, curvas y figuras
├── docs/                        # Documento técnico formato IEEE (.docx y .pdf)
├── requirements.txt
├── Dockerfile
└── .streamlit/config.toml
```

## Ejecución local

```bash
# 1. Crear entorno e instalar dependencias (Python 3.11 – 3.14)
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# 2. (Opcional) Reentrenar todo: ejecuta el notebook completo y regenera models/, reports/ y data/clientes_scored.csv
jupyter nbconvert --to notebook --execute --inplace notebooks/Churn_Prediction_CRISP_DM.ipynb

# 3. Lanzar la aplicación web
streamlit run app/Inicio.py
```
La aplicación abre en http://localhost:8501.

## Despliegue en la nube

### Opción A – Streamlit Community Cloud (gratis, recomendado)
1. Sube esta carpeta a un repositorio **público** de GitHub (los modelos pesan menos de 100 MB).
2. Entra a https://share.streamlit.io, inicia sesión con GitHub y elige **Create app → Deploy a public app from GitHub**.
3. Selecciona el repositorio, la rama `main` y como *Main file path* escribe `app/Inicio.py`.
4. En **Advanced settings** elige Python **3.12** y pulsa **Deploy**. En unos minutos tendrás una URL pública del tipo `https://<nombre>.streamlit.app`.

### Opción B – Docker (Render, Railway, Hugging Face Spaces, Cloud Run)
```bash
docker build -t churn-grupo6 .
docker run -p 8501:8501 churn-grupo6
```
En Render: *New → Web Service → Docker*. El contenedor usa la variable `PORT` si el servicio la define.

> Los modelos están serializados con las versiones exactas de `requirements.txt`. Si cambias las versiones de scikit-learn, XGBoost o LightGBM, vuelve a ejecutar el notebook para regenerar los `.joblib`.

## Resultados principales (conjunto de prueba, 1 409 clientes)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| LightGBM | 0.756 | 0.527 | 0.781 | 0.629 | 0.848 |
| Stacking | 0.774 | 0.556 | 0.735 | 0.633 | 0.848 |
| **XGBoost (en producción)** | 0.762 | 0.536 | 0.775 | **0.634** | 0.847 |
| Regresión Logística | 0.769 | 0.549 | 0.725 | 0.624 | 0.846 |
| Random Forest | 0.779 | 0.571 | 0.682 | 0.621 | 0.845 |
| Red Neuronal (MLP) | 0.770 | 0.552 | 0.714 | 0.622 | 0.834 |

- Métricas calculadas con el umbral óptimo de cada modelo (máximo F1 en predicciones out-of-fold).
- **XGBoost** se seleccionó con una regla definida antes de mirar el test: mejor ROC-AUC en validación cruzada (0.850) entre los modelos con explicación SHAP exacta.
- Contactando al 30 % de clientes de mayor riesgo se captura el **66.6 %** de los abandonos.
- Factores principales según SHAP: duración del contrato, antigüedad, cargo por servicio, método de pago, tipo de contrato y seguridad en línea.

El detalle completo está en `reports/`, en la página **Desempeño de modelos** de la app y en el documento técnico de `docs/`.
