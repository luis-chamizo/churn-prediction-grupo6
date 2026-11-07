# =====================================================
# APLICACIÓN WEB – CHURN PREDICTION INTELIGENTE (Grupo 6)
# Ejecutar desde la raíz del proyecto:  streamlit run app/Inicio.py
# =====================================================
import streamlit as st



import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # carpeta app/ (utils.py)
from utils import load_models, load_scored, sidebar_info  # noqa: E402

meta, final, todos = load_models()
scored = load_scored()
sidebar_info(meta)

st.title("📉 Churn Prediction Inteligente")
st.markdown(
    "Sistema de **predicción de abandono de clientes** de telecomunicaciones basado en *machine learning*, "
    "con **explicabilidad (SHAP)** y **recomendaciones de retención personalizadas**. "
    "Desarrollado con la metodología **CRISP-DM** sobre el dataset *Telco Customer Churn* (IBM / Kaggle)."
)
st.caption("Proyecto Final #1 · Minería de Datos · Grupo 6: Luis Mario Chamizo Tiafi y Juan Diego Chamizo Tiafi")

m = meta["metricas_test"][meta["modelo_final"]]
activos = scored[scored["ChurnReal"] == 0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo en producción", meta["modelo_final"])
c2.metric("ROC-AUC (test)", f"{m['ROC-AUC']:.3f}",
          help=f"IC 95 % bootstrap: {meta['auc_ic95_final'][0]:.3f} – {meta['auc_ic95_final'][1]:.3f}")
c3.metric("Recall churn (test)", f"{m['Recall']:.1%}", help="Proporción de clientes que abandonan detectados por el modelo")
c4.metric("Clientes activos en riesgo alto", f"{(activos['NivelRiesgo'] == 'Alto').sum():,}",
          help=f"Probabilidad ≥ {meta['umbral_riesgo_alto']:.0%}")

st.divider()
st.subheader("¿Qué puedes hacer en esta aplicación?")
a, b = st.columns(2)
with a:
    st.markdown("""
**🔮 Predicción individual**: ingresa los datos de un cliente (o elige uno existente) y obtén:
- la probabilidad de abandono y su nivel de riesgo,
- los factores que explican la predicción (SHAP),
- recomendaciones de retención priorizadas,
- una simulación de cuánto bajaría el riesgo con cada acción.

**📂 Predicción por lotes**: sube un CSV con muchos clientes y descarga los resultados.
""")
with b:
    st.markdown("""
**📊 Dashboard de riesgo**: cartera completa con filtros, clientes en riesgo, matriz **riesgo × valor**
para priorizar la inversión y descarga de listas de campaña.

**🧪 Desempeño de modelos**: comparación de los 6 modelos (Regresión Logística, Random Forest,
XGBoost, LightGBM, MLP y Stacking) con Accuracy, F1, ROC-AUC, curvas ROC, matrices de confusión
e importancia global SHAP.
""")

st.divider()
st.subheader("Metodología CRISP-DM")
fases = [
    ("1. Negocio", "Reducir el churn priorizando clientes por riesgo × valor. Criterios: ROC-AUC ≥ 0.84 y Recall ≥ 0.70."),
    ("2. Datos", "7 043 clientes, 21 variables. EDA, calidad (11 vacíos en TotalCharges) y pruebas χ² / Mann-Whitney."),
    ("3. Preparación", "Limpieza, RFM adaptado (antigüedad, servicios, gasto), 16 variables derivadas, one-hot y escalado en un Pipeline."),
    ("4. Modelado", "5 modelos + Stacking, búsqueda aleatoria de hiperparámetros con CV estratificada de 5 pliegues."),
    ("5. Evaluación", "Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, curva de ganancia, costo-beneficio y SHAP."),
    ("6. Despliegue", "Esta aplicación Streamlit con pipeline serializado (joblib) y plan de monitoreo."),
]
cols = st.columns(3)
for i, (t, d) in enumerate(fases):
    with cols[i % 3]:
        st.markdown(f"**{t}**")
        st.caption(d)

st.info("Usa el menú lateral para navegar entre las páginas.", icon="👈")
