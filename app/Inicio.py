# =====================================================
# APLICACIÓN WEB – CHURN PREDICTION INTELIGENTE (Grupo 6)
# Punto de entrada. Ejecutar desde la raíz del proyecto:
#     streamlit run app/Inicio.py
# =====================================================
import streamlit as st

from pathlib import Path

st.set_page_config(page_title="Churn Prediction Inteligente", page_icon="📉", layout="wide")
st.logo(str(Path(__file__).resolve().parent / "logo.svg"), size="large")

paginas = [
    st.Page("vistas/0_Inicio.py", title="Inicio", icon="🏠", default=True),
    st.Page("vistas/1_Prediccion_individual.py", title="Predicción individual", icon="🔮"),
    st.Page("vistas/2_Prediccion_por_lotes.py", title="Predicción por lotes", icon="📂"),
    st.Page("vistas/3_Dashboard_de_riesgo.py", title="Dashboard de riesgo", icon="📊"),
    st.Page("vistas/4_Desempeno_de_modelos.py", title="Desempeño de modelos", icon="🧪"),
]
st.navigation(paginas).run()
