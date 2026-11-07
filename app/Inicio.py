# =====================================================
# APLICACIÓN WEB – CHURN PREDICTION INTELIGENTE (Grupo 6)
# Punto de entrada. Ejecutar desde la raíz del proyecto:
#     streamlit run app/Inicio.py
# =====================================================
import streamlit as st

from pathlib import Path

from estilo import apply_style  # módulo liviano: no espera a sklearn/xgboost

st.set_page_config(page_title="Churn Prediction Inteligente", page_icon=":material/insights:", layout="wide")
APP = Path(__file__).resolve().parent
st.logo(str(APP / "logo.svg"), size="large", icon_image=str(APP / "logo_icono.svg"))
# El CSS se inyecta aquí, antes de ejecutar cualquier página: así está presente desde
# el primer instante y no desaparece al navegar mientras la página nueva carga.
apply_style()

paginas = [
    st.Page("vistas/0_Inicio.py", title="Inicio", icon=":material/space_dashboard:", default=True),
    st.Page("vistas/1_Prediccion_individual.py", title="Predicción individual", icon=":material/person_search:"),
    st.Page("vistas/2_Prediccion_por_lotes.py", title="Predicción por lotes", icon=":material/upload_file:"),
    st.Page("vistas/3_Dashboard_de_riesgo.py", title="Dashboard de riesgo", icon=":material/monitoring:"),
    st.Page("vistas/4_Desempeno_de_modelos.py", title="Desempeño de modelos", icon=":material/analytics:"),
]
st.navigation(paginas).run()
