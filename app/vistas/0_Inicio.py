# =====================================================
# PÁGINA DE INICIO – CHURN PREDICTION INTELIGENTE (Grupo 6)
# =====================================================
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # carpeta app/ (utils.py)
from utils import footer, load_models, load_scored, page_header, section, sidebar_info  # noqa: E402

meta, final, todos = load_models()
scored = load_scored()
sidebar_info(meta)

m = meta["metricas_test"][meta["modelo_final"]]
activos = scored[scored["ChurnReal"] == 0]

page_header(
    "📉", "Churn Prediction Inteligente",
    "Plataforma de <b>predicción de abandono de clientes</b> de telecomunicaciones con <i>machine learning</i>, "
    "<b>explicabilidad SHAP</b> y <b>recomendaciones de retención personalizadas</b>, desarrollada con la "
    "metodología CRISP-DM sobre el dataset <i>Telco Customer Churn</i>.",
    chips=["CRISP-DM", "5 modelos + Stacking", "XGBoost en producción", "SHAP", "RFM", "Grupo 6 · FUP"],
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes analizados", f"{len(scored):,}".replace(",", "."))
c2.metric("ROC-AUC del modelo", f"{m['ROC-AUC']:.3f}",
          help=f"Conjunto de prueba. IC 95 % bootstrap: {meta['auc_ic95_final'][0]:.3f} – {meta['auc_ic95_final'][1]:.3f}")
c3.metric("Abandonos detectados (Recall)", f"{m['Recall']:.1%}",
          help="Proporción de clientes que abandonan que el modelo identifica")
c4.metric("Clientes activos en riesgo alto", f"{(activos['NivelRiesgo'] == 'Alto').sum():,}",
          help=f"Probabilidad de abandono ≥ {meta['umbral_riesgo_alto']:.0%}")

section("Módulos de la plataforma")
modulos = [
    ("🔮", "Predicción individual", "Probabilidad de abandono de un cliente, explicación SHAP, recomendaciones "
     "priorizadas y simulador de acciones de retención.", "vistas/1_Prediccion_individual.py"),
    ("📂", "Predicción por lotes", "Sube un CSV con muchos clientes, valida el formato y descarga las "
     "predicciones con factores y recomendaciones.", "vistas/2_Prediccion_por_lotes.py"),
    ("📊", "Dashboard de riesgo", "Cartera completa con filtros, matriz riesgo × valor, plan de acción y lista "
     "descargable de clientes para campañas.", "vistas/3_Dashboard_de_riesgo.py"),
    ("🧪", "Desempeño de modelos", "Comparación de los 6 modelos: métricas, curvas ROC, matrices de confusión "
     "e importancia global SHAP.", "vistas/4_Desempeno_de_modelos.py"),
]
cols = st.columns(4)
for col, (ico, titulo, desc, pagina) in zip(cols, modulos):
    with col:
        st.markdown(f"<div class='feature'><div class='ico'>{ico}</div><h4>{titulo}</h4><p>{desc}</p></div>",
                    unsafe_allow_html=True)
        try:
            st.page_link(pagina, label=f"Abrir {titulo.lower()}", icon="➡️")
        except Exception:  # la página se ejecuta fuera del menú de navegación (p. ej. en pruebas)
            pass

section("Metodología CRISP-DM")
fases = [
    ("Negocio", "Reducir el churn priorizando por riesgo × valor. Meta: ROC-AUC ≥ 0,84 y Recall ≥ 0,70."),
    ("Datos", "7 043 clientes y 21 variables. EDA, calidad de datos y pruebas χ² / Mann-Whitney."),
    ("Preparación", "Limpieza, RFM adaptado y 16 variables derivadas en un Pipeline sin fuga de datos."),
    ("Modelado", "LR, Random Forest, XGBoost, LightGBM, MLP y Stacking con CV estratificada de 5 pliegues."),
    ("Evaluación", "Accuracy, F1, ROC-AUC, curva de ganancia, costo-beneficio y explicabilidad SHAP."),
    ("Despliegue", "Esta aplicación web en Streamlit Cloud, con plan de monitoreo y reentrenamiento."),
]
st.markdown("<div class='steps'>" + "".join(
    f"<div class='step'><div class='n'>{i}</div><b>{t}</b><p>{d}</p></div>" for i, (t, d) in enumerate(fases, 1)
) + "</div>", unsafe_allow_html=True)

section("Hallazgos clave")
h1, h2, h3 = st.columns(3)
for col, (ico, titulo, texto) in zip((h1, h2, h3), [
    ("📄", "Compromiso contractual", "El contrato mes a mes tiene 42,7 % de abandono frente a 2,8 % en contratos de 2 años."),
    ("⏳", "Primeros meses críticos", "Los clientes que se van llevan una mediana de 10 meses; el riesgo cae tras el primer año."),
    ("🛡️", "Servicios de protección", "Sin seguridad en línea ni soporte técnico el abandono sube a ~42 %."),
]):
    col.markdown(f"<div class='feature' style='min-height:130px'><div class='ico'>{ico}</div><h4>{titulo}</h4>"
                 f"<p>{texto}</p></div>", unsafe_allow_html=True)

footer()
