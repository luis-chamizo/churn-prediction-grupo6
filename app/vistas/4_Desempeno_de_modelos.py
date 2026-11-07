import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st



import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # carpeta app/ (utils.py)
from utils import (page_header, section, C_NO, FIGURES, MODEL_COLORS, PLOTLY_LAYOUT,  # noqa: E402
                   REPORTS, load_models, sidebar_info)

meta, _, _ = load_models()
sidebar_info(meta)
nombres = list(meta["archivos_modelos"].keys())
color = {n: MODEL_COLORS[i] for i, n in enumerate(nombres)}

page_header("analytics", "Desempeño y comparación de modelos",
            f"5 modelos requeridos + 1 ensamble Stacking, ajustados con búsqueda aleatoria y validación cruzada "
            f"estratificada de 5 pliegues. Entrenamiento: {meta['n_train']:,} clientes · prueba: {meta['n_test']:,} clientes.",
            chips=["Accuracy · F1 · ROC-AUC", "Curvas ROC", "Matrices de confusión", "SHAP"])

# -----------------------------------------------------
# Tabla de métricas
# -----------------------------------------------------
section("Métricas en el conjunto de prueba (umbral óptimo por modelo)", "table_chart")
met = pd.DataFrame(meta["metricas_test"]).T
met.insert(0, "Umbral", pd.Series(meta["umbrales_por_modelo"]))
met = met[["Umbral", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier", "roc_auc_cv"]]
met = met.rename(columns={"roc_auc_cv": "ROC-AUC (CV)"}).sort_values("ROC-AUC", ascending=False)
# Fondo claro + texto oscuro fijo: legible tanto en tema claro como en oscuro
resaltado = "background-color:#cde2fb;color:#0b2a4a;font-weight:700"
st.dataframe(met.style.format("{:.4f}")
             .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "ROC-AUC (CV)"], props=resaltado)
             .highlight_min(subset=["Brier"], props=resaltado),
             width="stretch")
st.caption("Umbral elegido maximizando F1 sobre predicciones out-of-fold del entrenamiento (sin usar el test). "
           "Celdas resaltadas = mejor valor por métrica. Brier: menor es mejor (calibración).")

# -----------------------------------------------------
# ROC y CV
# -----------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    curvas = json.load(open(REPORTS / "curvas_roc.json"))
    fig = go.Figure()
    for n, c in curvas.items():
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=f"{n} ({meta['metricas_test'][n]['ROC-AUC']:.3f})",
                                 line=dict(width=2, color=color[n]),
                                 hovertemplate=f"{n}<br>FPR %{{x:.2f}} · TPR %{{y:.2f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#898781", width=1),
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(**PLOTLY_LAYOUT, height=430, title="Curvas ROC (test)", xaxis_title="Tasa de falsos positivos",
                      yaxis_title="Tasa de verdaderos positivos", legend=dict(x=0.45, y=0.05, font_size=11))
    st.plotly_chart(fig, width="stretch")
with c2:
    cv = pd.read_csv(REPORTS / "validacion_cruzada.csv")
    metrica = st.radio("Métrica de validación cruzada", ["ROC-AUC", "F1", "Accuracy"], horizontal=True)
    fig = px.box(cv, x=metrica, y="modelo", color="modelo", color_discrete_map=color, points="all",
                 category_orders={"modelo": nombres})
    fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False, title=f"{metrica} en los 5 pliegues (umbral 0.5)",
                      yaxis_title=None)
    st.plotly_chart(fig, width="stretch")

# -----------------------------------------------------
# Matriz de confusión
# -----------------------------------------------------
c1, c2 = st.columns([1, 1.3])
with c1:
    section("Matriz de confusión", "grid_view")
    mats = json.load(open(REPORTS / "matrices_confusion.json"))
    sel = st.selectbox("Modelo", nombres, index=nombres.index(meta["modelo_final"]))
    cm = mats[sel]
    etiquetas = ["No abandona", "Abandona"]
    fig = go.Figure(go.Heatmap(z=cm, x=etiquetas, y=etiquetas, colorscale=[[0, "#f4f8fd"], [1, "#1c5cab"]],
                               showscale=False, text=[[f"{v:,}" for v in r] for r in cm], texttemplate="%{text}",
                               textfont=dict(size=18), hovertemplate="Real %{y} · Pred %{x}: %{z}<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="Predicción", yaxis_title="Real",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    st.caption(f"Detecta {tp} de {tp + fn} clientes que abandonan (Recall {tp / (tp + fn):.1%}). "
               f"{fp} falsas alarmas sobre {tn + fp} clientes que se quedan.")
with c2:
    section("Importancia global (SHAP)", "psychology")
    imp = pd.read_csv(REPORTS / "shap_importancia_global.csv", index_col=0).head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=imp.index, orientation="h", marker_color=C_NO,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_title=f"mean(|SHAP|) en {meta.get('shap_unidad', 'log-odds')}",
                      title=f"Modelo {meta['modelo_final']} · 15 variables más influyentes")
    st.plotly_chart(fig, width="stretch")

section("Visualizaciones de explicabilidad y negocio", "insights")
t1, t2, t3, t4 = st.tabs(["SHAP beeswarm", "Dependencia SHAP", "Curva de ganancia", "Costo-beneficio"])
for tab, f in zip([t1, t2, t3, t4], ["13_shap_beeswarm", "15_shap_dependencia", "11_curva_ganancia", "12_costo_beneficio"]):
    with tab:
        st.image(str(FIGURES / f"{f}.png"), width="stretch")

with st.expander("Hiperparámetros óptimos encontrados"):
    hp = pd.DataFrame(meta["busqueda_hiperparametros"])
    hp["mejores_hiperparametros"] = hp["mejores_hiperparametros"].astype(str)
    st.dataframe(hp, hide_index=True, width="stretch")
st.caption(f"Versiones: {', '.join(f'{k} {v}' for k, v in meta['versiones'].items())} · Entrenado: {meta['fecha_entrenamiento']}")
