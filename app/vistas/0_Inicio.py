# =====================================================
# INICIO – PANEL EJECUTIVO (Churn Prediction Inteligente · Grupo 6)
# Todas las cifras provienen de los artefactos reales del proyecto:
#   models/metadata.json, data/clientes_scored.csv,
#   reports/curvas_roc.json y reports/shap_importancia_global.csv
# =====================================================
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # carpeta app/ (utils.py)
from utils import (PLOTLY_LAYOUT, REPORTS, RISK_COLORS, fecha_es, icon,  # noqa: E402
                   load_models, load_scored, model_health, sidebar_info, sparkline_svg)

meta, final, todos = load_models()
scored = load_scored()
sidebar_info(meta)

FINAL = meta["modelo_final"]
THR, HIGH = meta["umbral_decision"], meta["umbral_riesgo_alto"]
m = meta["metricas_test"][FINAL]
activos = scored[scored["ChurnReal"] == 0]
n_alto = int((activos["NivelRiesgo"] == "Alto").sum())
miles = lambda n: f"{n:,}".replace(",", ".")  # noqa: E731
BLUE, GREEN, RED, VIOLET, YELLOW = "#2784E8", "#16B978", "#E84B55", "#852DDE", "#F5B51B"

# -----------------------------------------------------
# Encabezado con estado del sistema
# -----------------------------------------------------
ok = model_health()
chips = [("account_tree", "CRISP-DM"), ("model_training", "5 modelos + Stacking"), ("auto_awesome", "SHAP"), ("database", "RFM"),
         ("school", "Grupo 6 - FUP")]
estado = (f"<div class='ok'>{icon('check')}</div><div><b>Modelo operativo</b><small>En producción</small></div>" if ok else
          f"<div class='ok' style='background:{RED}'>{icon('close')}</div><div><b>Modelo no disponible</b>"
          "<small>Revisar los artefactos</small></div>")
st.markdown(
    "<div class='hero2'><div>"
    f"<h1>{icon('bar_chart')}Churn Prediction <em>Inteligente</em></h1>"
    "<p>Plataforma de predicción de abandono de clientes de telecomunicaciones con <i>machine learning</i>, "
    "explicabilidad SHAP y recomendaciones de retención, desarrollada con la metodología CRISP-DM.</p>"
    "<div class='chips'>" + "".join(f"<span class='chip'>{icon(i)}{t}</span>" for i, t in chips) + "</div></div>"
    "<div class='status'>"
    f"<div class='it'>{estado}</div>"
    f"<div class='it'>{icon('calendar_month')}<div><small>Última actualización</small>"
    f"<b>{fecha_es(meta['fecha_entrenamiento'])}</b></div></div>"
    f"<div class='it'>{icon('dataset')}<div><small>Dataset</small><b>Telco Customer Churn (IBM)</b></div></div>"
    "</div></div>", unsafe_allow_html=True)

# -----------------------------------------------------
# KPIs con minigráficas construidas a partir de datos reales
# -----------------------------------------------------
curvas = json.load(open(REPORTS / "curvas_roc.json"))
test = scored[scored["Conjunto"] == "test"]
orden = np.argsort(-test["ProbChurn"].values)
captura = np.cumsum(test["ChurnReal"].values[orden]) / test["ChurnReal"].sum()
gain = captura[np.linspace(0, len(captura) - 1, 25).astype(int)]
tenure_dist = scored["tenure"].value_counts().sort_index().rolling(6, min_periods=1).mean().values
prob_hist = np.histogram(activos["ProbChurn"], bins=24, range=(0, 1))[0]

d_auc, d_rec = m["ROC-AUC"] - 0.84, (m["Recall"] - 0.70) * 100
kpis = [
    ("blue", "group", "Clientes analizados", miles(len(scored)),
     f"<span class='dl neutral'>{miles(len(activos))} activos</span>",
     sparkline_svg(tenure_dist, BLUE, "Distribución de clientes por antigüedad (0–72 meses)")),
    ("green", "track_changes", "ROC-AUC del modelo", f"{m['ROC-AUC']:.3f}",
     f"<span class='dl {'up' if d_auc >= 0 else 'warn'}'>{'↑' if d_auc >= 0 else '↓'} {d_auc:+.3f} vs meta 0,84</span>",
     sparkline_svg(curvas[FINAL]["tpr"], GREEN, "Curva ROC del modelo en el conjunto de prueba")),
    ("red", "warning", "Abandonos detectados (Recall)", f"{m['Recall']:.1%}",
     f"<span class='dl {'up' if d_rec >= 0 else 'warn'}'>{'↑' if d_rec >= 0 else '↓'} {d_rec:+.1f} pp vs meta 70 %</span>",
     sparkline_svg(gain, RED, "Curva de ganancia: % de abandonos capturados al contactar por riesgo")),
    ("violet", "person_alert", "Clientes en riesgo alto", miles(n_alto),
     f"<span class='dl vio'>{n_alto / len(activos):.1%} de los activos</span>",
     sparkline_svg(prob_hist, VIOLET, "Distribución de la probabilidad de abandono de los clientes activos")),
]
st.markdown("<div class='kpis'>" + "".join(
    f"<div class='kpi {c}'><div class='ic'>{icon(i)}</div><div class='t'>{t}</div><div class='vv'>{v}</div>"
    f"<div class='ft'>{d}{sp}</div></div>" for c, i, t, v, d, sp in kpis) + "</div>", unsafe_allow_html=True)


def panel_header(titulo: str, ayuda: str, lead: str | None = None):
    lead_html = icon(lead, "lead") if lead else ""
    st.markdown(f"<div class='ph'>{lead_html}{titulo}<span class='mi' title='{ayuda}'>info</span></div>",
                unsafe_allow_html=True)


# -----------------------------------------------------
# Fila de gráficos
# -----------------------------------------------------
g1, g2 = st.columns([1, 1.35], gap="medium")
with g1:
    with st.container(key="panel_riesgo"):
        panel_header("Distribución de clientes por nivel de riesgo",
                     f"Clasificación del modelo: Bajo < {THR:.0%} ≤ Medio < {HIGH:.0%} ≤ Alto")
        poblacion = st.segmented_control("Población", ["Clientes activos", "Toda la cartera"],
                                         default="Clientes activos", key="pob_riesgo", label_visibility="collapsed")
        base = activos if poblacion != "Toda la cartera" else scored
        cnt = base["NivelRiesgo"].value_counts().reindex(["Alto", "Medio", "Bajo"]).fillna(0).astype(int)
        c_a, c_b = st.columns([1.05, 1])
        with c_a:
            fig = go.Figure(go.Pie(
                values=cnt.values, labels=[f"Riesgo {k.lower()}" for k in cnt.index], hole=0.7, sort=False,
                direction="clockwise", marker=dict(colors=[RISK_COLORS[k] for k in cnt.index], line=dict(color="#fff", width=3)),
                textinfo="none", hovertemplate="%{label}: %{value:,} clientes (%{percent})<extra></extra>"))
            fig.update_layout(**{**PLOTLY_LAYOUT, "margin": dict(l=0, r=0, t=0, b=0)}, height=215, showlegend=False,
                              annotations=[dict(text=f"<b style='font-size:26px'>{miles(cnt.sum())}</b>"
                                                     f"<br><span style='font-size:13px'>Clientes</span>",
                                                showarrow=False, x=0.5, y=0.5)])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with c_b:
            st.markdown("<div class='lg'>" + "".join(
                f"<div class='r'><i style='background:{RISK_COLORS[k]}'></i><span>Riesgo {k.lower()}</span>"
                f"<span><b>{miles(v)}</b> <small>({v / cnt.sum():.1%})</small></span></div>"
                for k, v in cnt.items()) + "</div>", unsafe_allow_html=True)

with g2:
    with st.container(key="panel_comportamiento"):
        h_a, h_b = st.columns([1.6, 1])
        with h_a:
            panel_header("Comportamiento del abandono",
                         "El dataset no tiene fechas: el eje es la antigüedad del cliente (meses desde que se afilió). "
                         "Abandonos = casos reales del histórico; en riesgo = probabilidad ≥ umbral de decisión.")
        ventanas = {"Primer año": (12, 1), "Primeros 2 años": (24, 2), "Ciclo completo": (72, 6)}
        with h_b:
            ventana = st.selectbox("Ventana", list(ventanas), index=2, key="ventana_abandono",
                                   label_visibility="collapsed")
        limite, paso = ventanas[ventana]
        t = scored[scored["tenure"] <= limite].copy()
        t["bin"] = (t["tenure"].clip(lower=1) - 1) // paso
        agg = t.groupby("bin").agg(riesgo=("ProbChurn", lambda s: int((s >= THR).sum())),
                                   abandonos=("ChurnReal", "sum")).reindex(range(limite // paso), fill_value=0)
        etiquetas = [f"M{i + 1}" if paso == 1 else f"{i * paso + 1}–{(i + 1) * paso}" for i in agg.index]
        fig = go.Figure()
        for col, nombre, color, fill in [("riesgo", "Clientes en riesgo", BLUE, "rgba(39,132,232,.14)"),
                                         ("abandonos", "Abandonos", VIOLET, "rgba(133,45,222,.10)")]:
            fig.add_trace(go.Scatter(x=etiquetas, y=agg[col], name=nombre, mode="lines+markers",
                                     line=dict(color=color, width=2.6, shape="spline", smoothing=0.6),
                                     marker=dict(size=8, color="#fff", line=dict(color=color, width=2.4)),
                                     fill="tozeroy", fillcolor=fill,
                                     hovertemplate=f"{nombre}: %{{y:,}}<extra>%{{x}}</extra>"))
        fig.update_layout(**{**PLOTLY_LAYOUT, "margin": dict(l=6, r=6, t=28, b=6)}, height=245, hovermode="x unified",
                          legend=dict(orientation="h", x=1, xanchor="right", y=1.14, font=dict(size=12)),
                          yaxis=dict(gridcolor="rgba(128,145,170,.18)", zeroline=False),
                          xaxis=dict(showgrid=False, tickangle=0, tickfont=dict(size=11)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# -----------------------------------------------------
# Fila de inteligencia analítica
# -----------------------------------------------------
p1, p2, p3 = st.columns([1, 1.05, 1], gap="medium")
with p1:
    with st.container(key="panel_shap"):
        panel_header("Top factores de abandono (SHAP)",
                     f"Importancia global: media del |valor SHAP| por variable original, modelo {FINAL}, conjunto de prueba")
        imp = pd.read_csv(REPORTS / "shap_importancia_global.csv", index_col=0)["mean_abs_shap"].head(6)
        st.markdown("<div class='shp'>" + "".join(
            f"<div class='r'><span>{nombre}</span><div class='tr'><div class='br' style='width:{v / imp.max() * 100:.0f}%'></div></div>"
            f"<b>{v:.2f}</b></div>" for nombre, v in imp.items()) + "</div>"
            f"<div class='shp-note'>Valores en {meta.get('shap_unidad', 'log-odds')} · fuente: análisis SHAP del modelo en producción.</div>",
            unsafe_allow_html=True)

with p2:
    with st.container(key="panel_prioritarios"):
        panel_header("Clientes prioritarios para retención",
                     "Clientes activos ordenados por probabilidad de abandono. Haz clic en una columna para ordenar.")
        buscar = st.text_input("Buscar", placeholder="Buscar ID de cliente…", key="buscar_id",
                               label_visibility="collapsed")
        tabla = (activos.sort_values("ProbChurn", ascending=False)[["customerID", "NivelRiesgo", "ProbChurn"]]
                 .rename(columns={"customerID": "ID Cliente", "NivelRiesgo": "Riesgo", "ProbChurn": "Prob. abandono"}))
        if buscar:
            tabla = tabla[tabla["ID Cliente"].str.contains(buscar.strip(), case=False, na=False)]
        # Fondos semitransparentes: se leen bien en tema claro y oscuro
        estilos = {"Alto": "background-color:rgba(232,75,85,.18);color:#d63a45;font-weight:600",
                   "Medio": "background-color:rgba(245,181,27,.20);color:#b47d00;font-weight:600",
                   "Bajo": "background-color:rgba(22,185,120,.18);color:#119a64;font-weight:600"}
        st.dataframe(tabla.head(200).style.map(lambda v: estilos.get(v, ""), subset=["Riesgo"])
                     .format({"Prob. abandono": "{:.2f}"}),
                     hide_index=True, width="stretch", height=230)

with p3:
    with st.container(key="panel_reco"):
        panel_header("Recomendación inteligente",
                     "Acciones del motor de recomendaciones del proyecto (reglas de negocio ordenadas por SHAP), "
                     "contadas sobre los clientes activos en riesgo alto.", lead="auto_awesome")
        alto = activos[activos["NivelRiesgo"] == "Alto"]
        acciones = alto["Recomendaciones"].dropna().str.split(" | ", regex=False).explode().value_counts().head(3)
        items = "".join(f"<li>{icon('check_circle')}<span>{a}<br><small>{miles(int(c))} clientes "
                        f"({c / max(len(alto), 1):.0%})</small></span></li>" for a, c in acciones.items())
        st.markdown(
            f"<div class='reco'><div class='tg'>{icon('target')}</div><div>"
            f"<p>Se recomienda priorizar <b>{miles(n_alto)} clientes</b> en riesgo alto con campañas de retención "
            f"personalizadas, como:</p><ul>{items}</ul></div></div>"
            "<div class='reco-note'>Las 3 acciones más frecuentes entre las recomendaciones personalizadas de esos clientes.</div>",
            unsafe_allow_html=True)
        try:
            st.page_link("vistas/3_Dashboard_de_riesgo.py", label="Ver lista de campaña", icon=":material/arrow_forward:")
        except Exception:  # fuera del menú de navegación (pruebas)
            pass
