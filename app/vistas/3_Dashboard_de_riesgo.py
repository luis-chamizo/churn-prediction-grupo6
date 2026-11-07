import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st



import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # carpeta app/ (utils.py)
from utils import (C_NO, PLOTLY_LAYOUT, PRIORITY_COLORS, RISK_COLORS,  # noqa: E402
                   load_models, load_scored, sidebar_info)

meta, _, _ = load_models()
data = load_scored()
sidebar_info(meta)
THR, HIGH = meta["umbral_decision"], meta["umbral_riesgo_alto"]
NIVELES = ["Bajo", "Medio", "Alto"]

st.title("📊 Dashboard de clientes en riesgo")

# -----------------------------------------------------
# Filtros (una sola fila)
# -----------------------------------------------------
f1, f2, f3, f4, f5 = st.columns([1.1, 1, 1, 1, 1.2])
solo_activos = f1.toggle("Solo clientes activos", value=True,
                         help="Excluye a los clientes que ya abandonaron (Churn = Yes en el histórico)")
contrato = f2.multiselect("Contrato", sorted(data.Contract.unique()), placeholder="Todos")
internet = f3.multiselect("Internet", sorted(data.InternetService.unique()), placeholder="Todos")
niveles = f4.multiselect("Nivel de riesgo", NIVELES, placeholder="Todos")
segmento = f5.multiselect("Segmento RFM", sorted(data.SegmentoRFM.unique()), placeholder="Todos")

df = data.copy()
if solo_activos:
    df = df[df.ChurnReal == 0]
for col, sel in [("Contract", contrato), ("InternetService", internet), ("NivelRiesgo", niveles), ("SegmentoRFM", segmento)]:
    if sel:
        df = df[df[col].isin(sel)]

if df.empty:
    st.warning("No hay clientes con esos filtros.")
    st.stop()

# -----------------------------------------------------
# KPIs
# -----------------------------------------------------
alto = df[df.NivelRiesgo == "Alto"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clientes", f"{len(df):,}")
k2.metric("🔴 Riesgo alto", f"{len(alto):,}", help=f"{len(alto) / len(df):.1%} de los clientes filtrados")
k3.metric("🟡 Riesgo medio", f"{(df.NivelRiesgo == 'Medio').sum():,}")
k4.metric("Prob. media de churn", f"{df.ProbChurn.mean():.1%}")
k5.metric("Ingreso anual en riesgo", f"USD {df.PerdidaEsperada.sum() / 1000:,.0f}k",
          help="Suma de probabilidad × valor anual (12 × cargo mensual)")

# -----------------------------------------------------
# Gráficos
# -----------------------------------------------------
g1, g2 = st.columns(2)
with g1:
    fig = px.histogram(df, x="ProbChurn", color="NivelRiesgo", nbins=40, color_discrete_map=RISK_COLORS,
                       category_orders={"NivelRiesgo": NIVELES},
                       labels={"ProbChurn": "Probabilidad de churn", "NivelRiesgo": "Riesgo"})
    fig.add_vline(x=THR, line_dash="dash", line_color="#898781", annotation_text=f"Umbral {THR:.0%}")
    fig.update_layout(**PLOTLY_LAYOUT, height=360, bargap=0.06, yaxis_title="Clientes",
                      title="Distribución de la probabilidad de abandono")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
with g2:
    muestra = df.sample(min(len(df), 2500), random_state=1)
    fig = px.scatter(muestra, x="ProbChurn", y="ValorAnual", color="Prioridad", opacity=0.7,
                     color_discrete_map=PRIORITY_COLORS, category_orders={"Prioridad": list(PRIORITY_COLORS)},
                     hover_data={"customerID": True, "ProbChurn": ":.1%", "ValorAnual": ":$.0f", "Prioridad": False},
                     labels={"ProbChurn": "Probabilidad de churn", "ValorAnual": "Valor anual (USD)"})
    fig.add_vline(x=THR, line_dash="dash", line_color="#898781")
    fig.add_hline(y=meta["corte_valor_alto"], line_dash="dash", line_color="#898781")
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))
    fig.update_layout(**PLOTLY_LAYOUT, height=360, title="Matriz riesgo × valor (priorización)",
                      legend=dict(orientation="h", y=-0.25, title=None, font_size=11))
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

g3, g4 = st.columns(2)
with g3:
    seg = (df.groupby("SegmentoRFM").agg(prob=("ProbChurn", "mean"), n=("ProbChurn", "size"))
           .sort_values("prob"))
    fig = go.Figure(go.Bar(x=seg.prob, y=seg.index, orientation="h", marker_color=C_NO,
                           text=[f"{p:.0%}  (n={n:,})" for p, n in zip(seg.prob, seg.n)], textposition="outside",
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=330, title="Probabilidad media por segmento RFM",
                      xaxis=dict(tickformat=".0%", range=[0, max(seg.prob) * 1.35]))
    st.plotly_chart(fig, width="stretch")
with g4:
    fact = (alto["FactoresRiesgo"].dropna().str.split(" \\| ").explode().value_counts().head(8).sort_values())
    fig = go.Figure(go.Bar(x=fact.values, y=fact.index, orientation="h", marker_color="#d03b3b",
                           text=fact.values, textposition="outside",
                           hovertemplate="%{y}: %{x} clientes<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=330, title="Factores SHAP más frecuentes en clientes de riesgo alto",
                      xaxis=dict(range=[0, (fact.max() if len(fact) else 1) * 1.2]))
    st.plotly_chart(fig, width="stretch")

# Resumen de prioridades
st.subheader("Plan de acción por prioridad")
pr = (df.groupby("Prioridad").agg(Clientes=("customerID", "size"), ProbMedia=("ProbChurn", "mean"),
                                  ValorAnual=("ValorAnual", "sum"), PerdidaEsperada=("PerdidaEsperada", "sum"))
      .reindex(list(PRIORITY_COLORS)).dropna())
pr["Estrategia sugerida"] = [
    "Llamada del equipo de retención en < 48 h, oferta de contrato anual con descuento y beneficios premium.",
    "Acciones digitales de bajo costo: correo/SMS con incentivo de pago automático o servicio de prueba.",
    "Programa de lealtad, venta cruzada y encuestas de satisfacción.",
    "Comunicación estándar; monitorear cambios en el riesgo.",
][:len(pr)]
st.dataframe(pr, width="stretch", column_config={
    "ProbMedia": st.column_config.NumberColumn("Prob. media", format="percent"),
    "ValorAnual": st.column_config.NumberColumn("Valor anual total", format="dollar"),
    "PerdidaEsperada": st.column_config.NumberColumn("Pérdida esperada", format="dollar")})

# -----------------------------------------------------
# Lista de clientes en riesgo
# -----------------------------------------------------
st.subheader("Clientes en riesgo (ordenados por pérdida esperada)")
n = st.slider("Número de clientes a mostrar", 10, 500, 50, step=10)
cols = ["customerID", "ProbChurn", "NivelRiesgo", "Prioridad", "ValorAnual", "PerdidaEsperada", "Contract",
        "tenure", "MonthlyCharges", "InternetService", "PaymentMethod", "SegmentoRFM", "FactoresRiesgo", "Recomendaciones"]
lista = df[df.NivelRiesgo != "Bajo"].sort_values("PerdidaEsperada", ascending=False)[cols]
st.dataframe(lista.head(n), hide_index=True, width="stretch", height=420, column_config={
    "ProbChurn": st.column_config.ProgressColumn("Prob. churn", format="percent", min_value=0, max_value=1),
    "ValorAnual": st.column_config.NumberColumn("Valor anual", format="dollar"),
    "PerdidaEsperada": st.column_config.NumberColumn("Pérdida esperada", format="dollar"),
    "tenure": st.column_config.NumberColumn("Antigüedad (m)"),
    "customerID": "ID cliente", "NivelRiesgo": "Nivel de riesgo", "Contract": "Contrato",
    "InternetService": "Internet", "PaymentMethod": "Método de pago", "SegmentoRFM": "Segmento RFM",
    "FactoresRiesgo": "Factores de riesgo (SHAP)",
    "MonthlyCharges": st.column_config.NumberColumn("Cargo mensual", format="dollar")})
st.download_button(f"⬇️ Descargar lista de campaña ({len(lista):,} clientes en riesgo medio/alto)",
                   lista.to_csv(index=False).encode("utf-8"), "campana_retencion.csv", "text/csv", type="primary")
st.caption("Las probabilidades de la cartera completa se calcularon con el modelo en producción. "
           "Para los clientes del conjunto de entrenamiento son estimaciones dentro de la muestra; "
           "el desempeño honesto del modelo se reporta en la página «Desempeño de modelos» (conjunto de prueba).")
