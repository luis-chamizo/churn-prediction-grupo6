import pandas as pd
import plotly.graph_objects as go
import streamlit as st



import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # carpeta app/ (utils.py)
from utils import (C_NO, C_SI, CATEGORY_OPTIONS, FEATURE_LABELS, MODEL_COLORS,  # noqa: E402
                   PLOTLY_LAYOUT, RAW_FEATURES, RISK_COLORS, engineered_row, explain, fmt_value,
                   load_models, load_scored, recommend, risk_badge, risk_level,
                   sidebar_info, value_segment, what_if)

meta, final, todos = load_models()
scored = load_scored()
sidebar_info(meta)
THR, HIGH = meta["umbral_decision"], meta["umbral_riesgo_alto"]

st.title("🔮 Predicción individual de churn")

# -----------------------------------------------------
# Selección de cliente o ingreso manual
# -----------------------------------------------------
modo = st.radio("Origen de los datos", ["Cliente existente de la cartera", "Ingresar datos manualmente"], horizontal=True)
defaults = scored.iloc[0][RAW_FEATURES].to_dict()
if modo == "Cliente existente de la cartera":
    activos = scored[scored["ChurnReal"] == 0].sort_values("ProbChurn", ascending=False)
    cid = st.selectbox("ID del cliente (ordenados de mayor a menor riesgo; solo clientes activos)",
                       activos["customerID"].tolist())
    defaults = scored.loc[scored["customerID"] == cid, RAW_FEATURES].iloc[0].to_dict()

opt = lambda col: CATEGORY_OPTIONS[col]  # noqa: E731
idx = lambda col: opt(col).index(defaults[col]) if defaults[col] in opt(col) else 0  # noqa: E731

with st.form("cliente"):
    st.markdown("##### 👤 Perfil del cliente")
    c1, c2, c3, c4 = st.columns(4)
    gender = c1.selectbox("Género", opt("gender"), index=idx("gender"))
    senior = c2.selectbox("Adulto mayor", ["No", "Sí"], index=int(defaults["SeniorCitizen"]))
    partner = c3.selectbox("Tiene pareja", opt("Partner"), index=idx("Partner"))
    dependents = c4.selectbox("Tiene dependientes", opt("Dependents"), index=idx("Dependents"))

    st.markdown("##### 📡 Servicios contratados")
    c1, c2, c3, c4 = st.columns(4)
    phone = c1.selectbox("Servicio telefónico", opt("PhoneService"), index=idx("PhoneService"))
    lines = c2.selectbox("Múltiples líneas", opt("MultipleLines"), index=idx("MultipleLines"))
    internet = c3.selectbox("Tipo de internet", opt("InternetService"), index=idx("InternetService"))
    security = c4.selectbox("Seguridad en línea", opt("OnlineSecurity"), index=idx("OnlineSecurity"))
    c1, c2, c3, c4 = st.columns(4)
    backup = c1.selectbox("Respaldo en línea", opt("OnlineBackup"), index=idx("OnlineBackup"))
    protection = c2.selectbox("Protección de equipo", opt("DeviceProtection"), index=idx("DeviceProtection"))
    support = c3.selectbox("Soporte técnico", opt("TechSupport"), index=idx("TechSupport"))
    tv = c4.selectbox("Streaming TV", opt("StreamingTV"), index=idx("StreamingTV"))
    c1, c2, c3, c4 = st.columns(4)
    movies = c1.selectbox("Streaming películas", opt("StreamingMovies"), index=idx("StreamingMovies"))

    st.markdown("##### 💳 Cuenta y facturación")
    c1, c2, c3, c4 = st.columns(4)
    contract = c1.selectbox("Tipo de contrato", opt("Contract"), index=idx("Contract"))
    paperless = c2.selectbox("Factura electrónica", opt("PaperlessBilling"), index=idx("PaperlessBilling"))
    payment = c3.selectbox("Método de pago", opt("PaymentMethod"), index=idx("PaymentMethod"))
    tenure = c4.number_input("Antigüedad (meses)", 0, 100, int(defaults["tenure"]))
    c1, c2, _, _ = st.columns(4)
    monthly = c1.number_input("Cargo mensual (USD)", 0.0, 200.0, float(defaults["MonthlyCharges"]), step=0.5)
    total = c2.number_input("Cargo total acumulado (USD)", 0.0, 10000.0,
                            float(defaults["TotalCharges"]), step=10.0,
                            help="Si el cliente es nuevo, aproximadamente antigüedad × cargo mensual")
    enviar = st.form_submit_button("Predecir riesgo de abandono", type="primary", width="stretch")

# Coherencia de servicios dependientes
if internet == "No":
    security = backup = protection = support = tv = movies = "No internet service"
if phone == "No":
    lines = "No phone service"

row = {"gender": gender, "SeniorCitizen": 1 if senior == "Sí" else 0, "Partner": partner,
       "Dependents": dependents, "tenure": int(tenure), "PhoneService": phone, "MultipleLines": lines,
       "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
       "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
       "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
       "PaymentMethod": payment, "MonthlyCharges": float(monthly), "TotalCharges": float(total)}
X = pd.DataFrame([row])[RAW_FEATURES]

if not enviar and modo == "Ingresar datos manualmente":
    st.info("Completa el formulario y presiona **Predecir**.")
    st.stop()

# -----------------------------------------------------
# Resultado
# -----------------------------------------------------
p = float(final.predict_proba(X)[0, 1])
nivel = risk_level(p, THR, HIGH)
prioridad = value_segment(p, monthly * 12, THR, meta["corte_valor_alto"])

st.divider()
c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1.4])
c1.metric("Probabilidad de churn", f"{p:.1%}")
c2.markdown("<div style='font-size:0.875rem;margin-bottom:0.6rem'>Nivel de riesgo</div>" + risk_badge(nivel),
            unsafe_allow_html=True)
c2.caption(f"Bajo < {THR:.0%} ≤ Medio < {HIGH:.0%} ≤ Alto")
c3.metric("Pérdida anual esperada", f"USD {p * monthly * 12:,.0f}", help="probabilidad × 12 × cargo mensual")
c4.markdown("**Prioridad de retención**  \n" + prioridad.replace(".", "\\.", 1))

# Barra de probabilidad con umbrales
fig = go.Figure()
fig.add_trace(go.Bar(x=[p], y=[""], orientation="h", marker_color=RISK_COLORS[nivel], width=0.5,
                     hovertemplate="Probabilidad: %{x:.1%}<extra></extra>"))
for v, lab in [(THR, f"Umbral decisión {THR:.0%}"), (HIGH, f"Riesgo alto {HIGH:.0%}")]:
    fig.add_vline(x=v, line_dash="dash", line_color="#898781", annotation_text=lab, annotation_position="top")
fig.update_layout(**PLOTLY_LAYOUT, height=130, xaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False)
st.plotly_chart(fig, width="stretch")

col_a, col_b = st.columns([1.15, 1])

# ---- Explicación SHAP
with col_a:
    st.subheader("¿Por qué este resultado? (SHAP)")
    contrib = explain(final, meta["modelo_final"], X).iloc[0]
    fe_row = engineered_row(final, X)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)[::-1]
    labels = [f"{FEATURE_LABELS.get(c, c)} = {fmt_value(fe_row.get(c, row.get(c)))}" for c in top.index]
    fig = go.Figure(go.Bar(
        x=top.values, y=labels, orientation="h",
        marker_color=[C_SI if v > 0 else C_NO for v in top.values],
        hovertemplate="%{y}<br>Contribución: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#898781")
    fig.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_title=f"Contribución SHAP ({meta.get('shap_unidad', 'log-odds')})")
    st.plotly_chart(fig, width="stretch")
    st.caption("🟧 Naranja: aumenta el riesgo de abandono · 🟦 Azul: lo reduce. "
               "Las contribuciones de las variables one-hot se agregaron por variable original.")

# ---- Recomendaciones
with col_b:
    st.subheader("Recomendaciones de retención")
    recs = recommend({**row, **fe_row}, contrib.to_dict(), top_n=4)
    for i, r in enumerate(recs, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {r['accion']}**")
            st.caption(r["detalle"])
            st.caption(f"📞 Canal: {r['canal']} · Factor asociado: {r['variable']}"
                       + (f" (SHAP {r['impacto']:+.2f})" if r["impacto"] else ""))

# ---- Simulación ¿qué pasaría si…?
st.subheader("🧪 Simulador: ¿cuánto bajaría el riesgo con cada acción?")
wi = what_if(final, row)
if wi.empty:
    st.success("No hay acciones simulables para este cliente.")
else:
    wi_show = wi.copy()
    wi_show["Probabilidad nueva"] = wi_show["Probabilidad nueva"].map("{:.1%}".format)
    wi_show["Reducción (p.p.)"] = wi_show["Reducción (p.p.)"].map(lambda v: f"↓ {v:.1f} p.p." if v >= 0 else f"↑ {-v:.1f} p.p.")
    st.dataframe(wi_show, hide_index=True, width="stretch")
    st.caption("Escenarios contrafactuales evaluados con el modelo en producción: se cambia una sola variable "
               "y se mantienen las demás. Son una guía para priorizar acciones, no un efecto causal garantizado.")

# ---- Comparación entre modelos
with st.expander("Comparar la predicción de los 6 modelos"):
    probs = {n: float(m.predict_proba(X)[0, 1]) for n, m in todos.items()}
    names = list(probs.keys())
    fig = go.Figure(go.Bar(x=list(probs.values()), y=names, orientation="h",
                           marker_color=MODEL_COLORS[:len(names)],
                           text=[f"{v:.1%}" for v in probs.values()], textposition="outside",
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.add_vline(x=THR, line_dash="dash", line_color="#898781")
    fig.update_layout(**PLOTLY_LAYOUT, height=300, xaxis=dict(range=[0, 1.1], tickformat=".0%"))
    st.plotly_chart(fig, width="stretch")
