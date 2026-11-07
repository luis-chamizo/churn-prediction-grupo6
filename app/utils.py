# =====================================================
# UTILIDADES COMPARTIDAS DE LA APLICACIÓN WEB
# =====================================================
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import (FEATURE_LABELS, RAW_FEATURES, CATEGORY_OPTIONS,  # noqa: E402
                      aggregate_shap, clean_raw, recommend, risk_level,
                      validate_input, value_segment)

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DATA = ROOT / "data"

# Paleta (validada para daltonismo) – misma que el notebook
C_NO, C_SI = "#2a78d6", "#eb6834"
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
RISK_COLORS = {"Bajo": "#0ca30c", "Medio": "#fab219", "Alto": "#d03b3b"}
RISK_ICONS = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}
PRIORITY_COLORS = {
    "1. Retener ya (alto riesgo, alto valor)": "#d03b3b",
    "2. Retención de bajo costo (alto riesgo, bajo valor)": "#ec835a",
    "3. Fidelizar (bajo riesgo, alto valor)": "#2a78d6",
    "4. Mantener (bajo riesgo, bajo valor)": "#898781",
}
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=13, color="#52514e"),
    title_font=dict(size=15, color="#0f2742"),
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    margin=dict(l=12, r=12, t=48, b=12),
    hoverlabel=dict(font_size=12, font_family="Inter, sans-serif"),
)


@st.cache_resource(show_spinner="Cargando modelos…")
def load_models():
    meta = json.load(open(MODELS / "metadata.json", encoding="utf-8"))
    final = joblib.load(MODELS / meta["archivo_modelo_final"])
    todos = {name: joblib.load(MODELS / f) for name, f in meta["archivos_modelos"].items()}
    return meta, final, todos


@st.cache_data(show_spinner="Cargando cartera de clientes…")
def load_scored():
    return pd.read_csv(DATA / "clientes_scored.csv")


@st.cache_resource(show_spinner=False)
def get_explainer(_model, name):
    """TreeSHAP para modelos de árboles, LinearSHAP para la regresión logística."""
    import shap
    from sklearn.linear_model import LogisticRegression
    est = _model.named_steps["model"]
    if isinstance(est, LogisticRegression):
        raw = clean_raw(pd.read_csv(DATA / "WA_Fn-UseC_-Telco-Customer-Churn.csv"))[RAW_FEATURES]
        bg = _model[:-1].transform(raw.sample(500, random_state=0))
        return shap.LinearExplainer(est, bg)
    return shap.TreeExplainer(est)


def explain(model, name, X: pd.DataFrame) -> pd.DataFrame:
    """Contribuciones SHAP agregadas por variable original (filas = clientes)."""
    prep = model[:-1]
    M = prep.transform(X)
    sv = get_explainer(model, name).shap_values(M)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return aggregate_shap(sv, prep.get_feature_names_out())


def fmt_value(v):
    if isinstance(v, (float, np.floating)):
        return int(v) if float(v).is_integer() else f"{v:.2f}"
    return v


def engineered_row(model, X: pd.DataFrame) -> dict:
    return model.named_steps["fe"].transform(X).iloc[0].to_dict()


# Escenarios "¿qué pasaría si…?" para cuantificar el efecto de cada acción
SCENARIOS = [
    ("Migrar a contrato de 1 año", lambda r: r["Contract"] == "Month-to-month",
     lambda r: {**r, "Contract": "One year"}),
    ("Migrar a contrato de 2 años", lambda r: r["Contract"] != "Two year",
     lambda r: {**r, "Contract": "Two year"}),
    ("Activar Seguridad en línea", lambda r: r["OnlineSecurity"] == "No",
     lambda r: {**r, "OnlineSecurity": "Yes"}),
    ("Activar Soporte técnico", lambda r: r["TechSupport"] == "No",
     lambda r: {**r, "TechSupport": "Yes"}),
    ("Pasar a pago automático", lambda r: "automatic" not in r["PaymentMethod"],
     lambda r: {**r, "PaymentMethod": "Credit card (automatic)"}),
    ("Descuento del 15 % en la tarifa", lambda r: r["MonthlyCharges"] > 30,
     lambda r: {**r, "MonthlyCharges": round(r["MonthlyCharges"] * 0.85, 2)}),
]


def what_if(model, row: dict) -> pd.DataFrame:
    base = model.predict_proba(pd.DataFrame([row])[RAW_FEATURES])[0, 1]
    out = []
    for name, cond, change in SCENARIOS:
        if cond(row):
            p = model.predict_proba(pd.DataFrame([change(row)])[RAW_FEATURES])[0, 1]
            out.append({"Acción simulada": name, "Probabilidad nueva": p, "Reducción (p.p.)": (base - p) * 100})
    return pd.DataFrame(out).sort_values("Reducción (p.p.)", ascending=False) if out else pd.DataFrame()


def score_frame(model, meta, X: pd.DataFrame, with_recs: bool = True) -> pd.DataFrame:
    """Puntúa un lote de clientes en formato crudo."""
    thr, high = meta["umbral_decision"], meta["umbral_riesgo_alto"]
    prob = model.predict_proba(X[RAW_FEATURES])[:, 1]
    out = pd.DataFrame({"ProbChurn": prob.round(4)}, index=X.index)
    out["NivelRiesgo"] = [risk_level(p, thr, high) for p in prob]
    valor = X["MonthlyCharges"].values * 12
    out["ValorAnual"] = valor.round(2)
    out["PerdidaEsperada"] = (prob * valor).round(2)
    out["Prioridad"] = [value_segment(p, v, thr, meta["corte_valor_alto"]) for p, v in zip(prob, valor)]
    if with_recs:
        agg = explain(model, meta["modelo_final"], X[RAW_FEATURES])
        fe = model.named_steps["fe"].transform(X[RAW_FEATURES])
        factores, recs = [], []
        for i in range(len(X)):
            s = agg.iloc[i]
            top = s[s > 0].sort_values(ascending=False).head(3)
            factores.append(" | ".join(FEATURE_LABELS.get(c, c) for c in top.index))
            row = {**X[RAW_FEATURES].iloc[i].to_dict(), **fe.iloc[i].to_dict()}
            recs.append(" | ".join(r["accion"] for r in recommend(row, s.to_dict())))
        out["FactoresRiesgo"] = factores
        out["Recomendaciones"] = recs
    return out


def risk_badge(level: str) -> str:
    """Insignia HTML de nivel de riesgo (icono + texto, nunca solo color)."""
    fg = "#0b0b0b" if level == "Medio" else "#ffffff"
    return (f"<span style='background:{RISK_COLORS[level]};color:{fg};padding:6px 14px;border-radius:999px;"
            f"font-weight:600;font-size:1.05rem'>{RISK_ICONS[level]} Riesgo {level}</span>")


# =====================================================
# DISEÑO DE LA INTERFAZ
# =====================================================
APP_DIR = Path(__file__).resolve().parent


def apply_style():
    css = (APP_DIR / "estilo.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str, chips=None):
    """Encabezado tipo 'hero' con título, descripción y etiquetas."""
    chips_html = ("<div class='chips'>" + "".join(f"<span class='chip'>{c}</span>" for c in chips) + "</div>"
                  if chips else "")
    st.markdown(f"<div class='hero'><h1>{icon} {title}</h1><p>{subtitle}</p>{chips_html}</div>",
                unsafe_allow_html=True)


def section(title: str):
    st.markdown(f"<div class='section-title'><div class='bar'></div><span>{title}</span></div>",
                unsafe_allow_html=True)


def footer():
    st.markdown("<div class='app-foot'>Churn Prediction Inteligente · Grupo 6 · Luis Mario Chamizo Tiafi y "
                "Juan Diego Chamizo Tiafi<br>Fundación Universitaria de Popayán · Especialización en Data Analytics "
                "para Marketing Digital · Minería de Datos · 2026</div>", unsafe_allow_html=True)


def sidebar_info(meta):
    apply_style()
    m = meta["metricas_test"][meta["modelo_final"]]
    with st.sidebar:
        st.markdown(
            f"<div class='sb-card'><div class='lbl'>Modelo en producción</div>"
            f"<div class='val'>{meta['modelo_final']}</div><div class='sb-grid'>"
            f"<div><div class='lbl'>ROC-AUC</div><b>{m['ROC-AUC']:.3f}</b></div>"
            f"<div><div class='lbl'>Recall</div><b>{m['Recall']:.1%}</b></div>"
            f"<div><div class='lbl'>F1</div><b>{m['F1']:.3f}</b></div>"
            f"<div><div class='lbl'>Umbral</div><b>{meta['umbral_decision']:.2f}</b></div></div></div>"
            "<div class='sb-foot'>Grupo 6 · Luis Mario Chamizo Tiafi<br>Juan Diego Chamizo Tiafi<br>"
            "Fundación Universitaria de Popayán</div>",
            unsafe_allow_html=True)
