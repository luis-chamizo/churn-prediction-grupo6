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
RISK_COLORS = {"Bajo": "#16B978", "Medio": "#F5B51B", "Alto": "#E84B55"}
RISK_ICONS = {"Bajo": "verified", "Medio": "warning", "Alto": "error"}  # Material Symbols
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
            f"font-weight:600;font-size:1.05rem;display:inline-flex;align-items:center;gap:.35rem'>"
            f"{icon(RISK_ICONS[level], 'sm')} Riesgo {level}</span>")


# =====================================================
# DISEÑO DE LA INTERFAZ
# =====================================================
APP_DIR = Path(__file__).resolve().parent


def apply_style():
    css = (APP_DIR / "estilo.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def icon(name: str, cls: str = "") -> str:
    """Ícono de Material Symbols Rounded (Google) para usar dentro de HTML."""
    return f"<span class='mi {cls}'>{name}</span>"


def page_header(icono: str, title: str, subtitle: str, chips=None):
    """Encabezado tipo 'hero' con ícono, título, descripción y etiquetas."""
    chips_html = ("<div class='chips'>" + "".join(f"<span class='chip'>{c}</span>" for c in chips) + "</div>"
                  if chips else "")
    st.markdown(f"<div class='hero'><div class='hero-row'><div class='hero-ico'>{icon(icono)}</div>"
                f"<h1>{title}</h1></div><p>{subtitle}</p>{chips_html}</div>",
                unsafe_allow_html=True)


def section(title: str, icono: str | None = None):
    ico = icon(icono, "sec") if icono else "<div class='bar'></div>"
    st.markdown(f"<div class='section-title'>{ico}<span>{title}</span></div>",
                unsafe_allow_html=True)


def footer():
    st.markdown("<div class='app-foot'>Churn Prediction Inteligente · Grupo 6 · Luis Mario Chamizo Tiafi y "
                "Juan Diego Chamizo Tiafi<br>Fundación Universitaria de Popayán · Especialización en Data Analytics "
                "para Marketing Digital · Minería de Datos · 2026</div>", unsafe_allow_html=True)


@st.cache_data(ttl=600, show_spinner=False)
def model_health() -> bool:
    """Verificación real del estado del modelo: carga el pipeline y puntúa un cliente de la cartera."""
    try:
        _, final, _ = load_models()
        muestra = load_scored().head(1)[RAW_FEATURES]
        p = float(final.predict_proba(muestra)[0, 1])
        return 0.0 <= p <= 1.0
    except Exception:
        return False


def fecha_es(txt: str) -> str:
    """'2026-11-07 12:52' -> '07 nov 2026, 12:52'."""
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    try:
        d = pd.Timestamp(txt)
        return f"{d.day:02d} {meses[d.month - 1]} {d.year}, {d:%H:%M}"
    except Exception:
        return str(txt)


def sparkline_svg(values, color: str, title: str, w: int = 130, h: int = 42) -> str:
    """Minigráfica SVG (línea + área) dibujada a partir de una serie real de datos."""
    v = np.asarray(values, dtype=float)
    if v.size < 2 or np.nanmax(v) == np.nanmin(v):
        v = np.array([0.0, 1.0])
    x = np.linspace(2, w - 2, v.size)
    y = h - 3 - (v - v.min()) / (v.max() - v.min()) * (h - 8)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    gid = "g" + color.strip("#") + str(abs(hash(title)) % 10000)
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'><title>{title}</title>"
            f"<defs><linearGradient id='{gid}' x1='0' y1='0' x2='0' y2='1'>"
            f"<stop offset='0' stop-color='{color}' stop-opacity='.35'/><stop offset='1' stop-color='{color}' stop-opacity='0'/>"
            f"</linearGradient></defs>"
            f"<polygon points='2,{h} {pts} {w - 2},{h}' fill='url(#{gid})'/>"
            f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2.2' stroke-linejoin='round' stroke-linecap='round'/>"
            f"</svg>")


def sidebar_info(meta):
    apply_style()
    m = meta["metricas_test"][meta["modelo_final"]]
    ok = model_health()
    estado = ("<i></i> En producción" if ok else
              "<i style='background:#E84B55;box-shadow:0 0 0 3px rgba(232,75,85,.25)'></i> No disponible")
    with st.sidebar:
        st.markdown(
            "<div class='sbm'><div class='cap'>Modelo en producción</div>"
            f"<div class='row'><div class='bx'>{icon('neurology')}</div><div>"
            f"<div class='nm'>{meta['modelo_final']}</div><div class='st'>{estado}</div></div></div>"
            "<div class='gr two'>"
            f"<div><div class='k'>F1-Score</div><div class='v'>{m['F1']:.3f}</div></div>"
            f"<div><div class='k'>Umbral</div><div class='v'>{meta['umbral_decision']:.2f}</div></div>"
            "</div></div>"
            "<div class='sbf'>"
            f"<div>{icon('groups')}<span>Grupo 6<br>Luis Mario Chamizo Tiafi<br>Juan Diego Chamizo Tiafi</span></div>"
            f"<div>{icon('account_balance')}<span>Fundación Universitaria<br>de Popayán</span></div>"
            "</div>",
            unsafe_allow_html=True)
