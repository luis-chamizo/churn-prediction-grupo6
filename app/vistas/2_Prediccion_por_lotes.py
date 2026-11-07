import pandas as pd
import plotly.express as px
import streamlit as st



import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # carpeta app/ (utils.py)
from utils import (page_header, DATA, PLOTLY_LAYOUT, RAW_FEATURES, RISK_COLORS, clean_raw,  # noqa: E402
                   load_models, score_frame, sidebar_info, validate_input)

meta, final, _ = load_models()
sidebar_info(meta)

page_header("upload_file", "Predicción por lotes",
            "Sube un archivo CSV con las columnas del dataset <i>Telco Customer Churn</i> (customerID es opcional y "
            "Churn se ignora). Se calcula la probabilidad, el nivel de riesgo, los factores SHAP y las "
            "recomendaciones de cada cliente.", chips=["CSV", "Hasta 20 000 filas", "Descarga de resultados"])

raw_example = pd.read_csv(DATA / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
plantilla = raw_example.drop(columns=["Churn"]).sample(25, random_state=7)
st.download_button("Descargar plantilla de ejemplo (25 clientes)",
                   plantilla.to_csv(index=False).encode("utf-8"), "plantilla_clientes.csv", "text/csv", icon=":material/download:")

archivo = st.file_uploader("Archivo CSV", type="csv")
usar_ejemplo = st.checkbox("Usar la plantilla de ejemplo en lugar de subir un archivo")

if archivo is None and not usar_ejemplo:
    st.stop()

df = pd.read_csv(archivo) if archivo is not None else plantilla.copy()
problemas = validate_input(df)
if problemas:
    st.error("El archivo no tiene el formato esperado:\n\n- " + "\n- ".join(problemas))
    st.stop()

df = clean_raw(df)
if len(df) > 20000:
    st.warning("Se procesarán solo las primeras 20 000 filas.")
    df = df.head(20000)

with st.spinner(f"Puntuando {len(df):,} clientes…"):
    res = score_frame(final, meta, df)

salida = pd.concat([df[["customerID"]] if "customerID" in df else pd.DataFrame(index=df.index), res], axis=1)
salida = salida.sort_values("ProbChurn", ascending=False)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes procesados", f"{len(salida):,}")
c2.metric("Riesgo alto", f"{(salida.NivelRiesgo == 'Alto').sum():,}")
c3.metric("Riesgo medio", f"{(salida.NivelRiesgo == 'Medio').sum():,}")
c4.metric("Pérdida anual esperada", f"USD {salida.PerdidaEsperada.sum():,.0f}")

fig = px.histogram(salida, x="ProbChurn", color="NivelRiesgo", nbins=25,
                   color_discrete_map=RISK_COLORS, category_orders={"NivelRiesgo": ["Bajo", "Medio", "Alto"]},
                   labels={"ProbChurn": "Probabilidad de churn", "NivelRiesgo": "Nivel de riesgo"})
fig.update_layout(**PLOTLY_LAYOUT, height=320, bargap=0.08, yaxis_title="Clientes")
fig.update_xaxes(tickformat=".0%")
st.plotly_chart(fig, width="stretch")

st.dataframe(
    salida, hide_index=True, width="stretch",
    column_config={"ProbChurn": st.column_config.ProgressColumn("Prob. churn", format="percent", min_value=0, max_value=1),
                   "ValorAnual": st.column_config.NumberColumn("Valor anual", format="dollar"),
                   "PerdidaEsperada": st.column_config.NumberColumn("Pérdida esperada", format="dollar"),
                   "NivelRiesgo": "Nivel de riesgo", "FactoresRiesgo": "Factores de riesgo (SHAP)"})
st.download_button("Descargar resultados (CSV)", salida.to_csv(index=False).encode("utf-8"),
                   "predicciones_churn.csv", "text/csv", type="primary", icon=":material/download:")
