# Genera contenido.json (texto + tablas + figuras) del artículo IEEE a partir
# de los resultados reales del notebook. Luego: node render_ieee.js contenido.json ../Articulo.docx
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import FEATURE_LABELS, RAW_FEATURES, clean_raw  # noqa: E402

FIG = ROOT / "reports" / "figures"
meta = json.load(open(ROOT / "models" / "metadata.json", encoding="utf-8"))
df = clean_raw(pd.read_csv(ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"))
scored = pd.read_csv(ROOT / "data" / "clientes_scored.csv")
cv = pd.read_csv(ROOT / "reports" / "validacion_cruzada.csv")
asoc = pd.read_csv(ROOT / "reports" / "pruebas_asociacion.csv")
shap_imp = pd.read_csv(ROOT / "reports" / "shap_importancia_global.csv", index_col=0)["mean_abs_shap"]
mats = json.load(open(ROOT / "reports" / "matrices_confusion.json"))

FINAL = meta["modelo_final"]
THR = meta["umbral_decision"]
HIGH = meta["umbral_riesgo_alto"]
M = meta["metricas_test"]
mf = M[FINAL]
pct = lambda x, d=1: f"{100 * x:.{d}f} %".replace(".", ",")          # noqa: E731
num = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")                    # noqa: E731
miles = lambda x: f"{x:,.0f}".replace(",", ".")                         # noqa: E731

# ----------------------------------------------------------------------------
# Cifras del análisis
# ----------------------------------------------------------------------------
N = len(df); tasa = df.Churn.mean()
med = df.groupby("Churn")[["tenure", "MonthlyCharges"]].median()
rate = lambda col, val: df.loc[df[col] == val, "Churn"].mean()        # noqa: E731
r_mtm, r_1y, r_2y = rate("Contract", "Month-to-month"), rate("Contract", "One year"), rate("Contract", "Two year")
r_fiber, r_echeck = rate("InternetService", "Fiber optic"), rate("PaymentMethod", "Electronic check")
r_auto = df.loc[df.PaymentMethod.str.contains("automatic"), "Churn"].mean()
r_nosec, r_senior = rate("OnlineSecurity", "No"), rate("SeniorCitizen", 1)
seg_rate = scored.groupby("SegmentoRFM")["ChurnReal"].agg(["mean", "size"]).sort_values("mean", ascending=False)

test = scored[scored.Conjunto == "test"].copy()
yt, pf = test.ChurnReal.values, test.ProbChurn.values
orden = np.argsort(-pf); cap = np.cumsum(yt[orden]) / yt.sum()
cap20, cap30 = cap[int(0.2 * len(yt)) - 1], cap[int(0.3 * len(yt)) - 1]
valor = test.MonthlyCharges.values * 12
benef = lambda t: (yt[pf >= t] * valor[pf >= t] * 0.35).sum() - 50 * (pf >= t).sum()   # noqa: E731
b_thr, b_opt = benef(THR), benef(meta["umbral_beneficio_maximo"])
cm = np.array(mats[FINAL]); tn, fp, fn, tp = cm.ravel()

cv_sum = cv.groupby("modelo")[["Accuracy", "F1", "ROC-AUC"]].agg(["mean", "std"])
rank = pd.DataFrame(M).T.sort_values("ROC-AUC", ascending=False)
best_test = rank.index[0]
auc_rng = (rank["ROC-AUC"].min(), rank["ROC-AUC"].max())
cv_rank = cv.groupby("modelo")["ROC-AUC"].mean().sort_values(ascending=False)
top_shap = [FEATURE_LABELS.get(c, c) for c in shap_imp.index[:6]]
activos = scored[scored.ChurnReal == 0]
n_alto = int((activos.NivelRiesgo == "Alto").sum()); n_medio = int((activos.NivelRiesgo == "Medio").sum())
perdida = activos.PerdidaEsperada.sum()
prio1 = activos[activos.Prioridad.str.startswith("1.")]
shap_unit = meta.get("shap_unidad", "log-odds")

# Ejemplo del simulador "¿qué pasaría si…?" con el cliente activo de mayor pérdida esperada
modelo = joblib.load(ROOT / "models" / "modelo_final.joblib")
ej = activos.sort_values("PerdidaEsperada", ascending=False).iloc[0]
row = ej[RAW_FEATURES].to_dict()
p0 = modelo.predict_proba(pd.DataFrame([row])[RAW_FEATURES])[0, 1]
def p_with(**kw):
    return modelo.predict_proba(pd.DataFrame([{**row, **kw}])[RAW_FEATURES])[0, 1]
p_1y = p_with(Contract="One year") if row["Contract"] == "Month-to-month" else p0
n_cols = len(modelo[:-1].get_feature_names_out())
p_sec = p_with(OnlineSecurity="Yes") if row["OnlineSecurity"] == "No" else p0

# ----------------------------------------------------------------------------
# Figura compacta propia del artículo: churn por 6 variables clave
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.grid": True, "grid.color": "#e1e0d9", "axes.axisbelow": True, "font.size": 9})
claves = [("Contract", "Contrato"), ("InternetService", "Internet"), ("PaymentMethod", "Método de pago"),
          ("OnlineSecurity", "Seguridad en línea"), ("TechSupport", "Soporte técnico"), ("SeniorCitizen", "Adulto mayor")]
abbr = {"Bank transfer (automatic)": "Transfer. (auto)", "Credit card (automatic)": "Tarjeta (auto)",
        "Electronic check": "Cheque electr.", "Mailed check": "Cheque correo", "No internet service": "Sin internet",
        "Month-to-month": "Mes a mes", "One year": "1 año", "Two year": "2 años", "Fiber optic": "Fibra óptica",
        0: "No", 1: "Sí", "Yes": "Sí"}
fig, axes = plt.subplots(3, 2, figsize=(6.6, 6.2))
for ax, (col, tit) in zip(axes.ravel(), claves):
    t = df.groupby(col)["Churn"].mean().sort_values()
    ax.barh([abbr.get(i, i) for i in t.index], t.values, height=0.6,
            color=["#eb6834" if v > tasa else "#2a78d6" for v in t.values])
    ax.axvline(tasa, color="#52514e", ls="--", lw=0.9)
    for i, v in enumerate(t.values):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8)
    ax.set_xlim(0, 0.6); ax.set_title(tit, fontsize=9.5, fontweight="bold"); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
plt.tight_layout()
plt.savefig(FIG / "doc_churn_variables_clave.png", dpi=220, bbox_inches="tight"); plt.close()

rel = lambda name: f"../../reports/figures/{name}.png"                 # noqa: E731
shots = ROOT / "docs" / "capturas"

# ----------------------------------------------------------------------------
# Contenido
# ----------------------------------------------------------------------------
B = []
h1 = lambda t: B.append({"type": "h1", "text": t})                     # noqa: E731
h2 = lambda t: B.append({"type": "h2", "text": t})                     # noqa: E731
p = lambda t, **k: B.append({"type": "p", "text": t, **k})             # noqa: E731
bul = lambda items: B.append({"type": "bullets", "items": items})      # noqa: E731
fig_ = lambda src, cap, wide=False, scale=1: B.append({"type": "widefig" if wide else "fig", "src": src, "caption": cap, "scale": scale})  # noqa: E731
tab = lambda cap, rows, widths=None, note=None: B.append({"type": "table", "caption": cap, "rows": rows, "widths": widths, "note": note})  # noqa: E731
eq = lambda t, n: B.append({"type": "eq", "text": t, "n": n})          # noqa: E731

# I. INTRODUCCIÓN -------------------------------------------------------------
h1("Introducción")
p("El abandono de clientes (*churn*) es uno de los indicadores más críticos en la industria de telecomunicaciones: "
  "los ingresos son recurrentes, los mercados están saturados y el costo de cambiar de proveedor para el usuario es bajo. "
  "Pequeñas mejoras en la retención producen aumentos desproporcionados en la rentabilidad [14], por lo que las empresas "
  "necesitan identificar **antes** de que ocurra qué clientes tienen mayor probabilidad de cancelar el servicio y, sobre todo, "
  "**por qué**, para diseñar acciones de retención focalizadas.")
p("Este trabajo presenta una solución completa de minería de datos para la predicción inteligente de *churn*, desarrollada "
  "con la metodología CRISP-DM [1] sobre el dataset público *Telco Customer Churn* de IBM [12]. Las contribuciones son: "
  "(i) un análisis exploratorio con pruebas estadísticas de asociación; (ii) una adaptación del análisis RFM [11] al contexto "
  "de suscripciones y 16 variables derivadas; (iii) la comparación rigurosa de cinco algoritmos, Regresión Logística, Random "
  "Forest, XGBoost, LightGBM y una red neuronal MLP, más un ensamble *Stacking*; (iv) la explicación global y local del "
  "modelo con SHAP [5], [6]; y (v) el despliegue en una aplicación web con dashboard de clientes en riesgo y un motor de "
  "recomendaciones de retención personalizadas guiado por la explicabilidad.")
p("El resto del documento sigue las seis fases de CRISP-DM: la Sección II resume trabajos relacionados; las Secciones III a "
  "VIII describen la comprensión del negocio, la comprensión de los datos, la preparación, el modelado, la evaluación y el "
  "despliegue; y la Sección IX presenta las conclusiones.")

# II. TRABAJOS RELACIONADOS ---------------------------------------------------
h1("Trabajos relacionados")
p("La predicción de *churn* en telecomunicaciones se ha abordado con árboles de decisión, regresión logística, redes "
  "neuronales y ensambles. Verbeke *et al.* [9] compararon numerosos clasificadores y propusieron evaluar los modelos con "
  "criterios de rentabilidad, no solo de exactitud. Burez y Van den Poel [10] mostraron que el desbalance de clases es un "
  "problema central y que el AUC es una métrica más adecuada que el *accuracy*. Los ensambles basados en árboles, "
  "*bagging* con Random Forest [2] y *boosting* con XGBoost [3] y LightGBM [4], son el estado del arte en datos tabulares, "
  "mientras que el *stacking* [7] combina modelos heterogéneos. Para la explicabilidad, SHAP [5] unifica varios métodos "
  "aditivos de atribución con base en los valores de Shapley, y TreeSHAP [6] los calcula de forma exacta y eficiente en "
  "modelos de árboles. Este trabajo integra estas líneas en un flujo reproducible que va de los datos a las decisiones.")

# III. FASE 1 -----------------------------------------------------------------
h1("Fase 1: Comprensión del negocio")
h2("Objetivos de negocio")
bul(["Identificar a los clientes con alta probabilidad de abandono.",
     "Comprender los factores de comportamiento, contrato, servicios y facturación asociados al abandono.",
     "Priorizar la inversión en retención según el **riesgo** y el **valor** del cliente.",
     "Proponer acciones de retención **personalizadas** para cada cliente."])
h2("Objetivos de minería de datos y criterios de éxito")
p("El problema se formula como una **clasificación binaria supervisada** (Churn = 1 si el cliente abandonó). Los criterios de "
  "éxito se fijaron *antes* del modelado (Tabla I). El supuesto económico para la evaluación de negocio es: costo de contacto e "
  "incentivo de USD 50 por cliente, tasa de éxito de la campaña del 35 % y valor recuperado igual a 12 meses del cargo mensual.")
tab("Criterios de éxito del proyecto", [
    ["Tipo", "Criterio"],
    ["Minería", "ROC-AUC ≥ 0,84 en el conjunto de prueba"],
    ["Minería", "Recall de la clase *churn* ≥ 0,70"],
    ["Negocio", "Top 30 % de riesgo captura ≥ 60 % del *churn*"],
    ["Despliegue", "App web con predicción, dashboard y recomendaciones"]], [0.28, 0.72])
h2("Plan del proyecto")
p("Herramientas: Python 3, pandas, scikit-learn [8], XGBoost, LightGBM, SHAP y Streamlit [16]. Todo el proceso es reproducible "
  "(semilla 42) y está encapsulado en un notebook ejecutable y en un módulo común (*src/features.py*) compartido por el "
  "entrenamiento y la aplicación web.")

# IV. FASE 2 ------------------------------------------------------------------
h1("Fase 2: Comprensión de los datos")
h2("Descripción y calidad")
p(f"El dataset contiene **{miles(N)} clientes** y 21 columnas: datos demográficos (género, adulto mayor, pareja, dependientes), "
  "servicios contratados (telefonía, múltiples líneas, internet DSL o fibra, seguridad, respaldo, protección, soporte, streaming), "
  "cuenta (antigüedad o *tenure*, contrato, facturación electrónica, método de pago, cargos mensual y total) y la variable objetivo. "
  f"La tasa de abandono es de **{pct(tasa)}** ({miles(df.Churn.sum())} clientes), es decir, las clases están desbalanceadas en una "
  "proporción de aproximadamente 1:2,8.")
p("La verificación de calidad no encontró valores nulos explícitos ni registros o identificadores duplicados. Sin embargo, "
  "*TotalCharges* está almacenada como texto y contiene **11 cadenas vacías**. Todas corresponden a clientes con *tenure* = 0 "
  "(recién incorporados y aún no facturados), por lo que no son datos perdidos al azar: su valor correcto es 0. Además, "
  "*SeniorCitizen* está codificada como 0/1, a diferencia del resto de variables binarias (Yes/No).")
h2("Exploración y análisis de comportamiento")
p(f"Los clientes que abandonan tienen una antigüedad mediana de **{med.loc[1, 'tenure']:.0f} meses**, frente a "
  f"{med.loc[0, 'tenure']:.0f} de los que permanecen, y un cargo mensual mediano mayor (USD {num(med.loc[1, 'MonthlyCharges'], 1)} "
  f"frente a {num(med.loc[0, 'MonthlyCharges'], 1)}) (Fig. 1). La Fig. 2 muestra los patrones de comportamiento más marcados: "
  f"el contrato **mes a mes** presenta {pct(r_mtm)} de *churn*, frente a {pct(r_1y)} (1 año) y {pct(r_2y)} (2 años); "
  f"la **fibra óptica**, {pct(r_fiber)}; el pago con **cheque electrónico**, {pct(r_echeck)}, frente a {pct(r_auto)} con pago "
  f"automático; la ausencia de **seguridad en línea**, {pct(r_nosec)}; y los **adultos mayores**, {pct(r_senior)}. "
  "El género y el servicio telefónico prácticamente no se asocian con el abandono.")
fig_(rel("02_numericas_vs_churn"), "Distribución (densidad) de antigüedad, cargo mensual y cargo total según el estado de churn. "
     "Los clientes que abandonan se concentran en los primeros meses y en cargos mensuales altos.", wide=True, scale=0.92)
fig_(rel("doc_churn_variables_clave"), "Tasa de churn por categoría en seis variables clave. Línea punteada: tasa global; "
     "naranja: categorías por encima del promedio.")
top_as = asoc.head(8)
tab("Asociación con el churn (prueba de hipótesis)",
    [["Variable", "Prueba", "Efecto", "p-valor"]] +
    [[FEATURE_LABELS.get(r.variable, r.variable), "χ²" if r.prueba.startswith("Chi") else "M-W U",
      num(r.efecto), "< 0,001" if r.p_valor < 0.001 else num(r.p_valor)] for r in top_as.itertuples()],
    [0.42, 0.18, 0.18, 0.22],
    note="Efecto: V de Cramér (χ²) o correlación biserial de rangos (Mann-Whitney U). Top 8 de 19 variables.")
p("Para cuantificar estas relaciones se aplicaron pruebas χ² con V de Cramér a las variables categóricas y la prueba de "
  f"Mann-Whitney U a las numéricas (Tabla II). Las asociaciones más fuertes corresponden a **{FEATURE_LABELS.get(asoc.variable[0])}**, "
  f"**{FEATURE_LABELS.get(asoc.variable[1])}** y **{FEATURE_LABELS.get(asoc.variable[2])}**. Además, *TotalCharges* está "
  "fuertemente correlacionada con *tenure* (ρ de Spearman ≈ 0,89), redundancia que motiva una variable derivada de gasto promedio.")

# V. FASE 3 -------------------------------------------------------------------
h1("Fase 3: Preparación de los datos")
h2("Limpieza y partición")
p("*TotalCharges* se convirtió a numérica e imputó con 0 en los 11 casos de *tenure* = 0; se eliminaron espacios en los textos, "
  "*Churn* se codificó como 0/1 y *customerID* se excluyó del modelado. Los datos se dividieron en **80 % de entrenamiento "
  f"({miles(meta['n_train'])}) y 20 % de prueba ({miles(meta['n_test'])})** de forma estratificada **antes** de cualquier "
  "transformación aprendida, para evitar fuga de información.")
h2("Ingeniería de características: RFM adaptado")
p("El análisis RFM (*Recency, Frequency, Monetary*) [11] se adaptó a un servicio de suscripción: **R** (relación) = antigüedad en "
  "meses; **F** (frecuencia de uso) = número de servicios contratados; **M** (monetario) = cargo mensual. Cada componente se "
  "discretiza en un puntaje de 1 a 5 según los quintiles q_k aprendidos *solo* con el entrenamiento:")
eq("s(x) = 1 + Σ_k 1[x > q_k],   RFM = s_R + s_F + s_M", 1)
p("Con los puntajes se definieron cinco segmentos: *Campeones* (R ≥ 4 y F ≥ 4), *Leales* (R ≥ 4), *Alto valor en riesgo* "
  "(R ≤ 2 y M ≥ 4), *Nuevos de bajo compromiso* (R ≤ 2 y F ≤ 2) y *Potenciales*. Como muestra la Fig. 3, los segmentos "
  f"discriminan fuertemente el riesgo: *{seg_rate.index[0]}* tiene {pct(seg_rate['mean'].iloc[0])} de *churn*, frente a "
  f"{pct(seg_rate['mean'].iloc[-1])} en *{seg_rate.index[-1]}*.")
fig_(rel("05_segmentos_rfm"), "Izquierda: tasa de churn por segmento RFM (entrenamiento). Derecha: tasa de churn por "
     "combinación de puntajes R (antigüedad) y F (servicios).", wide=True, scale=0.92)
tab("Variables derivadas (feature engineering)", [
    ["Variable", "Definición"],
    ["NumServicios", "Servicios contratados (0-9)"],
    ["NumProteccion", "Seguridad, respaldo, protección y soporte"],
    ["GastoPromedioMensual", "TotalCharges / tenure"],
    ["VariacionTarifa", "MonthlyCharges − gasto promedio"],
    ["CargoPorServicio", "MonthlyCharges / (servicios + 1)"],
    ["DuracionContratoMeses", "1, 12 o 24 según el contrato"],
    ["PagoAutomatico", "1 si el pago es automático"],
    ["TieneFamilia", "Pareja o dependientes"],
    ["FibraSinProteccion", "Fibra sin seguridad ni soporte"],
    ["GrupoAntiguedad", "Tramos 0-12, 13-24, 25-48, 49-72 m"],
    ["R, F, M, RFM, Segmento", "Puntajes y segmento RFM"]], [0.45, 0.55])
p("La información mutua con la variable objetivo (Fig. 4) confirma el valor del *feature engineering*: varias variables derivadas, "
  "como la duración del contrato, el puntaje R, el segmento RFM o la fibra sin protección, aparecen entre las más informativas.")
fig_(rel("06_informacion_mutua"), "Top 20 variables por información mutua con el churn. En naranja, variables creadas en la "
     "ingeniería de características.", scale=0.95)
h2("Transformación")
p("Las variables numéricas se estandarizaron y las categóricas se codificaron con *one-hot* (descartando una categoría en las "
  f"binarias), lo que produjo {n_cols} columnas. Todo el flujo se encapsuló en un *Pipeline* de scikit-learn "
  "(FeatureEngineer → ColumnTransformer → modelo), de modo que en cada pliegue de validación se reaprenden los cortes RFM y el "
  "escalado, y en producción basta con entregar los datos crudos. El desbalance se trató con **ajuste del umbral de decisión** "
  "sobre predicciones *out-of-fold*, que preserva probabilidades bien calibradas para el dashboard de riesgo.")

# VI. FASE 4 ------------------------------------------------------------------
h1("Fase 4: Modelado")
h2("Técnicas")
p("Se implementaron los cinco modelos requeridos: **Regresión Logística** [18] con regularización L1/L2, como línea base "
  "interpretable; **Random Forest** [2], ensamble de *bagging* que reduce la varianza promediando árboles descorrelacionados; "
  "**XGBoost** [3], *boosting* por gradiente de segundo orden con regularización; **LightGBM** [4], *boosting* con crecimiento "
  "por hojas e histogramas; y una **red neuronal MLP** [17] con parada temprana. Adicionalmente se construyó un **Stacking** [7] "
  "que combina los cinco modelos ajustados mediante un meta-modelo logístico entrenado sobre sus probabilidades *out-of-fold*.")
h2("Diseño de la prueba y ajuste de hiperparámetros")
p("Cada modelo se ajustó con búsqueda aleatoria [13] (15-25 configuraciones) y **validación cruzada estratificada de 5 pliegues**, "
  "con ROC-AUC como métrica de optimización por ser independiente del umbral y robusta al desbalance [10], [15]. La Tabla IV "
  "resume las configuraciones óptimas.")
hp_rows = [["Modelo", "Hiperparámetros óptimos", "AUC CV"]]
abrev = {"learning_rate": "lr", "n_estimators": "n_est", "max_depth": "depth", "min_samples_leaf": "min_leaf",
         "max_features": "max_feat", "colsample_bytree": "colsample", "min_child_weight": "min_cw", "reg_lambda": "λ",
         "min_child_samples": "min_cs", "num_leaves": "leaves", "subsample_freq": None, "hidden_layer_sizes": "capas",
         "learning_rate_init": "lr", "activation": "act", "penalty": "pen", "gamma": "γ", "subsample": "subsample", "alpha": "α"}
for b in meta["busqueda_hiperparametros"]:
    parts = []
    for k, v in b["mejores_hiperparametros"].items():
        k2 = abrev.get(k, k)
        if k2 is None:
            continue
        v = num(v, 3) if isinstance(v, float) else (str(tuple(v)) if isinstance(v, list) else v)
        parts.append(f"{k2}={v}")
    hp_rows.append([b["modelo"].replace("Red Neuronal (MLP)", "MLP").replace("Regresión Logística", "Reg. Logística"),
                    ", ".join(parts), num(b["roc_auc_cv"])])
tab("Hiperparámetros óptimos (búsqueda aleatoria, CV 5 pliegues)", hp_rows, [0.22, 0.6, 0.18])
fig_(rel("07_validacion_cruzada"), "Distribución de Accuracy, F1 y ROC-AUC en los 5 pliegues de validación cruzada "
     "(umbral 0,5) para los seis modelos.", wide=True, scale=0.95)
p(f"La Fig. 5 muestra que, en validación cruzada, todos los modelos alcanzan un ROC-AUC similar (entre {num(cv_rank.min())} y "
  f"{num(cv_rank.max())}), con *{cv_rank.index[0]}* en primer lugar. Con el umbral por defecto de 0,5 el F1 de la clase *churn* "
  "es moderado, lo que justifica optimizar el umbral.")

# VII. FASE 5 -----------------------------------------------------------------
h1("Fase 5: Evaluación")
h2("Métricas")
p("Sean TP, TN, FP y FN los verdaderos y falsos positivos y negativos. Se reportan *Accuracy* = (TP+TN)/N, *Precision* = "
  "TP/(TP+FP), *Recall* = TP/(TP+FN), F1 (2), el área bajo la curva ROC (ROC-AUC) [15], el área bajo la curva "
  "Precision-Recall (PR-AUC) y la puntuación de Brier, que mide la calibración de las probabilidades.")
eq("F1 = 2 · Precision · Recall / (Precision + Recall)", 2)
p("Para cada modelo se eligió el umbral que maximiza F1 sobre las predicciones *out-of-fold* del entrenamiento, sin usar el "
  "conjunto de prueba. La Tabla V presenta los resultados en los datos de prueba, que nunca se vieron durante el entrenamiento.")
met_rows = [["Modelo", "Umb.", "Acc.", "Prec.", "Rec.", "F1", "AUC", "PR-AUC"]]
for m, r in rank.iterrows():
    nm = m.replace("Red Neuronal (MLP)", "MLP").replace("Regresión Logística", "Reg. Log.").replace("Random Forest", "R. Forest")
    if m == FINAL:
        nm = f"**{nm}**"
    met_rows.append([nm, num(meta["umbrales_por_modelo"][m], 2), num(r["Accuracy"]), num(r["Precision"]), num(r["Recall"]),
                     num(r["F1"]), num(r["ROC-AUC"]), num(r["PR-AUC"])])
tab("Desempeño en el conjunto de prueba (umbral óptimo por modelo)", met_rows,
    [0.2, 0.1, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115],
    note=f"n = {miles(meta['n_test'])}; ordenado por ROC-AUC. En negrita, el modelo desplegado.")
fig_(rel("08_curvas_roc_pr"), "Curvas ROC (izquierda) y Precision-Recall (derecha) de los seis modelos en el conjunto de prueba.",
     wide=True, scale=0.9)
p(f"Las diferencias entre modelos son pequeñas: el ROC-AUC en prueba varía entre {num(auc_rng[0])} y {num(auc_rng[1])} (Fig. 6), "
  f"y *{best_test}* obtiene el mayor valor. Este resultado es coherente con la literatura sobre este dataset: la señal está dominada "
  "por pocas variables con relaciones casi monótonas (contrato, antigüedad, tipo de internet), que incluso un modelo lineal "
  "regularizado captura bien.")
h2("Selección del modelo final")
p("La regla de selección se definió antes de observar el conjunto de prueba: (1) mayor ROC-AUC promedio en validación cruzada; "
  "(2) si la diferencia con el mejor es menor que 0,003, valor inferior a la desviación estándar entre pliegues, se prefiere el "
  "modelo que admite explicaciones SHAP exactas (árboles o lineal) por parsimonia y velocidad de inferencia. Con esta regla se "
  f"seleccionó **{FINAL}**, con ROC-AUC en prueba de **{num(mf['ROC-AUC'])}** (IC 95 % por *bootstrap*: "
  f"[{num(meta['auc_ic95_final'][0])}; {num(meta['auc_ic95_final'][1])}]), Recall de **{num(mf['Recall'])}** y F1 de "
  f"**{num(mf['F1'])}** con umbral {num(THR, 2)}. Su matriz de confusión (Fig. 7) muestra que detecta {tp} de {tp + fn} "
  f"clientes que abandonan, a costa de {fp} falsas alarmas sobre {tn + fp} clientes que permanecen.")
fig_(rel("09_matrices_confusion"), "Matrices de confusión en el conjunto de prueba con el umbral óptimo de cada modelo.",
     wide=True, scale=0.8)
h2("Evaluación orientada al negocio")
p(f"La curva de ganancia acumulada (Fig. 8) indica que contactando al **20 %** de clientes con mayor riesgo se captura el "
  f"**{pct(cap20)}** de los abandonos (*lift* = {num(cap20 / 0.2, 2)}) y con el **30 %**, el **{pct(cap30)}** "
  f"(*lift* = {num(cap30 / 0.3, 2)}). Bajo los supuestos económicos de la Fase 1, la campaña dirigida con el umbral F1 "
  f"generaría un beneficio neto de **USD {miles(b_thr)}** sobre los {miles(len(test))} clientes de prueba; el umbral que maximiza "
  f"el beneficio es {num(meta['umbral_beneficio_maximo'], 2)} (USD {miles(b_opt)}), como se ve en la Fig. 9. Esto confirma que el "
  "umbral debe ajustarse al costo real de cada campaña, y la aplicación permite hacerlo.")
fig_(rel("11_curva_ganancia"), "Curva de ganancia acumulada del modelo final frente a una selección aleatoria (prueba).", scale=0.95)
fig_(rel("12_costo_beneficio"), "Beneficio neto simulado de la campaña de retención según el umbral de decisión.", scale=0.95)
h2("Interpretabilidad con SHAP")
p("SHAP [5] descompone cada predicción como la suma de un valor base y las contribuciones φ_j de cada variable, con base en "
  "los valores de Shapley de la teoría de juegos:")
eq("f(x) = φ₀ + Σⱼ φⱼ(x)", 3)
p(f"Para el modelo final se usó TreeSHAP [6] (contribuciones en {shap_unit}). Las contribuciones de las columnas *one-hot* se "
  "**sumaron** por variable original para facilitar la lectura. La importancia global (Fig. 10) ubica en los primeros lugares a "
  f"**{top_shap[0]}**, **{top_shap[1]}**, **{top_shap[2]}** y **{top_shap[3]}**, seguidos de {top_shap[4].lower()} y "
  f"{top_shap[5].lower()}. El gráfico *beeswarm* (Fig. 11) muestra la dirección de los efectos: el contrato mes a mes, la "
  "antigüedad baja, la fibra óptica y la falta de seguridad o soporte aumentan el riesgo, mientras que los contratos largos lo "
  "reducen.")
fig_(rel("14_shap_importancia_global"), "Importancia global SHAP (media del valor absoluto) agregada por variable original.", scale=0.92)
fig_(rel("13_shap_beeswarm"), "Gráfico beeswarm de SHAP en el conjunto de prueba. Cada punto es un cliente; el color indica el valor "
     "de la variable (azul bajo, naranja alto).", scale=0.95)
p("Los gráficos de dependencia (Fig. 12) revelan efectos **no lineales**: el riesgo asociado a la antigüedad es muy alto en los "
  "primeros meses y cae marcadamente después del primer año, y su magnitud depende del tipo de contrato. A nivel individual "
  "(Fig. 13), SHAP identifica los factores concretos que llevan a un cliente a una probabilidad alta, que es exactamente la "
  "información que necesita el equipo de retención.")
fig_(rel("15_shap_dependencia"), "Gráficos de dependencia SHAP para antigüedad, cargo mensual y variación de tarifa. Color: tipo de "
     "contrato (naranja = mes a mes).", wide=True, scale=0.95)
fig_(rel("16_shap_local_alto"), "Explicación local SHAP del cliente de mayor riesgo del conjunto de prueba (10 factores principales).")
h2("Verificación de criterios de éxito")
tab("Cumplimiento de los criterios de éxito", [
    ["Criterio", "Resultado", "¿Cumple?"],
    ["ROC-AUC ≥ 0,84", num(mf["ROC-AUC"]), "Sí" if mf["ROC-AUC"] >= 0.84 else "No"],
    ["Recall ≥ 0,70", num(mf["Recall"]), "Sí" if mf["Recall"] >= 0.70 else "No"],
    ["Top 30 % capta ≥ 60 %", pct(cap30), "Sí" if cap30 >= 0.60 else "No"],
    ["App web desplegable", "Streamlit (Sec. VIII)", "Sí"]], [0.45, 0.33, 0.22])

# VIII. FASE 6 ----------------------------------------------------------------
h1("Fase 6: Despliegue")
h2("Arquitectura")
p("El *pipeline* final se serializó con *joblib* junto con un archivo de metadatos (umbrales, métricas y versiones). La "
  "aplicación web se desarrolló en **Streamlit** [16] y reutiliza el mismo módulo de preprocesamiento del entrenamiento, lo que "
  "garantiza que los datos nuevos se transformen exactamente igual. Puede ejecutarse localmente, publicarse gratis en Streamlit "
  "Community Cloud desde un repositorio de GitHub, o desplegarse en cualquier servicio de contenedores con el *Dockerfile* incluido. "
  "La aplicación está publicada en **https://churn-prediction-grupo6.streamlit.app** y el código fuente en "
  "**https://github.com/luis-chamizo/churn-prediction-grupo6**. Tiene cinco módulos:")
bul(["**Predicción individual:** formulario (o selección de un cliente existente) que devuelve la probabilidad de *churn*, el nivel "
     f"de riesgo (Bajo < {num(THR, 2)} ≤ Medio < {num(HIGH, 2)} ≤ Alto), la explicación SHAP, las recomendaciones y un simulador.",
     "**Predicción por lotes:** carga de un CSV, validación de columnas y descarga de los resultados.",
     "**Dashboard de riesgo:** KPIs, filtros, distribución del riesgo, matriz riesgo × valor, factores SHAP frecuentes y lista "
     "descargable de clientes para campañas.",
     "**Desempeño de modelos:** comparación interactiva de métricas, curvas ROC, matrices de confusión e importancia SHAP.",
     "**Inicio (panel ejecutivo):** estado real del modelo, KPI con minigráficas, distribución del riesgo, comportamiento "
     "del abandono por antigüedad, factores SHAP, clientes prioritarios y la recomendación agregada para la campaña."])
if (shots / "dashboard.png").exists():
    fig_("../capturas/dashboard.png", "Panel principal de la aplicación web: KPI con minigráficas construidas con datos reales, "
         "distribución de clientes activos por nivel de riesgo y comportamiento del abandono según la antigüedad.", wide=True, scale=0.85)
h2("Motor de recomendaciones de retención")
p("Cada factor de riesgo accionable se asocia con una regla de negocio. Por ejemplo: contrato mes a mes → migración a contrato "
  "anual con descuento; antigüedad ≤ 12 meses → programa de bienvenida; sin seguridad en línea o soporte → prueba gratuita del "
  "servicio; fibra óptica → revisión técnica de la calidad; cheque electrónico → incentivo por pago automático; cargo ≥ USD 70 → "
  "optimización del plan. Lo distintivo es que las reglas activas se **ordenan por la contribución SHAP individual** de la variable "
  "asociada, de modo que a cada cliente se le propone primero la acción que ataca su principal factor de riesgo. Los clientes se "
  "priorizan además con una **matriz riesgo × valor** (pérdida esperada = probabilidad × 12 × cargo mensual).")
p("El **simulador contrafactual** recalcula la probabilidad cambiando una sola variable para estimar el efecto de cada acción. "
  f"Por ejemplo, para el cliente activo con mayor pérdida esperada ({ej.customerID}, probabilidad {pct(p0)}), migrar a contrato "
  f"de un año reduce el riesgo a {pct(p_1y)} y activar la seguridad en línea, a {pct(p_sec)}. Estas estimaciones orientan la "
  "priorización, pero no constituyen un efecto causal garantizado.")
if (shots / "prediccion.png").exists():
    fig_("../capturas/prediccion.png", "Módulo de predicción individual: explicación SHAP del cliente (naranja = aumenta el riesgo) y "
         "recomendaciones de retención ordenadas por su impacto SHAP.", wide=True, scale=0.85)
h2("Resultados sobre la cartera y plan de monitoreo")
p(f"Al puntuar a los {miles(len(activos))} clientes activos, el modelo identifica **{miles(n_alto)} en riesgo alto** y "
  f"{miles(n_medio)} en riesgo medio, con una pérdida anual esperada total de **USD {miles(perdida)}**. El segmento prioritario "
  f"*Retener ya* (alto riesgo y alto valor) agrupa a {miles(len(prio1))} clientes que concentran USD {miles(prio1.PerdidaEsperada.sum())} "
  "de esa pérdida esperada. Para el mantenimiento se propone: monitoreo mensual del *drift* con el índice PSI (alerta si PSI > 0,2), "
  "contraste mensual de las predicciones con los abandonos reales (reentrenar si el ROC-AUC baja de 0,80), pruebas A/B con grupo de "
  "control para medir el efecto real de las campañas y reentrenamiento trimestral ejecutando el notebook reproducible.")

# IX. CONCLUSIONES ------------------------------------------------------------
h1("Conclusiones")
p(f"Se desarrolló una solución completa de predicción de *churn* siguiendo las seis fases de CRISP-DM. Los cinco modelos "
  f"requeridos y el *Stacking* alcanzaron un desempeño alto y muy similar (ROC-AUC en prueba entre {num(auc_rng[0])} y "
  f"{num(auc_rng[1])}). El modelo desplegado, {FINAL}, cumple los criterios de éxito definidos (ROC-AUC = {num(mf['ROC-AUC'])}, "
  f"Recall = {num(mf['Recall'])}) y permite capturar el {pct(cap30)} de los abandonos contactando solo al 30 % de los clientes.")
p("El análisis de comportamiento y SHAP coinciden en que el abandono se explica sobre todo por la **falta de compromiso "
  "contractual** (contrato mes a mes y baja antigüedad), por la **relación precio-servicio** (fibra óptica y cargos altos) y por la "
  "**ausencia de servicios de protección y soporte**. Todos son factores accionables. La adaptación RFM aportó variables con alto "
  "poder discriminante y una segmentación fácil de comunicar al negocio.")
p("Como trabajo futuro se propone: incorporar datos temporales de uso y de interacciones con el servicio al cliente; modelar el "
  "*uplift* de cada acción con experimentos controlados; calibrar el umbral con los costos reales de la empresa; y evaluar modelos "
  "de supervivencia para estimar *cuándo* abandonará cada cliente.")

B.append({"type": "h1_nonum", "text": "Referencias"})
B.append({"type": "refs", "items": [
    "P. Chapman *et al.*, “CRISP-DM 1.0: Step-by-step data mining guide,” SPSS Inc., 2000.",
    "L. Breiman, “Random forests,” *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.",
    "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.",
    "G. Ke *et al.*, “LightGBM: A highly efficient gradient boosting decision tree,” in *Advances in Neural Information Processing Systems 30*, 2017, pp. 3146–3154.",
    "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Advances in Neural Information Processing Systems 30*, 2017, pp. 4765–4774.",
    "S. M. Lundberg *et al.*, “From local explanations to global understanding with explainable AI for trees,” *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, 2020.",
    "D. H. Wolpert, “Stacked generalization,” *Neural Networks*, vol. 5, no. 2, pp. 241–259, 1992.",
    "F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *J. Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.",
    "W. Verbeke, K. Dejaeger, D. Martens, J. Hur, and B. Baesens, “New insights into churn prediction in the telecommunication sector: A profit driven data mining approach,” *European J. Operational Research*, vol. 218, no. 1, pp. 211–229, 2012.",
    "J. Burez and D. Van den Poel, “Handling class imbalance in customer churn prediction,” *Expert Systems with Applications*, vol. 36, no. 3, pp. 4626–4636, 2009.",
    "A. M. Hughes, *Strategic Database Marketing*. Chicago, IL, USA: Probus Publishing, 1994.",
    "BlastChar (IBM Sample Data Sets), “Telco Customer Churn,” Kaggle, 2018. [Online]. Available: https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
    "J. Bergstra and Y. Bengio, “Random search for hyper-parameter optimization,” *J. Machine Learning Research*, vol. 13, pp. 281–305, 2012.",
    "F. F. Reichheld and W. E. Sasser, “Zero defections: Quality comes to services,” *Harvard Business Review*, vol. 68, no. 5, pp. 105–111, 1990.",
    "T. Fawcett, “An introduction to ROC analysis,” *Pattern Recognition Letters*, vol. 27, no. 8, pp. 861–874, 2006.",
    "Streamlit Inc., “Streamlit documentation.” [Online]. Available: https://docs.streamlit.io",
    "D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by back-propagating errors,” *Nature*, vol. 323, pp. 533–536, 1986.",
    "D. R. Cox, “The regression analysis of binary sequences,” *J. Royal Statistical Society: Series B*, vol. 20, no. 2, pp. 215–242, 1958.",
]})

spec = {
    "title": "Churn Prediction Inteligente: modelo predictivo explicable para la retención de clientes de telecomunicaciones con CRISP-DM",
    "authors": ["Luis Mario Chamizo Tiafi", "Juan Diego Chamizo Tiafi"],
    "affiliation": ["Fundación Universitaria de Popayán – Especialización en Data Analytics para Marketing Digital",
                    "Minería de Datos – Proyecto Final #1 (Grupo 6) · Docente: PhD. Cristian Camilo Ordoñez Quintero",
                    "Popayán, Colombia · Noviembre de 2026"],
    "abstract": (
        "El abandono de clientes (churn) reduce los ingresos recurrentes de las empresas de telecomunicaciones. Este trabajo "
        "presenta una solución de minería de datos, desarrollada con la metodología CRISP-DM, que predice qué clientes tienen alta "
        "probabilidad de abandono, explica por qué y recomienda acciones de retención personalizadas. Sobre el dataset Telco "
        f"Customer Churn ({miles(N)} clientes, {pct(tasa)} de churn) se realizó un análisis exploratorio con pruebas de asociación, "
        "una adaptación del análisis RFM a suscripciones y la creación de 16 variables derivadas. Se compararon cinco modelos "
        "(Regresión Logística, Random Forest, XGBoost, LightGBM y una red neuronal MLP) y un ensamble Stacking, ajustados con "
        "búsqueda aleatoria y validación cruzada estratificada. En el conjunto de prueba, los modelos obtuvieron un ROC-AUC entre "
        f"{num(auc_rng[0])} y {num(auc_rng[1])}; el modelo desplegado ({FINAL}) alcanzó ROC-AUC = {num(mf['ROC-AUC'])}, "
        f"F1 = {num(mf['F1'])} y Recall = {num(mf['Recall'])}, y captura el {pct(cap30)} de los abandonos contactando al 30 % de "
        "clientes de mayor riesgo. La explicabilidad con SHAP identificó como principales factores el tipo de contrato, la "
        "antigüedad, la fibra óptica y la ausencia de servicios de seguridad y soporte. La solución se desplegó en una aplicación "
        "web con predicción individual y por lotes, un dashboard de clientes en riesgo y un motor de recomendaciones ordenadas "
        "por su impacto SHAP."),
    "keywords": "churn, minería de datos, CRISP-DM, aprendizaje automático, XGBoost, LightGBM, SHAP, RFM, retención de clientes.",
    "blocks": B,
}
json.dump(spec, open(HERE / "contenido.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("contenido.json generado:", len(B), "bloques | modelo final:", FINAL)
