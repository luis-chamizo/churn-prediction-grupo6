# =====================================================
# MÓDULO COMPARTIDO: LIMPIEZA, FEATURE ENGINEERING (RFM),
# PREPROCESAMIENTO, EXPLICACIONES Y RECOMENDACIONES
# Proyecto 1 - Churn Prediction Inteligente - Grupo 6
# Luis Mario Chamizo Tiafi / Juan Diego Chamizo Tiafi
# =====================================================
"""Funciones reutilizadas por el notebook (entrenamiento) y por la
aplicación web (inferencia). Mantenerlas en un solo lugar garantiza que
los datos nuevos pasen EXACTAMENTE por el mismo preprocesamiento que los
datos de entrenamiento."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------
# COLUMNAS DEL DATASET ORIGINAL
# -----------------------------------------------------
ID_COL = "customerID"
TARGET = "Churn"

RAW_CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
RAW_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_FEATURES = RAW_CATEGORICAL + RAW_NUMERIC

# Valores válidos de cada variable categórica (usados por el formulario web
# y para validar archivos CSV cargados por lotes)
CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
PROTECTION_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAMING_COLS = ["StreamingTV", "StreamingMovies"]

# Variables creadas por el FeatureEngineer
ENG_NUMERIC = [
    "NumServicios", "NumProteccion", "NumStreaming", "GastoPromedioMensual",
    "VariacionTarifa", "CargoPorServicio", "DuracionContratoMeses",
    "PagoAutomatico", "TieneFamilia", "FibraSinProteccion",
    "R_Score", "F_Score", "M_Score", "RFM_Score",
]
ENG_CATEGORICAL = ["GrupoAntiguedad", "SegmentoRFM"]

MODEL_NUMERIC = RAW_NUMERIC + ENG_NUMERIC
MODEL_CATEGORICAL = RAW_CATEGORICAL + ENG_CATEGORICAL

# Nombres legibles en español (para gráficos, app y documento)
FEATURE_LABELS = {
    "gender": "Género", "SeniorCitizen": "Adulto mayor", "Partner": "Tiene pareja",
    "Dependents": "Tiene dependientes", "tenure": "Antigüedad (meses)",
    "PhoneService": "Servicio telefónico", "MultipleLines": "Múltiples líneas",
    "InternetService": "Tipo de internet", "OnlineSecurity": "Seguridad en línea",
    "OnlineBackup": "Respaldo en línea", "DeviceProtection": "Protección de equipo",
    "TechSupport": "Soporte técnico", "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming películas", "Contract": "Tipo de contrato",
    "PaperlessBilling": "Factura electrónica", "PaymentMethod": "Método de pago",
    "MonthlyCharges": "Cargo mensual", "TotalCharges": "Cargo total acumulado",
    "NumServicios": "N.º de servicios", "NumProteccion": "N.º servicios de protección",
    "NumStreaming": "N.º servicios de streaming",
    "GastoPromedioMensual": "Gasto promedio mensual",
    "VariacionTarifa": "Variación de tarifa", "CargoPorServicio": "Cargo por servicio",
    "DuracionContratoMeses": "Duración contrato (meses)",
    "PagoAutomatico": "Pago automático", "TieneFamilia": "Tiene familia",
    "FibraSinProteccion": "Fibra sin protección", "R_Score": "Puntaje R (antigüedad)",
    "F_Score": "Puntaje F (servicios)", "M_Score": "Puntaje M (gasto)",
    "RFM_Score": "Puntaje RFM total", "GrupoAntiguedad": "Grupo de antigüedad",
    "SegmentoRFM": "Segmento RFM",
}


# -----------------------------------------------------
# 1. LIMPIEZA
# -----------------------------------------------------
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica del dataset Telco.

    - Elimina espacios en columnas de texto.
    - Convierte TotalCharges a numérico. Los 11 registros vacíos
      corresponden a clientes con tenure = 0 (recién llegados que aún no
      han sido facturados), por lo que se imputan con 0.
    - Convierte la variable objetivo Churn a 0/1 si existe.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = (df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]).astype(float)
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    if TARGET in df.columns and df[TARGET].dtype != "int64":
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def validate_input(df: pd.DataFrame) -> list:
    """Devuelve una lista de problemas encontrados en un CSV de entrada."""
    problems = []
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        problems.append(f"Faltan columnas: {', '.join(missing)}")
        return problems
    for col, opts in CATEGORY_OPTIONS.items():
        bad = set(df[col].astype(str).str.strip().unique()) - set(opts)
        if bad:
            problems.append(f"Valores no reconocidos en {col}: {', '.join(sorted(bad)[:5])}")
    return problems


# -----------------------------------------------------
# 2. FEATURE ENGINEERING (RFM + variables derivadas)
# -----------------------------------------------------
def _score(values, edges):
    """Asigna un puntaje 1..5 según los cortes (cuantiles) aprendidos."""
    return np.clip(np.digitize(values, edges, right=True) + 1, 1, 5)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Crea variables derivadas y el análisis RFM adaptado a telecomunicaciones.

    Adaptación RFM (originalmente Recency-Frequency-Monetary):
      R = Relación / antigüedad del cliente (tenure). Más meses = más fidelidad.
      F = Frecuencia de uso: número de servicios contratados.
      M = Valor monetario: cargo mensual (MonthlyCharges).

    Los cortes por quintiles se aprenden SOLO con los datos de entrenamiento
    (fit) para evitar fuga de información hacia el conjunto de prueba.
    """

    def fit(self, X, y=None):
        X = X.copy()
        n_serv = self._num_servicios(X)
        q = [0.2, 0.4, 0.6, 0.8]
        self.r_edges_ = np.unique(np.quantile(X["tenure"], q))
        self.f_edges_ = np.unique(np.quantile(n_serv, q))
        self.m_edges_ = np.unique(np.quantile(X["MonthlyCharges"], q))
        return self

    @staticmethod
    def _num_servicios(X):
        n = sum((X[c] == "Yes").astype(int) for c in SERVICE_COLS)
        return n + (X["InternetService"] != "No").astype(int)

    @staticmethod
    def _segmento(r, f, m):
        if r >= 4 and f >= 4:
            return "Campeones"
        if r >= 4:
            return "Leales"
        if r <= 2 and m >= 4:
            return "Alto valor en riesgo"
        if r <= 2 and f <= 2:
            return "Nuevos de bajo compromiso"
        return "Potenciales"

    def transform(self, X):
        X = X.copy()
        tenure_safe = X["tenure"].clip(lower=1)
        X["NumServicios"] = self._num_servicios(X)
        X["NumProteccion"] = sum((X[c] == "Yes").astype(int) for c in PROTECTION_COLS)
        X["NumStreaming"] = sum((X[c] == "Yes").astype(int) for c in STREAMING_COLS)
        X["GastoPromedioMensual"] = np.where(
            X["tenure"] > 0, X["TotalCharges"] / tenure_safe, X["MonthlyCharges"])
        # >0 significa que hoy paga más que su promedio histórico (aumento de tarifa)
        X["VariacionTarifa"] = X["MonthlyCharges"] - X["GastoPromedioMensual"]
        X["CargoPorServicio"] = X["MonthlyCharges"] / (X["NumServicios"] + 1)
        X["DuracionContratoMeses"] = X["Contract"].map(
            {"Month-to-month": 1, "One year": 12, "Two year": 24}).fillna(1)
        X["PagoAutomatico"] = X["PaymentMethod"].str.contains("automatic").astype(int)
        X["TieneFamilia"] = ((X["Partner"] == "Yes") | (X["Dependents"] == "Yes")).astype(int)
        X["FibraSinProteccion"] = ((X["InternetService"] == "Fiber optic")
                                   & (X["OnlineSecurity"] != "Yes")
                                   & (X["TechSupport"] != "Yes")).astype(int)
        X["GrupoAntiguedad"] = pd.cut(
            X["tenure"], bins=[-1, 12, 24, 48, 72, np.inf],
            labels=["0-12 m", "13-24 m", "25-48 m", "49-72 m", "72+ m"]).astype(str)
        # ---- RFM ----
        X["R_Score"] = _score(X["tenure"], self.r_edges_)
        X["F_Score"] = _score(X["NumServicios"], self.f_edges_)
        X["M_Score"] = _score(X["MonthlyCharges"], self.m_edges_)
        X["RFM_Score"] = X["R_Score"] + X["F_Score"] + X["M_Score"]
        X["SegmentoRFM"] = [self._segmento(r, f, m) for r, f, m in
                            zip(X["R_Score"], X["F_Score"], X["M_Score"])]
        return X[MODEL_NUMERIC + MODEL_CATEGORICAL]

    def get_feature_names_out(self, input_features=None):
        return np.array(MODEL_NUMERIC + MODEL_CATEGORICAL, dtype=object)


# -----------------------------------------------------
# 3. PREPROCESAMIENTO (escalado + one-hot)
# -----------------------------------------------------
def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), MODEL_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                  sparse_output=False), MODEL_CATEGORICAL),
        ],
        verbose_feature_names_out=True,
    )


def original_feature(transformed_name: str) -> str:
    """'cat__Contract_Two year' -> 'Contract'; 'num__tenure' -> 'tenure'."""
    prefix, name = transformed_name.split("__", 1)
    if prefix == "num":
        return name
    for col in sorted(MODEL_CATEGORICAL, key=len, reverse=True):
        if name.startswith(col + "_"):
            return col
    return name


def aggregate_shap(shap_values: np.ndarray, transformed_names) -> pd.DataFrame:
    """Suma las contribuciones SHAP de las columnas one-hot para expresarlas
    en términos de la variable original (más fácil de interpretar)."""
    mapping = [original_feature(n) for n in transformed_names]
    df = pd.DataFrame(shap_values, columns=mapping)
    return df.T.groupby(level=0).sum().T


# -----------------------------------------------------
# 4. NIVELES DE RIESGO Y RECOMENDACIONES DE RETENCIÓN
# -----------------------------------------------------
def risk_level(prob: float, threshold: float, high: float) -> str:
    if prob >= high:
        return "Alto"
    if prob >= threshold:
        return "Medio"
    return "Bajo"


def value_segment(prob, annual_value, threshold, value_cut):
    """Matriz riesgo x valor para priorizar la inversión en retención."""
    alto_riesgo = prob >= threshold
    alto_valor = annual_value >= value_cut
    if alto_riesgo and alto_valor:
        return "1. Retener ya (alto riesgo, alto valor)"
    if alto_riesgo:
        return "2. Retención de bajo costo (alto riesgo, bajo valor)"
    if alto_valor:
        return "3. Fidelizar (bajo riesgo, alto valor)"
    return "4. Mantener (bajo riesgo, bajo valor)"


# Cada regla: (variable SHAP asociada, condición, acción, detalle, canal)
_RULES = [
    ("Contract", lambda r: r["Contract"] == "Month-to-month",
     "Migrar a contrato anual con beneficio",
     "Ofrecer 15-20 % de descuento o un mes gratis al pasar a contrato de 1 o 2 años.",
     "Llamada del equipo de retención"),
    ("tenure", lambda r: r["tenure"] <= 12,
     "Programa de bienvenida (onboarding)",
     "Seguimiento proactivo en los primeros 12 meses: llamada de bienvenida, revisión del plan a los 3 y 6 meses.",
     "Llamada + correo"),
    ("OnlineSecurity", lambda r: r["OnlineSecurity"] == "No",
     "Prueba gratuita de Seguridad en línea",
     "Activar 3 meses gratis del servicio de seguridad; aumenta la dependencia del servicio.",
     "SMS / app"),
    ("TechSupport", lambda r: r["TechSupport"] == "No",
     "Soporte técnico prioritario",
     "Incluir soporte técnico sin costo por 3 meses y asignar un agente de referencia.",
     "Llamada"),
    ("InternetService", lambda r: r["InternetService"] == "Fiber optic",
     "Revisión de la experiencia de fibra óptica",
     "Verificar calidad del servicio (velocidad, cortes) y ofrecer visita técnica preventiva.",
     "Visita técnica / llamada"),
    ("PaymentMethod", lambda r: r["PaymentMethod"] == "Electronic check",
     "Incentivo por pago automático",
     "Bonificación de USD 5 en la factura al domiciliar el pago con tarjeta o débito automático.",
     "Correo / app"),
    ("MonthlyCharges", lambda r: r["MonthlyCharges"] >= 70,
     "Optimizar el plan (right-sizing)",
     "Revisar consumo real y proponer un paquete con mejor relación precio/servicios.",
     "Asesor comercial"),
    ("PaperlessBilling", lambda r: r["PaperlessBilling"] == "Yes" and r["tenure"] <= 24,
     "Comunicación de valor en la factura",
     "Incluir en la factura electrónica un resumen de beneficios y ahorros obtenidos.",
     "Correo"),
    ("SeniorCitizen", lambda r: r["SeniorCitizen"] == 1,
     "Atención preferencial para adulto mayor",
     "Línea de atención dedicada y acompañamiento en el uso de los servicios.",
     "Llamada"),
    ("NumServicios", lambda r: r.get("NumServicios", 9) <= 2,
     "Venta cruzada de servicios (bundle)",
     "Ofrecer un paquete con servicios adicionales a precio preferencial para aumentar el vínculo.",
     "Asesor comercial"),
]


def recommend(row: dict, shap_by_feature: dict | None = None, top_n: int = 3) -> list:
    """Genera recomendaciones de retención personalizadas.

    Las reglas de negocio se activan según las características del cliente
    y se ORDENAN por la contribución SHAP de la variable asociada: primero
    se atacan los factores que más empujan a ESE cliente hacia el abandono.
    """
    recs = []
    for feat, cond, accion, detalle, canal in _RULES:
        try:
            ok = cond(row)
        except (KeyError, TypeError):
            ok = False
        if ok:
            impacto = float(shap_by_feature.get(feat, 0.0)) if shap_by_feature else 0.0
            recs.append({"accion": accion, "detalle": detalle, "canal": canal,
                         "variable": FEATURE_LABELS.get(feat, feat), "impacto": impacto})
    recs.sort(key=lambda d: d["impacto"], reverse=True)
    if not recs:
        recs.append({"accion": "Mantener relación y programa de lealtad",
                     "detalle": "Cliente sin factores de riesgo destacados: enviar beneficios de lealtad y encuestas de satisfacción.",
                     "canal": "Correo", "variable": "-", "impacto": 0.0})
    return recs[:top_n]
