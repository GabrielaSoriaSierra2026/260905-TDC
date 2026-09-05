"""
Modelo Streamlit: Optimización de mezcla de productos de tarjeta de crédito
mediante frontera eficiente tipo Markowitz.

La unidad optimizable del modelo es una combinación de:
    Segmento + Productos_Banco

Ejemplo de activo/cartera optimizable:
    Premium | 3 productos banco
    Clásica | 1 producto banco

El archivo Excel esperado debe contener, al menos, las hojas:
    - Clientes
    - Historico_Mensual

También puede contener:
    - Rendimientos_Mensuales
    - Diccionario_Datos
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize


# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Frontera eficiente TDC",
    page_icon="💳",
    layout="wide",
)

RISK_FREE_RATE_DEFAULT = 0.085
MIN_OBSERVATIONS = 6
EPS = 1e-10

REQUIRED_CLIENTES = [
    "Cliente_ID",
    "Segmento",
    "Edad_años",
    "Ingreso_Mensual_MXN",
    "Antigüedad_meses",
    "Productos_Banco",
    "Estatus",
    "Límite_Crédito_MXN",
    "Saldo_Promedio_MXN",
    "Utilización_Línea_pct",
    "Tasa_Interés_Anual_pct",
    "Score_Crediticio",
    "PD_12m_pct",
    "LGD_pct",
    "Cliente_Revolvente",
    "Ingreso_Intereses_Anual_MXN",
    "Costo_Fondeo_Anual_MXN",
    "Pérdida_Esperada_Anual_MXN",
    "Margen_Neto_Anual_MXN",
    "Rentabilidad_Estimada_Anual_pct",
]

REQUIRED_HISTORICO = [
    "Fecha",
    "Cliente_ID",
    "Segmento",
    "Límite_Crédito_MXN",
    "Saldo_Promedio_MXN",
    "Utilización_Línea_pct",
    "Compras_Mes_MXN",
    "Pago_Mes_MXN",
    "Mora_Días",
    "Ingreso_Intereses_Mes_MXN",
    "Comisiones_Mes_MXN",
    "Costo_Fondeo_Mes_MXN",
    "Pérdida_Crédito_Mes_MXN",
    "Margen_Neto_Mes_MXN",
    "Rendimiento_Mensual_pct",
]


@dataclass
class OptimizationResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float
    success: bool
    message: str


# -----------------------------------------------------------------------------
# Utilidades de lectura y limpieza
# -----------------------------------------------------------------------------
def normalize_column_name(col: str) -> str:
    """Normaliza nombres para tolerar espacios accidentales."""
    return str(col).strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, required_cols: List[str], sheet_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"La hoja '{sheet_name}' no contiene las columnas requeridas: {missing}"
        )


def pct_to_decimal(series: pd.Series) -> pd.Series:
    """
    Convierte porcentajes a decimales cuando vienen como 15 en lugar de 0.15.
    Si la columna ya está en decimal, la conserva.
    """
    s = pd.to_numeric(series, errors="coerce")
    median_abs = s.dropna().abs().median()
    if pd.notna(median_abs) and median_abs > 1.5:
        return s / 100.0
    return s


@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = {name: normalize_columns(pd.read_excel(xls, sheet_name=name)) for name in xls.sheet_names}

    if "Clientes" not in sheets or "Historico_Mensual" not in sheets:
        raise ValueError(
            "El Excel debe contener las hojas 'Clientes' y 'Historico_Mensual'."
        )

    clientes = sheets["Clientes"]
    historico = sheets["Historico_Mensual"]

    require_columns(clientes, REQUIRED_CLIENTES, "Clientes")
    require_columns(historico, REQUIRED_HISTORICO, "Historico_Mensual")

    clientes = clientes.copy()
    historico = historico.copy()

    clientes["Cliente_ID"] = clientes["Cliente_ID"].astype(str)
    historico["Cliente_ID"] = historico["Cliente_ID"].astype(str)
    historico["Fecha"] = pd.to_datetime(historico["Fecha"], errors="coerce")

    pct_cols_clientes = [
        "Utilización_Línea_pct",
        "Tasa_Interés_Anual_pct",
        "PD_12m_pct",
        "LGD_pct",
        "Rentabilidad_Estimada_Anual_pct",
    ]
    for col in pct_cols_clientes:
        clientes[col] = pct_to_decimal(clientes[col])

    pct_cols_hist = ["Utilización_Línea_pct", "Rendimiento_Mensual_pct"]
    for col in pct_cols_hist:
        historico[col] = pct_to_decimal(historico[col])

    numeric_clientes = [
        "Edad_años",
        "Ingreso_Mensual_MXN",
        "Antigüedad_meses",
        "Productos_Banco",
        "Límite_Crédito_MXN",
        "Saldo_Promedio_MXN",
        "Score_Crediticio",
        "Ingreso_Intereses_Anual_MXN",
        "Costo_Fondeo_Anual_MXN",
        "Pérdida_Esperada_Anual_MXN",
        "Margen_Neto_Anual_MXN",
    ]
    numeric_historico = [
        "Límite_Crédito_MXN",
        "Saldo_Promedio_MXN",
        "Compras_Mes_MXN",
        "Pago_Mes_MXN",
        "Mora_Días",
        "Ingreso_Intereses_Mes_MXN",
        "Comisiones_Mes_MXN",
        "Costo_Fondeo_Mes_MXN",
        "Pérdida_Crédito_Mes_MXN",
        "Margen_Neto_Mes_MXN",
    ]

    for col in numeric_clientes:
        clientes[col] = pd.to_numeric(clientes[col], errors="coerce")
    for col in numeric_historico:
        historico[col] = pd.to_numeric(historico[col], errors="coerce")

    return clientes, historico, sheets


# -----------------------------------------------------------------------------
# Construcción de activos optimizables
# -----------------------------------------------------------------------------
def build_universe(
    clientes: pd.DataFrame,
    historico: pd.DataFrame,
    group_cols: List[str],
    only_active: bool,
    min_clients_per_asset: int,
    min_months_per_asset: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye la matriz mensual de rendimientos por activo optimizable.

    Cada activo es un grupo definido por group_cols, por ejemplo:
        Segmento + Productos_Banco

    El rendimiento mensual de cada activo se calcula como promedio ponderado por
    Saldo_Promedio_MXN de los clientes que pertenecen al grupo.
    """
    clientes_f = clientes.copy()
    historico_f = historico.copy()

    if only_active:
        clientes_f = clientes_f[clientes_f["Estatus"].astype(str).str.lower().eq("activo")]

    group_cols_safe = [c for c in group_cols if c in clientes_f.columns]
    if not group_cols_safe:
        raise ValueError("Debe existir al menos una columna válida para agrupar los productos.")

    clientes_f["Activo_Markowitz"] = clientes_f[group_cols_safe].astype(str).agg(" | ".join, axis=1)

    base = historico_f.merge(
        clientes_f[["Cliente_ID", "Activo_Markowitz"] + group_cols_safe],
        on="Cliente_ID",
        how="inner",
        suffixes=("", "_cliente"),
    )

    base = base.dropna(subset=["Fecha", "Rendimiento_Mensual_pct", "Saldo_Promedio_MXN"])
    base = base[base["Saldo_Promedio_MXN"] > 0]

    if base.empty:
        raise ValueError("No hay observaciones válidas después de aplicar filtros.")

    # Rendimiento ponderado por saldo promedio mensual.
    base["ret_x_saldo"] = base["Rendimiento_Mensual_pct"] * base["Saldo_Promedio_MXN"]
    grouped = (
        base.groupby(["Fecha", "Activo_Markowitz"], as_index=False)
        .agg(
            ret_x_saldo=("ret_x_saldo", "sum"),
            saldo=("Saldo_Promedio_MXN", "sum"),
            clientes=("Cliente_ID", "nunique"),
        )
    )
    grouped["Rendimiento_Grupo"] = grouped["ret_x_saldo"] / grouped["saldo"]

    returns = grouped.pivot(index="Fecha", columns="Activo_Markowitz", values="Rendimiento_Grupo").sort_index()

    asset_counts = base.groupby("Activo_Markowitz")["Cliente_ID"].nunique()
    valid_assets = asset_counts[asset_counts >= min_clients_per_asset].index.tolist()
    returns = returns[valid_assets]

    valid_months = returns.notna().sum()
    returns = returns.loc[:, valid_months >= max(min_months_per_asset, MIN_OBSERVATIONS)]
    returns = returns.dropna(axis=0, how="all")
    returns = returns.fillna(returns.mean())

    if returns.shape[1] < 2:
        raise ValueError(
            "El universo optimizable requiere al menos dos activos/grupos con suficientes datos. "
            "Reduzca el mínimo de clientes por grupo o cambie la agrupación."
        )

    # Estadísticos descriptivos por activo.
    meta = (
        base.groupby("Activo_Markowitz")
        .agg(
            Clientes=("Cliente_ID", "nunique"),
            Saldo_Promedio_MXN=("Saldo_Promedio_MXN", "mean"),
            Compras_Mes_MXN=("Compras_Mes_MXN", "mean"),
            Pago_Mes_MXN=("Pago_Mes_MXN", "mean"),
            Mora_Días=("Mora_Días", "mean"),
            Margen_Neto_Mes_MXN=("Margen_Neto_Mes_MXN", "mean"),
            Perdida_Credito_Mes_MXN=("Pérdida_Crédito_Mes_MXN", "mean"),
            Utilizacion=("Utilización_Línea_pct", "mean"),
        )
    )

    client_meta = (
        clientes_f.groupby("Activo_Markowitz")
        .agg(
            Edad_Promedio=("Edad_años", "mean"),
            Ingreso_Mensual_Promedio_MXN=("Ingreso_Mensual_MXN", "mean"),
            Antiguedad_Promedio_meses=("Antigüedad_meses", "mean"),
            Score_Promedio=("Score_Crediticio", "mean"),
            PD_12m=("PD_12m_pct", "mean"),
            LGD=("LGD_pct", "mean"),
            Rentabilidad_Estimada_Anual=("Rentabilidad_Estimada_Anual_pct", "mean"),
            Margen_Neto_Anual_MXN=("Margen_Neto_Anual_MXN", "sum"),
            Limite_Credito_MXN=("Límite_Crédito_MXN", "sum"),
        )
    )

    meta = meta.join(client_meta, how="left")
    meta = meta.loc[returns.columns]

    return returns, meta


# -----------------------------------------------------------------------------
# Motor de optimización Markowitz
# -----------------------------------------------------------------------------
def portfolio_metrics(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
) -> Tuple[float, float, float]:
    ret = float(np.dot(weights, mu))
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
    sharpe = (ret - risk_free_rate) / vol if vol > EPS else np.nan
    return ret, vol, sharpe


def optimize_max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
    max_weight: float,
) -> OptimizationResult:
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = tuple((0, max_weight) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    def objective(w: np.ndarray) -> float:
        ret, vol, _ = portfolio_metrics(w, mu, cov, risk_free_rate)
        if vol <= EPS:
            return 1e6
        return -((ret - risk_free_rate) / vol)

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    ret, vol, sharpe = portfolio_metrics(res.x, mu, cov, risk_free_rate)
    return OptimizationResult(res.x, ret, vol, sharpe, bool(res.success), str(res.message))


def optimize_max_return_for_risk(
    mu: np.ndarray,
    cov: np.ndarray,
    max_volatility: float,
    max_weight: float,
    risk_free_rate: float,
) -> OptimizationResult:
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = tuple((0, max_weight) for _ in range(n))
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {
            "type": "ineq",
            "fun": lambda w: max_volatility - np.sqrt(np.dot(w.T, np.dot(cov, w))),
        },
    )

    res = minimize(lambda w: -np.dot(w, mu), x0, method="SLSQP", bounds=bounds, constraints=constraints)
    ret, vol, sharpe = portfolio_metrics(res.x, mu, cov, risk_free_rate)
    return OptimizationResult(res.x, ret, vol, sharpe, bool(res.success), str(res.message))


def optimize_min_volatility(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
    max_weight: float,
) -> OptimizationResult:
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = tuple((0, max_weight) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    res = minimize(
        lambda w: np.sqrt(np.dot(w.T, np.dot(cov, w))),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    ret, vol, sharpe = portfolio_metrics(res.x, mu, cov, risk_free_rate)
    return OptimizationResult(res.x, ret, vol, sharpe, bool(res.success), str(res.message))


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
    max_weight: float,
    points: int = 40,
) -> pd.DataFrame:
    n = len(mu)
    x0 = np.repeat(1 / n, n)
    bounds = tuple((0, max_weight) for _ in range(n))
    targets = np.linspace(float(np.min(mu)), float(np.max(mu)), points)
    rows = []

    for target in targets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mu) - t},
        )
        res = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(cov, w))),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if res.success:
            ret, vol, sharpe = portfolio_metrics(res.x, mu, cov, risk_free_rate)
            rows.append({"Riesgo": vol, "Rentabilidad": ret, "Sharpe": sharpe})
    return pd.DataFrame(rows).drop_duplicates().sort_values("Riesgo")


# -----------------------------------------------------------------------------
# Reportes y salidas
# -----------------------------------------------------------------------------
def build_weight_table(
    result: OptimizationResult,
    asset_names: List[str],
    meta: pd.DataFrame,
    min_display_weight: float = 0.0001,
) -> pd.DataFrame:
    df = pd.DataFrame({"Activo_Markowitz": asset_names, "Peso_Optimo": result.weights})
    df = df[df["Peso_Optimo"] >= min_display_weight].copy()
    df = df.merge(meta.reset_index(), on="Activo_Markowitz", how="left")
    df["Rentabilidad_Portafolio_Aportada"] = df["Peso_Optimo"] * df["Rentabilidad_Estimada_Anual"]
    return df.sort_values("Peso_Optimo", ascending=False)


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def clean_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text.strip("_")[:80]


# -----------------------------------------------------------------------------
# Interfaz Streamlit
# -----------------------------------------------------------------------------
st.title("💳 Optimización de mezcla de productos TDC con frontera eficiente")
st.caption(
    "Modelo de frontera eficiente tipo Markowitz aplicado a grupos de clientes de tarjeta de crédito. "
    "La mezcla óptima se calcula sobre combinaciones de Segmento y Productos_Banco, o sobre la agrupación seleccionada."
)

with st.sidebar:
    st.header("1. Cargar base")
    uploaded_file = st.file_uploader(
        "Archivo Excel del modelo TDC",
        type=["xlsx", "xls"],
        help="Debe incluir las hojas Clientes e Historico_Mensual.",
    )

    st.header("2. Definir activos")
    group_options = ["Segmento", "Productos_Banco", "Cliente_Revolvente", "Estatus"]
    group_cols = st.multiselect(
        "Agrupar mezcla optimizable por:",
        group_options,
        default=["Segmento", "Productos_Banco"],
    )
    only_active = st.checkbox("Usar solo clientes activos", value=True)
    min_clients = st.slider("Mínimo de clientes por grupo", 1, 100, 10, 1)
    min_months = st.slider("Mínimo de meses por grupo", 6, 24, 12, 1)

    st.header("3. Parámetros financieros")
    risk_free_rate = st.number_input(
        "Tasa libre de riesgo anual",
        min_value=0.0,
        max_value=1.0,
        value=RISK_FREE_RATE_DEFAULT,
        step=0.005,
        format="%.3f",
    )
    risk_penalty_lambda = st.slider(
        "Penalización por riesgo de crédito PD×LGD",
        0.0,
        3.0,
        1.0,
        0.1,
        help="Ajusta la rentabilidad esperada restando lambda × PD × LGD.",
    )
    max_weight = st.slider("Peso máximo por grupo", 0.10, 1.00, 0.45, 0.05)

    objective = st.selectbox(
        "Objetivo de optimización",
        [
            "Máximo Sharpe ajustado por riesgo",
            "Máxima rentabilidad con límite de riesgo",
            "Mínima volatilidad",
        ],
    )
    max_volatility = st.slider(
        "Riesgo anual máximo permitido",
        0.01,
        0.80,
        0.25,
        0.01,
        disabled=objective != "Máxima rentabilidad con límite de riesgo",
    )

if uploaded_file is None:
    st.info("Carga el archivo Excel para ejecutar el modelo. En GitHub puedes incluir un archivo de ejemplo en la carpeta `data/`.")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    clientes, historico, all_sheets = load_excel(file_bytes)
    returns_m, meta = build_universe(
        clientes=clientes,
        historico=historico,
        group_cols=group_cols,
        only_active=only_active,
        min_clients_per_asset=min_clients,
        min_months_per_asset=min_months,
    )
except Exception as exc:
    st.error(f"No fue posible preparar la base: {exc}")
    st.stop()

# Rendimientos esperados y matriz de covarianzas anualizadas.
asset_names = returns_m.columns.tolist()
monthly_mu = returns_m.mean()
annual_mu_historical = (1 + monthly_mu) ** 12 - 1
annual_cov = returns_m.cov() * 12

# Ajuste de rentabilidad por riesgo de crédito: retorno esperado - lambda × PD × LGD.
credit_penalty = (meta["PD_12m"].fillna(0) * meta["LGD"].fillna(0)) * risk_penalty_lambda
adjusted_mu = annual_mu_historical.reindex(asset_names).fillna(0) - credit_penalty.reindex(asset_names).fillna(0)
mu = adjusted_mu.values.astype(float)
cov = annual_cov.reindex(index=asset_names, columns=asset_names).fillna(0).values.astype(float)

# Regularización mínima para evitar matrices semidefinidas problemáticas.
cov = cov + np.eye(len(asset_names)) * 1e-8

if max_weight * len(asset_names) < 1:
    st.error(
        "El peso máximo por grupo es demasiado bajo para el número de activos disponibles. "
        "Aumenta el peso máximo o reduce el mínimo de clientes por grupo."
    )
    st.stop()

if objective == "Máximo Sharpe ajustado por riesgo":
    opt = optimize_max_sharpe(mu, cov, risk_free_rate, max_weight)
elif objective == "Máxima rentabilidad con límite de riesgo":
    opt = optimize_max_return_for_risk(mu, cov, max_volatility, max_weight, risk_free_rate)
else:
    opt = optimize_min_volatility(mu, cov, risk_free_rate, max_weight)

frontier = efficient_frontier(mu, cov, risk_free_rate, max_weight, points=50)
weights_df = build_weight_table(opt, asset_names, meta)

# -----------------------------------------------------------------------------
# Resultados visuales
# -----------------------------------------------------------------------------
if not opt.success:
    st.warning(f"El optimizador devolvió una advertencia: {opt.message}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rentabilidad esperada anual", f"{opt.expected_return:.2%}")
col2.metric("Riesgo anual", f"{opt.volatility:.2%}")
col3.metric("Sharpe ajustado", f"{opt.sharpe:.2f}" if np.isfinite(opt.sharpe) else "N/D")
col4.metric("Grupos optimizados", f"{len(asset_names):,}")

st.subheader("Frontera eficiente")
fig_frontier = px.line(
    frontier,
    x="Riesgo",
    y="Rentabilidad",
    markers=True,
    title="Frontera eficiente anualizada",
)
fig_frontier.add_trace(
    go.Scatter(
        x=[opt.volatility],
        y=[opt.expected_return],
        mode="markers",
        marker=dict(size=14, symbol="star"),
        name="Cartera óptima",
    )
)
fig_frontier.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
st.plotly_chart(fig_frontier, use_container_width=True)

st.subheader("Mezcla óptima de productos / grupos")
col_a, col_b = st.columns([1.2, 1])
with col_a:
    display_cols = [
        "Activo_Markowitz",
        "Peso_Optimo",
        "Clientes",
        "Rentabilidad_Estimada_Anual",
        "PD_12m",
        "LGD",
        "Score_Promedio",
        "Saldo_Promedio_MXN",
        "Mora_Días",
        "Margen_Neto_Anual_MXN",
    ]
    st.dataframe(
        weights_df[display_cols].style.format(
            {
                "Peso_Optimo": "{:.2%}",
                "Rentabilidad_Estimada_Anual": "{:.2%}",
                "PD_12m": "{:.2%}",
                "LGD": "{:.2%}",
                "Score_Promedio": "{:,.0f}",
                "Saldo_Promedio_MXN": "${:,.0f}",
                "Mora_Días": "{:.1f}",
                "Margen_Neto_Anual_MXN": "${:,.0f}",
            }
        ),
        use_container_width=True,
        height=420,
    )

with col_b:
    fig_weights = px.bar(
        weights_df,
        x="Peso_Optimo",
        y="Activo_Markowitz",
        orientation="h",
        title="Pesos óptimos por grupo",
    )
    fig_weights.update_layout(xaxis_tickformat=".0%", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_weights, use_container_width=True)

st.subheader("Diagnóstico del universo optimizable")
summary_df = meta.copy()
summary_df["Rentabilidad_Historica_Anual"] = annual_mu_historical.reindex(summary_df.index)
summary_df["Rentabilidad_Ajustada_Riesgo"] = adjusted_mu.reindex(summary_df.index)
summary_df["Volatilidad_Anual"] = np.sqrt(np.diag(cov))
summary_df = summary_df.reset_index().sort_values("Rentabilidad_Ajustada_Riesgo", ascending=False)

st.dataframe(
    summary_df.style.format(
        {
            "Rentabilidad_Historica_Anual": "{:.2%}",
            "Rentabilidad_Ajustada_Riesgo": "{:.2%}",
            "Volatilidad_Anual": "{:.2%}",
            "PD_12m": "{:.2%}",
            "LGD": "{:.2%}",
            "Utilizacion": "{:.2%}",
            "Saldo_Promedio_MXN": "${:,.0f}",
            "Compras_Mes_MXN": "${:,.0f}",
            "Pago_Mes_MXN": "${:,.0f}",
            "Margen_Neto_Anual_MXN": "${:,.0f}",
        }
    ),
    use_container_width=True,
    height=360,
)

st.subheader("Matriz de correlación")
corr = returns_m.corr()
fig_corr = px.imshow(
    corr,
    text_auto=False,
    aspect="auto",
    title="Correlación de rendimientos mensuales entre grupos",
    zmin=-1,
    zmax=1,
)
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("Descargas")
col_d1, col_d2, col_d3 = st.columns(3)
with col_d1:
    st.download_button(
        "Descargar mezcla óptima CSV",
        data=convert_df_to_csv(weights_df),
        file_name="mezcla_optima_tdc.csv",
        mime="text/csv",
    )
with col_d2:
    st.download_button(
        "Descargar diagnóstico CSV",
        data=convert_df_to_csv(summary_df),
        file_name="diagnostico_universo_tdc.csv",
        mime="text/csv",
    )
with col_d3:
    st.download_button(
        "Descargar rendimientos por grupo CSV",
        data=convert_df_to_csv(returns_m.reset_index()),
        file_name="rendimientos_grupos_tdc.csv",
        mime="text/csv",
    )

with st.expander("Interpretación del modelo"):
    st.markdown(
        """
        **Unidad optimizable:** cada activo es un grupo de clientes definido por la combinación seleccionada, por defecto `Segmento + Productos_Banco`.

        **Rentabilidad esperada:** se estima con el rendimiento mensual histórico promedio anualizado y se ajusta por riesgo de crédito mediante `PD_12m × LGD × penalización`.

        **Riesgo:** se mide con la matriz de covarianza anualizada de los rendimientos mensuales por grupo.

        **Optimización:** el modelo busca una mezcla de pesos que sume 100%, sin posiciones cortas, respetando el peso máximo por grupo definido por el usuario.

        **Uso recomendado:** comparar mezclas por segmento, número de productos, comportamiento revolvente y riesgo crediticio para identificar la combinación con mejor rentabilidad ajustada por riesgo.
        """
    )
