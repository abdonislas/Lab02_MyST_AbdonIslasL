"""Métricas de desempeño sobre la curva de valor del portafolio.

Convenciones declaradas:
- Retornos simples por vela de 1 hora; tasa libre de riesgo igual a 0.
- Anualización con 7 velas por sesión regular × 252 sesiones = 1,764 velas por año.
- Rendimiento anualizado compuesto: (V_final / V_inicial)^(velas_por_año / n) − 1.
- Calmar = rendimiento anualizado / |máximo drawdown|.
- Win Rate = operaciones con P&L neto de comisiones > 0 entre el total de operaciones cerradas.
- Payoff ratio = ganancia media de las operaciones ganadoras / |pérdida media de las perdedoras|.
- Turnover anual = monto operado (entradas + salidas) / equity promedio, por año.
"""

import numpy as np
import pandas as pd

VELAS_POR_ANIO = 7 * 252


def retornos(valor: pd.Series) -> pd.Series:
    """Retornos simples por vela de la curva de valor."""
    return valor.pct_change().dropna()


def rendimiento_anualizado(valor: pd.Series, velas_por_anio: int = VELAS_POR_ANIO) -> float:
    n = len(valor) - 1
    if n <= 0:
        return np.nan
    return (valor.iloc[-1] / valor.iloc[0]) ** (velas_por_anio / n) - 1


def sharpe(valor: pd.Series, velas_por_anio: int = VELAS_POR_ANIO) -> float:
    """Sharpe = E[r] / σ(r) · √(velas por año)."""
    r = retornos(valor)
    sd = r.std()
    return np.nan if sd == 0 or np.isnan(sd) else r.mean() / sd * np.sqrt(velas_por_anio)


def sortino(valor: pd.Series, velas_por_anio: int = VELAS_POR_ANIO) -> float:
    """Sortino = E[r] / σ₋ · √(velas por año), con σ₋ = √(E[min(r, 0)²])."""
    r = retornos(valor)
    sd_neg = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return np.nan if sd_neg == 0 or np.isnan(sd_neg) else r.mean() / sd_neg * np.sqrt(velas_por_anio)


def curva_drawdown(valor: pd.Series) -> pd.Series:
    """Drawdown en cada vela: V_t / máx(V_0..V_t) − 1 (valores ≤ 0)."""
    return valor / valor.cummax() - 1


def max_drawdown(valor: pd.Series) -> float:
    """Máximo drawdown como número negativo (−0.10 = caída de 10% desde el máximo)."""
    return float(curva_drawdown(valor).min())


def calmar(valor: pd.Series, velas_por_anio: int = VELAS_POR_ANIO) -> float:
    """Calmar = rendimiento anualizado / |máximo drawdown|."""
    mdd = abs(max_drawdown(valor))
    return np.nan if mdd == 0 else rendimiento_anualizado(valor, velas_por_anio) / mdd


def volatilidad_anualizada(valor: pd.Series, velas_por_anio: int = VELAS_POR_ANIO) -> float:
    """σ(r) · √(velas por año)."""
    return float(retornos(valor).std() * np.sqrt(velas_por_anio))


def payoff_ratio(operaciones: pd.DataFrame) -> float:
    """Ganancia media de las ganadoras entre la pérdida media (en valor absoluto) de las perdedoras."""
    if operaciones.empty or "pnl" not in operaciones:
        return np.nan
    pnl = operaciones["pnl"].dropna()
    ganan, pierden = pnl[pnl > 0], pnl[pnl < 0]
    if ganan.empty or pierden.empty:
        return np.nan
    return float(ganan.mean() / abs(pierden.mean()))


def win_rate(operaciones: pd.DataFrame) -> float:
    """Proporción de operaciones cerradas con P&L neto positivo."""
    if operaciones.empty or "pnl" not in operaciones:
        return np.nan
    pnl = operaciones["pnl"].dropna()
    return float((pnl > 0).mean()) if len(pnl) else np.nan


def resumen_metricas(valor: pd.Series, operaciones: pd.DataFrame) -> dict:
    """Métricas obligatorias más retorno, volatilidad, payoff y número de operaciones."""
    return {
        "retorno_total": valor.iloc[-1] / valor.iloc[0] - 1,
        "rendimiento_anualizado": rendimiento_anualizado(valor),
        "volatilidad": volatilidad_anualizada(valor),
        "sharpe": sharpe(valor),
        "sortino": sortino(valor),
        "calmar": calmar(valor),
        "max_drawdown": max_drawdown(valor),
        "win_rate": win_rate(operaciones),
        "payoff": payoff_ratio(operaciones),
        "operaciones": int(len(operaciones)),
        "turnover_anual": turnover(operaciones, valor)["turnover_anual"],
    }


def tabla_retornos(valor: pd.Series, frecuencia: str) -> pd.Series:
    """Retorno por periodo calendario: 'ME' mensual, 'QE' trimestral, 'YE' anual.

    El primer periodo se mide contra el valor inicial de la curva.
    """
    cierres = valor.resample(frecuencia).last().dropna()
    previo = pd.concat([pd.Series([valor.iloc[0]]), cierres.iloc[:-1]]).to_numpy()
    return pd.Series(cierres.to_numpy() / previo - 1, index=cierres.index, name="retorno")


def tablas_retornos(valor: pd.Series) -> dict:
    """Tablas de retornos mensuales, trimestrales y anuales."""
    return {nombre: tabla_retornos(valor, f) for nombre, f in
            (("mensual", "ME"), ("trimestral", "QE"), ("anual", "YE"))}


def metricas_por_regimen(valor: pd.Series, operaciones: pd.DataFrame, etiquetas: pd.Series) -> pd.DataFrame:
    """Métricas de la estrategia separadas por el régimen vigente (velas) y de entrada (operaciones)."""
    r = retornos(valor)
    reg = etiquetas.reindex(r.index)
    filas = {}
    for nombre, grupo in r.groupby(reg):
        ops = operaciones[operaciones.get("regimen") == nombre] if "regimen" in operaciones else pd.DataFrame()
        sd, sd_neg = grupo.std(), np.sqrt((np.minimum(grupo, 0) ** 2).mean())
        filas[nombre] = {
            "proporcion_tiempo": len(grupo) / len(r),
            "retorno_acumulado": float((1 + grupo).prod() - 1),
            "sharpe": grupo.mean() / sd * np.sqrt(VELAS_POR_ANIO) if sd > 0 else np.nan,
            "sortino": grupo.mean() / sd_neg * np.sqrt(VELAS_POR_ANIO) if sd_neg > 0 else np.nan,
            "operaciones": len(ops),
            "win_rate": win_rate(ops),
            "pnl": float(ops["pnl"].sum()) if len(ops) else 0.0,
        }
    return pd.DataFrame(filas).T


def prueba_diferencia_regimenes(operaciones: pd.DataFrame) -> dict:
    """Kruskal-Wallis sobre el retorno por operación entre regímenes (H0: misma distribución)."""
    from scipy.stats import kruskal

    ret = operaciones["pnl"] / (operaciones["acciones"] * operaciones["precio_entrada"])
    grupos = [g.to_numpy() for _, g in ret.groupby(operaciones["regimen"]) if len(g) >= 3]
    if len(grupos) < 2:
        return {"estadistico": np.nan, "p_valor": np.nan, "grupos": len(grupos)}
    h, p = kruskal(*grupos)
    return {"estadistico": float(h), "p_valor": float(p), "grupos": len(grupos)}


def turnover(operaciones: pd.DataFrame, equity: pd.Series, velas_por_anio: int = VELAS_POR_ANIO) -> dict:
    """Monto operado (entradas + salidas) entre el equity promedio, total y anualizado."""
    if operaciones.empty:
        return {"turnover_total": 0.0, "turnover_anual": 0.0}
    salida = operaciones["precio_salida"].fillna(operaciones["precio_entrada"]) if "precio_salida" in operaciones \
        else operaciones["precio_entrada"]
    monto = (operaciones["acciones"] * (operaciones["precio_entrada"] + salida)).sum()
    total = monto / equity.mean()
    return {"turnover_total": float(total), "turnover_anual": float(total * velas_por_anio / len(equity))}


def winrate_equilibrio(sl_atr: float, tp_atr: float, atr_rel: float, costo_ida_vuelta: float) -> float:
    """p* tal que la esperanza por operación es cero (todo en fracción del precio).

    Gana W = tp·ATR − C con probabilidad p y pierde L = sl·ATR + C con 1 − p:
    p·W − (1 − p)·L = 0  ⇒  p* = L / (W + L).
    """
    ganancia = tp_atr * atr_rel - costo_ida_vuelta
    perdida = sl_atr * atr_rel + costo_ida_vuelta
    return perdida / (ganancia + perdida) if ganancia > 0 else np.nan


def exposicion(posicion: pd.Series, cierre: pd.Series, equity: pd.Series) -> pd.Series:
    """Exposición bruta por vela: |acciones · Close| / equity (0 = fuera del mercado, 1 = 100% invertido)."""
    return (posicion.abs() * cierre.reindex(posicion.index) / equity.reindex(posicion.index)).fillna(0.0)


def buy_hold_escalado(cierre: pd.Series, fraccion: float, capital: float = 1_000_000.0) -> pd.Series:
    """Buy & hold con una fracción constante del capital invertida (rebalanceo continuo, sin costos): el
    benchmark con la misma exposición media que la estrategia."""
    r = cierre.pct_change().fillna(0.0)
    return capital * (1 + fraccion * r).cumprod()


def bootstrap_media(valores, n: int = 10_000, semilla: int = 42) -> dict:
    """Bootstrap de la media de los retornos por operación: intervalo de 95% y p-valor unilateral de H0: media ≤ 0
    (proporción de remuestreos con media ≤ 0)."""
    x = np.asarray(valores, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {"media": np.nan, "ic_bajo": np.nan, "ic_alto": np.nan, "p_valor": np.nan, "n": len(x)}
    rng = np.random.default_rng(semilla)
    medias = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {"media": float(x.mean()), "ic_bajo": float(np.quantile(medias, 0.025)),
            "ic_alto": float(np.quantile(medias, 0.975)), "p_valor": float((medias <= 0).mean()), "n": len(x)}


def retorno_por_operacion(operaciones: pd.DataFrame) -> pd.Series:
    """Retorno neto de cada operación sobre el monto invertido."""
    if operaciones.empty:
        return pd.Series(dtype=float)
    return operaciones["pnl"] / (operaciones["acciones"] * operaciones["precio_entrada"])
