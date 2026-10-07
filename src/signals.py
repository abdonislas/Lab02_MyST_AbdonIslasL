"""Indicadores técnicos y regla de confirmación 2 de 3 de la estrategia de "ruptura" (seguimiento de tendencia).

Idea: comprar cuando el precio rompe hacia arriba con fuerza dentro de una tendencia alcista y vender en corto
cuando rompe hacia abajo dentro de una tendencia bajista; la posición se cierra cuando el precio regresa por
debajo (largos) o por encima (cortos) de su media.

El estilo "rebote" (reversión a la media: comprar caídas y vender subidas) se conserva como comparación
(PARAMETROS_BASE["estilo"]) y se comparó en el walk-forward.

Indicadores (familia del PDF entre paréntesis):
- Bandas de Bollinger (bb_n, bb_k) (volatilidad).
- RSI de 14 velas (momento).
- Estocástico %K de 14 velas con %D de 3 (momento).
- EMA de ema_n velas como filtro de dirección (tendencia).
- ATR de 14 velas para el stop y el objetivo (volatilidad, no vota).

Votos al cierre de la vela t (cada uno en {−1, 0, +1}); un evento sigue vigente `memoria` velas.
Ruptura (oficial):
  v_vol = +1 si Close > banda superior,      −1 si Close < banda inferior
  v_rsi = +1 si RSI > 100 − rsi_bajo,        −1 si RSI < rsi_bajo
  v_sto = +1 si %K > 100 − sto_bajo,         −1 si %K < sto_bajo
Rebote (comparación): los mismos umbrales con el signo invertido (+1 en sobreventa, −1 en sobrecompra).

Regla de confirmación (fórmula):
  preparación_t = +1 si Σ_i 1[v_i,t = +1] ≥ 2 ;  −1 si Σ_i 1[v_i,t = −1] ≥ 2 ;  0 en otro caso.
Filtro de dirección: largo solo si Close_t > EMA_t; corto solo si Close_t < EMA_t.
  señal_t = +1 si preparación_t = +1 y Close_t > EMA_t ; −1 si preparación_t = −1 y Close_t < EMA_t ; 0 si no.
Salida (decidida al cierre de t):
  ruptura: salida_t = −1 (cierra largos) si Close_t < media de Bollinger; +1 (cierra cortos) si Close_t > media.
  rebote:  salida_t = −1 (cierra largos) si Close_t ≥ media de Bollinger; +1 (cierra cortos) si Close_t ≤ media.

Todo usa información hasta el cierre de t; el motor ejecuta en la apertura de t+1.
"""

import numpy as np
import pandas as pd
import ta

PARAMETROS_BASE = {
    "bb_n": 20, "bb_k": 2.0,
    "rsi_n": 14, "rsi_bajo": 30,
    "sto_n": 14, "sto_d": 3, "sto_bajo": 20,
    "ema_n": 140, "atr_n": 14,
    "memoria": 3,
    "estilo": "ruptura",
}
ESTILOS = ("ruptura", "rebote")

MODOS = ("dos_de_tres", "estricta", "solo_bollinger", "solo_rsi", "solo_estocastico")
VOTOS = ("voto_bollinger", "voto_rsi", "voto_estocastico")


def ema(df: pd.DataFrame, window: int):
    return ta.trend.EMAIndicator(df["Close"], window=window).ema_indicator()


def bollinger(df: pd.DataFrame, window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    bb = ta.volatility.BollingerBands(df["Close"], window=window, window_dev=std_dev)
    return pd.DataFrame({"bb_high": bb.bollinger_hband(), "bb_low": bb.bollinger_lband(),
                         "bb_mid": bb.bollinger_mavg()}, index=df.index)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR de Wilder, idéntico a ta.volatility.AverageTrueRange pero vectorizado (ta lo calcula con un ciclo
    de Python y es el cuello de botella de la optimización). NaN mientras calienta."""
    previo = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - previo).abs(), (df["Low"] - previo).abs()],
                   axis=1).max(axis=1)
    out = pd.Series(np.nan, index=df.index)
    if len(df) < window:
        return out
    semilla = pd.Series([tr.iloc[:window].mean()], index=[df.index[window - 1]])
    out.iloc[window - 1:] = pd.concat([semilla, tr.iloc[window:]]).ewm(alpha=1 / window, adjust=False).mean().to_numpy()
    return out


def rsi(df: pd.DataFrame, window: int = 14):
    return ta.momentum.RSIIndicator(df["Close"], window=window).rsi()


def estocastico(df: pd.DataFrame, window: int = 14, smooth: int = 3) -> pd.DataFrame:
    st = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=window, smooth_window=smooth)
    return pd.DataFrame({"stoch_k": st.stoch(), "stoch_d": st.stoch_signal()}, index=df.index)


def calcular_indicadores(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """OHLCV más Bollinger, RSI, Estocástico, EMA y ATR."""
    out = df[["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df else [])].copy()
    out = out.join(bollinger(df, int(p["bb_n"]), float(p["bb_k"])))
    out["rsi"] = rsi(df, int(p["rsi_n"]))
    out = out.join(estocastico(df, int(p["sto_n"]), int(p["sto_d"])))
    out["ema"] = ema(df, int(p["ema_n"]))
    out["atr"] = atr(df, int(p["atr_n"]))
    out.loc[out["atr"] <= 0, "atr"] = np.nan  # ta regresa 0 mientras calienta
    return out


def mantener_activo(evento: pd.Series, memoria: int) -> pd.Series:
    """Un evento sigue vigente `memoria` velas (incluida la suya). Si hay compra y venta vigentes, 0."""
    compra = (evento == 1).astype(int).rolling(memoria, min_periods=1).max()
    venta = (evento == -1).astype(int).rolling(memoria, min_periods=1).max()
    return (compra - venta).astype(int)


def calcular_votos(ind: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Tres votos con memoria. Ruptura: +1 en fuerza alcista (sobrecompra), −1 en fuerza bajista (sobreventa).
    Rebote: +1 en sobreventa, −1 en sobrecompra."""
    alto_rsi, alto_sto = 100 - p["rsi_bajo"], 100 - p["sto_bajo"]
    signo = 1 if p.get("estilo", "ruptura") == "ruptura" else -1
    eventos = {
        "voto_bollinger": signo * ((ind["Close"] > ind["bb_high"]).astype(int) - (ind["Close"] < ind["bb_low"]).astype(int)),
        "voto_rsi": signo * ((ind["rsi"] > alto_rsi).astype(int) - (ind["rsi"] < p["rsi_bajo"]).astype(int)),
        "voto_estocastico": signo * ((ind["stoch_k"] > alto_sto).astype(int) - (ind["stoch_k"] < p["sto_bajo"]).astype(int)),
    }
    return pd.DataFrame({k: mantener_activo(v, int(p["memoria"])) for k, v in eventos.items()}, index=ind.index)


def regla_confirmacion(votos: pd.DataFrame, minimo: int = 2) -> pd.Series:
    """+1 si al menos `minimo` votos son +1, −1 si al menos `minimo` son −1, 0 en otro caso."""
    compra = (votos == 1).sum(axis=1) >= minimo
    venta = (votos == -1).sum(axis=1) >= minimo
    return (compra & ~venta).astype(int) - (venta & ~compra).astype(int)


def generar_senales(df: pd.DataFrame, p: dict | None = None, modo: str = "dos_de_tres") -> pd.DataFrame:
    """Indicadores, votos, preparación, filtro, señal de entrada y señal de salida por vela.

    Modos: dos_de_tres (oficial), estricta (los tres votos) o un solo voto (solo_bollinger, solo_rsi,
    solo_estocastico) para la pregunta 1 del reporte.
    """
    if modo not in MODOS:
        raise ValueError(f"modo desconocido: {modo}")
    p = {**PARAMETROS_BASE, **(p or {})}
    if p["estilo"] not in ESTILOS:
        raise ValueError(f"estilo desconocido: {p['estilo']}")
    ind = calcular_indicadores(df, p)
    votos = calcular_votos(ind, p)
    if modo == "dos_de_tres":
        preparacion = regla_confirmacion(votos, 2)
    elif modo == "estricta":
        preparacion = regla_confirmacion(votos, 3)
    else:
        preparacion = votos[{"solo_bollinger": "voto_bollinger", "solo_rsi": "voto_rsi", "solo_estocastico": "voto_estocastico"}[modo]]
    filtro = (ind["Close"] > ind["ema"]).astype(int) - (ind["Close"] < ind["ema"]).astype(int)
    senal = preparacion.where(preparacion == filtro, 0)
    if p.get("estilo", "ruptura") == "ruptura":
        salida = pd.Series(np.where(ind["Close"] < ind["bb_mid"], -1, np.where(ind["Close"] > ind["bb_mid"], 1, 0)),
                           index=ind.index)
    else:
        salida = pd.Series(np.where(ind["Close"] >= ind["bb_mid"], -1, np.where(ind["Close"] <= ind["bb_mid"], 1, 0)),
                           index=ind.index)
    invalido = ind[["bb_mid", "rsi", "stoch_k", "ema", "atr"]].isna().any(axis=1)
    senal[invalido] = 0
    salida[invalido] = 0
    return ind.join(votos).assign(preparacion=preparacion, filtro=filtro, senal=senal.astype(int),
                                  salida=salida.astype(int))
