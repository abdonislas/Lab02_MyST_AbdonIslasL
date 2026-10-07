"""Señales: causalidad, fórmula 2 de 3, filtro de dirección, votos y salida según el estilo (rebote o ruptura)."""

import numpy as np
import pandas as pd
import pytest

from src.data import cargar_datos
from src.signals import generar_senales, regla_confirmacion


@pytest.fixture(scope="module")
def datos():
    return cargar_datos().iloc[:2500]


@pytest.mark.parametrize("t", [400, 1500, 2499])
def test_causalidad_senal(datos, t):
    """Recalcular sobre df.iloc[:t+1] da la misma fila (indicadores, votos, señal y salida) en t."""
    completa = generar_senales(datos)
    parcial = generar_senales(datos.iloc[: t + 1])
    pd.testing.assert_series_equal(parcial.iloc[-1], completa.iloc[t], check_names=False)


def test_regla_confirmacion_dos_de_tres():
    """Con un solo voto a favor no se abre posición; con dos o más, sí; votos encontrados se anulan."""
    votos = pd.DataFrame(
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [-1, 0, 0], [-1, -1, 0], [1, -1, 0], [1, 1, -1]],
        columns=["voto_bollinger", "voto_rsi", "voto_estocastico"],
    )
    assert regla_confirmacion(votos).tolist() == [0, 1, 1, 0, -1, 0, 1]
    assert regla_confirmacion(votos, 3).tolist() == [0, 0, 1, 0, 0, 0, 0]


def test_senal_respeta_filtro_y_preparacion(datos):
    s = generar_senales(datos)
    largos, cortos = s[s["senal"] == 1], s[s["senal"] == -1]
    assert len(largos) > 0 and len(cortos) > 0          # la estrategia opera en ambas direcciones
    assert (largos["preparacion"] == 1).all() and (largos["Close"] > largos["ema"]).all()
    assert (cortos["preparacion"] == -1).all() and (cortos["Close"] < cortos["ema"]).all()


def test_salida_por_media_de_bollinger(datos):
    s = generar_senales(datos, {"estilo": "rebote"}).dropna(subset=["bb_mid", "ema", "atr", "rsi", "stoch_k"])
    assert (s.loc[s["Close"] > s["bb_mid"], "salida"] == -1).all()   # rebote: cierra largos arriba de la media
    assert (s.loc[s["Close"] < s["bb_mid"], "salida"] == 1).all()    # rebote: cierra cortos abajo de la media
    r = generar_senales(datos, {"estilo": "ruptura"}).dropna(subset=["bb_mid", "ema", "atr", "rsi", "stoch_k"])
    assert (r.loc[r["Close"] < r["bb_mid"], "salida"] == -1).all()   # ruptura: cierra largos abajo de la media
    assert (r.loc[r["Close"] > r["bb_mid"], "salida"] == 1).all()    # ruptura: cierra cortos arriba de la media


@pytest.mark.parametrize("estilo", ["rebote", "ruptura"])
def test_votos_segun_estilo(datos, estilo):
    """Ruptura vota +1 en fuerza alcista (Close sobre la banda superior); rebote vota +1 en sobreventa."""
    s = generar_senales(datos, {"estilo": estilo, "memoria": 1}).dropna(subset=["bb_high", "bb_low"])
    arriba, abajo = s["Close"] > s["bb_high"], s["Close"] < s["bb_low"]
    signo = 1 if estilo == "ruptura" else -1
    assert (s.loc[arriba, "voto_bollinger"] == signo).all() and (s.loc[abajo, "voto_bollinger"] == -signo).all()


def test_atr_vectorizado_igual_a_ta(datos):
    import ta
    from src.signals import atr
    referencia = ta.volatility.AverageTrueRange(datos["High"], datos["Low"], datos["Close"], 14).average_true_range()
    np.testing.assert_allclose(atr(datos).iloc[13:], referencia.iloc[13:], rtol=1e-10)
