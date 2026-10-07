"""Motor: contabilidad, golden file a mano, stop primero, una sola posición, posiciones de varios días,
gaps, tamaño sin apalancamiento, salida con el θ del régimen de entrada y truncamiento de todo el pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import SIN_COSTOS, Costos, Posicion, backtest, buy_and_hold, tamano_por_fraccion
from src.data import cargar_datos
from src.optimize import BASE, correr
from src.regimes import KMedias, calcular_variables_regimen

HORAS = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]


def _sesiones(cierres_por_dia, inicio="2026-03-02"):
    """Velas de 1 hora: una lista de 7 cierres por sesión, en días hábiles consecutivos."""
    dias = pd.bdate_range(inicio, periods=len(cierres_por_dia))
    idx = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} {h}", tz="America/New_York") for d in dias for h in HORAS])
    c = np.concatenate([np.asarray(x, float) for x in cierres_por_dia])
    return pd.DataFrame({"Open": c, "High": c + 0.05, "Low": c - 0.05, "Close": c, "atr": 0.5}, index=idx)


@pytest.fixture(scope="module")
def datos():
    return cargar_datos().iloc[:2500]


@pytest.fixture(scope="module")
def resultado(datos):
    v = calcular_variables_regimen(datos)
    etiquetas = KMedias().ajustar(v.iloc[:1500]).etiquetar(v)
    return correr(datos, etiquetas, {r: BASE for r in ("tendencia", "reversion", "crisis")}, datos.index[400])


def test_contabilidad(resultado):
    """Equity = efectivo + posición · Close; comisiones = tasa · monto de cada apertura y cada cierre."""
    r = resultado
    ops = r["operaciones"]
    assert len(ops) > 5
    assert r["equity"].iloc[-1] == pytest.approx(r["efectivo"].iloc[-1] + r["posicion"].iloc[-1] * r["velas"]["Close"].iloc[-1])
    monto = (ops["acciones"] * (ops["precio_entrada"] + ops["precio_salida"])).sum()
    assert r["comisiones"] == pytest.approx(Costos().comision * monto)
    assert r["posicion"].iloc[-1] == 0                      # lo abierto al final se cierra con costos


def test_golden_file_equity_calculado_a_mano():
    """Cuentas en papel (capital 100,000; comisión 0.1%; sin spread ni impacto; SL 2 ATR, TP 3 ATR; 50% del equity):

    Señal +1 al cierre de 10:30 → compra en la apertura de 11:30 a 100. Acciones = 0.5 · 100,000 / (100 · 1.001)
    = 499.5005; comisión de entrada = 0.001 · 499.5005 · 100 = 49.95005 → efectivo = 50,000.
    Stop = 98, objetivo = 103. 12:30: equity = 50,000 + 499.5005 · 101 = 100,449.55.
    13:30: el máximo 103.5 toca el objetivo → venta a 103; comisión = 0.001 · 499.5005 · 103 = 51.4486.
    Efectivo final = 50,000 + 51,448.55 − 51.4486 = 101,397.10.
    """
    d = _sesiones([[100, 100, 100, 100, 101, 103.2, 103.2]])
    d.loc[d.index[4], ["High", "Low"]] = [101.5, 100.0]
    d.loc[d.index[5], ["Open", "High", "Low"]] = [101.0, 103.5, 100.8]
    senal = pd.Series([0, 1, 0, 0, 0, 0, 0], index=d.index)
    r = backtest(d, senal, theta={"sl_atr": 4.0, "tp_atr": 6.0, "max_velas": 50, "fraccion": 0.5},
                 costos=Costos(0.001, 0.0, 0.0), capital=100_000)
    acciones = 0.5 * 100_000 / (100 * 1.001)
    final = 50_000 + acciones * 103 * (1 - 0.001)
    assert r["operaciones"].iloc[0]["motivo"] == "take_profit"
    assert r["equity"].iloc[3] == pytest.approx(50_000 + acciones * 100.0)
    assert r["equity"].iloc[4] == pytest.approx(50_000 + acciones * 101.0)
    assert r["equity"].iloc[-1] == pytest.approx(final)
    assert r["operaciones"].iloc[0]["pnl"] == pytest.approx(final - 100_000)


def test_stop_y_objetivo_en_la_misma_vela_cierra_por_stop():
    pos = Posicion(lado=1, acciones=100, precio_entrada=100.0, stop_loss=98.0, take_profit=103.0)
    assert pos.salida_intrabar(alto=104.0, bajo=97.0, apertura=100.5) == (98.0, "stop_loss")
    corto = Posicion(lado=-1, acciones=100, precio_entrada=100.0, stop_loss=102.0, take_profit=97.0)
    assert corto.salida_intrabar(alto=103.0, bajo=96.0, apertura=100.0) == (102.0, "stop_loss")


def test_nunca_hay_dos_posiciones_abiertas():
    rng = np.random.default_rng(42)
    d = _sesiones([100 + rng.normal(0, 0.3, 7).cumsum() for _ in range(30)])
    r = backtest(d, pd.Series(rng.choice([-1, 0, 1], len(d)), index=d.index), costos=SIN_COSTOS)
    ops = r["operaciones"]
    assert len(ops) > 10
    assert (ops["entrada"].iloc[1:].to_numpy() >= ops["salida"].iloc[:-1].to_numpy()).all()


def test_posicion_se_mantiene_de_noche_y_sale_en_la_apertura_siguiente():
    d = _sesiones([[100] * 7, [101] * 7])
    senal = pd.Series(0, index=d.index)
    senal.iloc[2] = 1
    salida = pd.Series(0, index=d.index)
    salida.iloc[6] = -1                      # salida decidida al cierre de las 15:30 del día 1
    r = backtest(d, senal, theta={"sl_atr": 20, "tp_atr": 40, "max_velas": 100, "fraccion": 1.0},
                 costos=SIN_COSTOS, salida=salida)
    op = r["operaciones"].iloc[0]
    assert op["salida"] == d.index[7] and op["motivo"] == "salida"
    assert op["precio_salida"] == pytest.approx(101.0)      # captura el gap de la noche


def test_senal_de_las_1530_no_abre_al_dia_siguiente():
    d = _sesiones([[100] * 7, [100] * 7])
    senal = pd.Series(0, index=d.index)
    senal.iloc[6] = 1
    assert backtest(d, senal, costos=SIN_COSTOS)["operaciones"].empty


def test_stop_con_gap_se_llena_en_la_apertura():
    d = _sesiones([[100] * 7, [80] * 7])
    senal = pd.Series(0, index=d.index)
    senal.iloc[2] = 1
    r = backtest(d, senal, theta={"sl_atr": 4, "tp_atr": 40, "max_velas": 100, "fraccion": 1.0}, costos=SIN_COSTOS)
    op = r["operaciones"].iloc[0]
    assert op["motivo"] == "stop_loss" and op["precio_salida"] == pytest.approx(80.0)


def test_tamano_sin_apalancamiento():
    d = _sesiones([[100] * 7])
    senal = pd.Series(0, index=d.index)
    senal.iloc[1] = 1
    r = backtest(d, senal, theta={"fraccion": 0.6}, costos=SIN_COSTOS)
    op = r["operaciones"].iloc[0]
    assert op["acciones"] * op["precio_entrada"] == pytest.approx(600_000)
    assert tamano_por_fraccion(1_000_000, 100.0, 1.0, 0.00125) * 100.0 < 1_000_000
    with pytest.raises(ValueError):
        tamano_por_fraccion(1_000_000, 100.0, 1.5)


def test_salida_usa_el_theta_del_regimen_de_entrada():
    """Una posición abierta en 'reversion' ignora la señal de salida de 'tendencia' aunque el régimen cambie."""
    d = _sesiones([[100] * 7, [100] * 7])
    senal = pd.Series(0, index=d.index)
    senal.iloc[1] = 1
    regimen = pd.Series("reversion", index=d.index)
    regimen.iloc[4:] = "tendencia"
    salidas = pd.DataFrame({"reversion": 0, "tendencia": 0}, index=d.index)
    salidas.iloc[5, salidas.columns.get_loc("tendencia")] = -1   # no debe cerrar
    salidas.iloc[9, salidas.columns.get_loc("reversion")] = -1   # sí cierra, en la apertura de la vela 10
    r = backtest(d, senal, theta={"sl_atr": 20, "tp_atr": 40, "max_velas": 100}, costos=SIN_COSTOS,
                 salida=salidas, regimen=regimen)
    op = r["operaciones"].iloc[0]
    assert op["regimen"] == "reversion" and op["salida"] == d.index[10]


def test_buy_and_hold_paga_comision_en_ambos_lados():
    d = _sesiones([[100] * 7, [110] * 7])
    valor, op = buy_and_hold(d, Costos(0.001, 0, 0), capital=100_000)
    assert valor.iloc[-1] == pytest.approx(100_000 / 1.001 * 1.1 * 0.999)
    assert len(op) == 1


@pytest.mark.parametrize("t", [900, 1700, 2499])
def test_truncamiento_de_todo_el_pipeline(datos, t):
    """Indicadores → régimen → señales por régimen → backtest sobre df.iloc[:t+1]: el valor en t no cambia."""
    v = calcular_variables_regimen(datos)
    modelo = KMedias().ajustar(v.iloc[:800])
    theta = {r: BASE for r in ("tendencia", "reversion", "crisis")}
    completo = correr(datos, modelo.etiquetar(v), theta, datos.index[400])
    parcial_datos = datos.iloc[: t + 1]
    parcial = correr(parcial_datos, modelo.etiquetar(calcular_variables_regimen(parcial_datos)), theta, datos.index[400])
    # en t la posición abierta del parcial se cierra por fin de datos (con costos); antes de t todo coincide
    antes = completo["equity"].loc[: datos.index[t - 1]]
    assert len(antes) == len(parcial["equity"]) - 1
    np.testing.assert_allclose(parcial["equity"].iloc[:-1], antes, rtol=1e-12)


def test_trailing_stop_sube_con_el_precio_e_ignora_la_salida_por_la_media():
    """Con trailing = 1 el stop sigue al cierre a sl_atr ATR (nunca baja) y la señal de salida se ignora."""
    d = _sesiones([[100, 101, 102, 103, 104, 105, 106], [102] * 7])
    senal = pd.Series(0, index=d.index)
    senal.iloc[0] = 1
    salida = pd.Series(-1, index=d.index)            # sin trailing cerraría en la vela siguiente a la entrada
    theta = {"sl_atr": 4, "tp_atr": 40, "max_velas": 100, "fraccion": 1.0}
    sin = backtest(d, senal, theta=theta, costos=SIN_COSTOS, salida=salida)["operaciones"].iloc[0]
    con = backtest(d, senal, theta={**theta, "trailing": 1.0}, costos=SIN_COSTOS, salida=salida)["operaciones"].iloc[0]
    assert sin["motivo"] == "salida" and sin["salida"] == d.index[2]
    assert con["motivo"] == "stop_loss" and con["salida"] == d.index[7]     # el stop subió de 99 a 104
    assert con["precio_salida"] == pytest.approx(102.0)                    # gap: se llena en la apertura
    assert con["pnl"] > 0
