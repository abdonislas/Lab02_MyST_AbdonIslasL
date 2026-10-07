"""Optimización y walk-forward: ventanas sin traslape con embargo, meseta y restricción de operaciones."""

import optuna
import pandas as pd
import pytest

from src.data import cargar_datos, separar_train_test_validacion
from src.optimize import EMBARGO, MIN_OPERACIONES, PENALIZACION, RANGOS, seleccionar_meseta, ventanas_walk_forward


@pytest.fixture(scope="module")
def desarrollo():
    train, test, _ = separar_train_test_validacion(cargar_datos())
    return pd.concat([train, test])


@pytest.mark.parametrize("modo", ["rolling", "anchored"])
def test_ventanas_sin_traslape_y_con_embargo(desarrollo, modo):
    vs = ventanas_walk_forward(desarrollo, modo)
    assert len(vs) > 12
    for a, b, c, d in vs:
        assert a < b < c <= d
        assert desarrollo.index.get_loc(c) - desarrollo.index.get_loc(b) == EMBARGO + 1
    for (_, _, c1, d1), (_, _, c2, _) in zip(vs, vs[1:]):
        assert d1 < c2                                    # meses de prueba consecutivos y sin traslape
    if modo == "rolling":
        assert all((c - a).days <= 31 * 6 + 3 for a, _, c, _ in vs)


def test_meseta_prefiere_vecindario_estable():
    """Un pico aislado pierde contra una zona amplia con Calmar alto."""
    estudio = optuna.create_study(direction="maximize")
    base = {k: (lo + hi) / 2 if t is float else (lo + hi) // 2 for k, (lo, hi, t, _) in RANGOS.items()}
    dist = optuna.distributions
    distros = {k: (dist.IntDistribution(lo, hi) if t is int else dist.FloatDistribution(lo, hi))
               for k, (lo, hi, t, _) in RANGOS.items()}
    for i in range(15):                                   # meseta: Calmar 2 cerca del centro
        p = {**base, "bb_k": 2.0 + 0.01 * i}
        estudio.add_trial(optuna.trial.create_trial(params=p, distributions=distros, value=2.0))
    pico = {**{k: lo for k, (lo, hi, t, _) in RANGOS.items()}}
    estudio.add_trial(optuna.trial.create_trial(params=pico, distributions=distros, value=5.0))
    for i in range(12):                                   # el pico está rodeado de configuraciones malas
        estudio.add_trial(optuna.trial.create_trial(params={**pico, "bb_k": 1.5 + 0.01 * (i + 1)},
                                                    distributions=distros, value=-1.0))
    estudio.add_trial(optuna.trial.create_trial(params=base, distributions=distros, value=PENALIZACION))
    m = seleccionar_meseta(estudio)
    assert m["params"]["ema_n"] == base["ema_n"] and not m["es_argmax"]


def test_minimo_de_operaciones_es_positivo():
    assert MIN_OPERACIONES >= 1 and PENALIZACION < 0


def test_regimen_fuera_de_operar_no_se_optimiza(desarrollo, monkeypatch):
    """Un régimen fuera de OPERAR queda apagado sin correr Optuna; los demás sí se optimizan."""
    import src.optimize as opt
    from src.regimes import calcular_variables_regimen
    llamados = []
    monkeypatch.setattr(opt, "OPERAR", ("tendencia", "reversion"))
    monkeypatch.setattr(opt, "optimizar_regimen", lambda datos, etiquetas, r, *a, **k:
                        llamados.append(r) or {"parametros": {**opt.FIJOS, **opt.BASE}, "calmar": 1.0,
                                               "calmar_meseta": 1.0, "es_argmax": True, "factibles": 1,
                                               "pruebas": 1, "estudio": None})
    tramo = desarrollo.iloc[:1500]
    res = opt.optimizar_todos(tramo, calcular_variables_regimen(tramo), tramo.index[300], tramo.index[-1])
    assert sorted(llamados) == ["reversion", "tendencia"]
    assert res["parametros"]["crisis"] is None
    assert all(res["parametros"][r] is not None for r in ("tendencia", "reversion"))
