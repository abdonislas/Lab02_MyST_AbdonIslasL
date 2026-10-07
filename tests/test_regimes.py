"""Régimen: causalidad de las variables y de las etiquetas (K-means y HMM filtrado) y K-means congelado."""

import json

import numpy as np
import pytest

from src.data import cargar_datos, separar_train_test_validacion
from src.regimes import HMM, KMedias, calcular_variables_regimen, regime_features


@pytest.fixture(scope="module")
def datos():
    return cargar_datos()


@pytest.fixture(scope="module")
def modelo(datos):
    train, _, _ = separar_train_test_validacion(datos)
    return KMedias().ajustar(calcular_variables_regimen(train))


@pytest.mark.parametrize("t", [500, 2222, 4000])
def test_regimen_no_cambia_con_datos_posteriores(modelo, datos, t):
    completa = modelo.etiquetar(calcular_variables_regimen(datos))
    parcial = modelo.etiquetar(calcular_variables_regimen(datos.iloc[: t + 1]))
    assert (parcial == completa.iloc[: t + 1]).all()


@pytest.mark.parametrize("t", [100, 1234, 3000])
def test_truncamiento_por_columna(datos, t):
    completo, parcial = regime_features(datos), regime_features(datos.iloc[: t + 1])
    for col in completo.columns:
        a, b = parcial[col].iloc[-1], completo[col].iloc[t]
        assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-9, abs=1e-12), col


def test_hmm_filtrado_no_cambia_con_datos_posteriores(datos):
    """La etiqueta filtrada del HMM en t solo depende de datos hasta t (Viterbi no cumple esto)."""
    train, _, _ = separar_train_test_validacion(datos)
    hmm = HMM().ajustar(regime_features(train))
    completa = hmm.etiquetar(regime_features(datos))
    for t in (2000, 4000):
        assert (hmm.etiquetar(regime_features(datos.iloc[: t + 1])) == completa.iloc[: t + 1]).all()


def test_kmeans_congelado_etiqueta_igual(modelo, datos):
    v = calcular_variables_regimen(datos)
    copia = KMedias.desde_dict(json.loads(json.dumps(modelo.a_dict())))
    assert (copia.etiquetar(v) == modelo.etiquetar(v)).all()
