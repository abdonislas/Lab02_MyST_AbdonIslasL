"""Optimización por régimen con Optuna, walk-forward y análisis de robustez.

- Parámetros por régimen: θ = {θ_tendencia, θ_reversion, θ_crisis}. Cada θ_r se optimiza por separado con un
  backtest que solo abre posiciones en las velas del régimen r (las posiciones se gestionan con θ_r hasta que
  cierran, aunque el régimen cambie).
- Objetivo: θ_r* = argmax Calmar(backtest(train, θ_r)), con comisión, spread e impacto.
- Restricción: una configuración con menos de `MIN_OPERACIONES` operaciones en la ventana vale −10 en lugar
  de su Calmar, para que el optimizador no premie tres operaciones ganadoras.
- Búsqueda: primero `N_ALEATORIAS` pruebas aleatorias y luego `N_TPE` pruebas bayesianas (TPE de Optuna),
  150 por régimen por ventana, con semilla fija.
- θ_r* se toma del centro de la mejor meseta (promedio de los k vecinos más cercanos en el espacio normalizado
  de parámetros), no del argmax literal.
- Por régimen se optimizan 5 parámetros (RANGOS); el resto queda fijo en valores estándar (FIJOS).
- Si el mejor Calmar o el de la meseta de un régimen no es positivo, ese régimen queda apagado (no abre posiciones).
- Los regímenes fuera de OPERAR no se optimizan ni abren posiciones.
- Walk-forward: 1 mes de prueba, paso mensual. Anchored (oficial: entrenamiento desde el inicio de los datos,
  al menos 6 meses) y rolling (comparación: últimos 6 meses). Embargo de una sesión (7 velas) entre
  entrenamiento y prueba.
  La curva fuera de muestra es una sola simulación: cada mes usa las señales del θ congelado de su ventana y
  una posición abierta pasa al mes siguiente (no se cierra a la fuerza al cambiar de ventana).
"""

import json
import time

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from src.backtest import COLUMNAS_OPERACION, THETA_OPERACION, Costos, backtest
from src.metrics import calmar, rendimiento_anualizado, resumen_metricas, sharpe
from src.regimes import REGIMENES, KMedias, calcular_variables_regimen
from src.signals import PARAMETROS_BASE, atr, generar_senales

N_ALEATORIAS, N_TPE = 50, 100
MIN_OPERACIONES = 6
MESES_TRAIN = 6
EMBARGO = 7
VECINOS_MESETA = 10
PENALIZACION = -10.0
SEMILLA = 42

# Estilo de la estrategia (src/signals.py): "ruptura" (seguir la tendencia) o "rebote" (reversión a la media).
# Cada estilo tiene su propio espacio de búsqueda y sus fijos; se eligió con el walk-forward de train + test.
ESTILO = "ruptura"
# Regímenes en los que se opera; un régimen fuera de OPERAR no abre posiciones (una posición abierta en otro
# régimen sí sigue viva si el mercado entra en él: regla de transición).
OPERAR = ("tendencia", "reversion", "crisis")

# Espacio de búsqueda por régimen: (mínimo, máximo, tipo, justificación). Solo los 5 parámetros que más
# mueven el resultado se optimizan por régimen; el resto queda fijo en valores estándar (FIJOS).
_CONFIG = {
    "rebote": {
        "rsi": (20, 40, int, "rebote: sobreventa si RSI14 < rsi_bajo; sobrecompra si RSI14 > 100 − rsi_bajo"),
        "fijos": {"tp_atr": 8.0, "max_velas": 35},   # la reversión se completa en pocas sesiones
        "rsi_base": 30,
    },
    "ruptura": {
        "rsi": (30, 50, int, "ruptura: fuerza alcista si RSI14 > 100 − rsi_bajo (50 a 70); bajista si RSI14 < rsi_bajo"),
        "fijos": {"tp_atr": 12.0, "max_velas": 70},  # una tendencia necesita más espacio y más tiempo
        "rsi_base": 40,
    },
}[ESTILO]
RANGOS = {
    "bb_k": (1.5, 3.0, float, "qué tan extrema debe ser la ruptura o el rebote respecto a la banda"),
    "rsi_bajo": _CONFIG["rsi"],
    "ema_n": (50, 300, int, "filtro de dirección: EMA de 7 a 43 sesiones"),
    "sl_atr": (2.0, 10.0, float, "stop-loss de 2 a 10 ATR horarios"),
    "fraccion": (0.2, 1.0, float, "tamaño de posición como fracción del equity, sin apalancamiento"),
}
FIJOS = {"estilo": ESTILO, "bb_n": 20, "sto_bajo": 20, "memoria": 3, **_CONFIG["fijos"]}
BASE = {"bb_k": 2.0, "rsi_bajo": _CONFIG["rsi_base"], "ema_n": 140, "sl_atr": 6.0, "fraccion": 1.0}
# Extensiones evaluadas solo en el walk-forward de train + test, después de congelar θ* (python main.py
# --extensiones). No cambian θ* ni validation: documentan qué se cambiaría en una siguiente versión.
EXTENSIONES = {
    "trailing_stop": {"fijos": {"trailing": 1.0}, "descripcion": "trailing stop en ATR en lugar de salir en la media"},
    "tres_de_tres": {"confirmacion": "estricta", "descripcion": "confirmación 3 de 3 en lugar de 2 de 3"},
    "fraccion_100": {"fijos": {"fraccion": 1.0}, "descripcion": "fracción fija de 100% del equity (Optuna no la elige)"},
}


def fijos_de(ext: dict | None) -> dict:
    return {**FIJOS, **(ext or {}).get("fijos", {})}


def rangos_de(ext: dict | None) -> dict:
    return {k: v for k, v in RANGOS.items() if k not in (ext or {}).get("fijos", {})}


def confirmacion_de(ext: dict | None) -> str:
    return (ext or {}).get("confirmacion", "dos_de_tres")


MODO_OFICIAL = "anchored"  # esquema del walk-forward oficial y del θ* final; "rolling" se reporta como comparación
UMBRAL_MESETA = 0.0  # un régimen se activa solo si el Calmar promedio de su meseta supera este umbral

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------------------- señales por régimen
def separar(p: dict) -> tuple[dict, dict]:
    """Parámetros de señal y parámetros de operación (stop, objetivo, holding y tamaño)."""
    senal = {k: v for k, v in p.items() if k not in COLUMNAS_OPERACION}
    operacion = {k: float(p.get(k, THETA_OPERACION[k])) for k in COLUMNAS_OPERACION}
    return senal, operacion


def combinar(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict, modo: str = "dos_de_tres"):
    """Señales con el θ del régimen vigente en cada vela.

    `parametros` = {régimen: dict o None}; None = régimen apagado (sin entradas).
    Regresa (velas, salidas): velas trae OHLCV, atr, regimen, senal y los parámetros de operación por vela;
    salidas trae una columna por régimen con su señal de salida (la usa cada posición según su régimen de entrada).
    """
    reg = etiquetas.reindex(datos.index).fillna("sin_datos")
    velas = datos[["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in datos else [])].copy()
    velas["atr"] = atr(datos, PARAMETROS_BASE["atr_n"]).where(lambda x: x > 0)
    velas["regimen"] = reg
    velas["senal"] = 0
    for k, v in THETA_OPERACION.items():
        velas[k] = float(v)
    salidas = {}
    for r, p in parametros.items():
        if p is None:
            continue
        p_senal, p_operacion = separar(p)
        propia = generar_senales(datos, p_senal, modo)
        mascara = reg == r
        velas.loc[mascara, "senal"] = propia.loc[mascara, "senal"]
        for k, v in p_operacion.items():
            velas.loc[mascara, k] = v
        salidas[r] = propia["salida"]
    return velas, pd.DataFrame(salidas, index=datos.index)


def correr(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict, desde, hasta=None,
           costos: Costos = Costos(), modo: str = "dos_de_tres") -> dict:
    """Backtest entre `desde` y `hasta`. Los indicadores se calculan con toda la historia hasta `hasta`
    (nunca después), así que el resultado no depende de dónde empiece el archivo de datos."""
    historia = datos.loc[:hasta] if hasta is not None else datos
    velas, salidas = combinar(historia, etiquetas, parametros, modo)
    velas, salidas = velas.loc[desde:], salidas.loc[desde:]
    r = backtest(velas, velas["senal"], costos=costos, por_vela=velas[list(COLUMNAS_OPERACION)],
                 salida=salidas, regimen=velas["regimen"])
    r["velas"], r["salidas"] = velas, salidas
    return r


# ----------------------------------------------------------------------------------- optimización
def sugerir(trial, ext: dict | None = None) -> dict:
    return {**fijos_de(ext), **{k: (trial.suggest_int(k, lo, hi) if t is int else trial.suggest_float(k, lo, hi))
                                for k, (lo, hi, t, _) in rangos_de(ext).items()}}


def seleccionar_meseta(estudio, vecinos: int = VECINOS_MESETA, rangos: dict | None = None) -> dict:
    """Centro de la mejor meseta: el trial cuyo vecindario (k vecinos más cercanos, parámetros normalizados
    a [0, 1]) tiene el Calmar promedio más alto. Con pocos trials factibles, el argmax."""
    trials = [t for t in estudio.trials if t.value is not None and t.value > PENALIZACION]
    if not trials:
        return {"params": None, "valor": np.nan, "meseta": np.nan, "es_argmax": True}
    if len(trials) <= vecinos:
        b = max(trials, key=lambda t: t.value)
        return {"params": b.params, "valor": b.value, "meseta": b.value, "es_argmax": True}
    rangos = rangos or RANGOS
    X = np.array([[(t.params[k] - rangos[k][0]) / (rangos[k][1] - rangos[k][0]) for k in rangos] for t in trials])
    v = np.array([t.value for t in trials])
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    promedio = v[np.argsort(d, axis=1)[:, :vecinos]].mean(axis=1)
    i = int(np.argmax(promedio))
    return {"params": trials[i].params, "valor": float(v[i]), "meseta": float(promedio[i]),
            "es_argmax": bool(trials[i].number == estudio.best_trial.number)}


def optimizar_regimen(datos: pd.DataFrame, etiquetas: pd.Series, regimen: str, desde, hasta,
                      semilla: int = SEMILLA, n_aleatorias: int = N_ALEATORIAS, n_tpe: int = N_TPE,
                      ext: dict | None = None) -> dict:
    """Random search seguido de TPE maximizando el Calmar del backtest que solo abre en `regimen`."""
    tramo = etiquetas.loc[desde:hasta]
    if not (tramo == regimen).any():
        return {"parametros": None, "calmar": np.nan, "calmar_meseta": np.nan, "es_argmax": True,
                "factibles": 0, "pruebas": 0, "estudio": None}

    def objetivo(trial):
        r = correr(datos, etiquetas, {regimen: sugerir(trial, ext)}, desde, hasta, modo=confirmacion_de(ext))
        n = len(r["operaciones"])
        trial.set_user_attr("operaciones", n)
        if n < MIN_OPERACIONES:
            return PENALIZACION
        valor = calmar(r["equity"])
        return float(valor) if np.isfinite(valor) else PENALIZACION

    estudio = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(n_startup_trials=n_aleatorias, seed=semilla))
    estudio.enqueue_trial({k: v for k, v in BASE.items() if k in rangos_de(ext)})
    estudio.optimize(objetivo, n_trials=n_aleatorias + n_tpe)
    factibles = sum(1 for t in estudio.trials if t.value is not None and t.value > PENALIZACION)
    m = seleccionar_meseta(estudio, rangos=rangos_de(ext))
    activo = m["params"] is not None and estudio.best_value > 0 and m["valor"] > 0 and m["meseta"] > UMBRAL_MESETA
    return {"parametros": {**fijos_de(ext), **m["params"]} if activo else None, "calmar": float(estudio.best_value),
            "calmar_meseta": m["meseta"], "es_argmax": m["es_argmax"], "factibles": factibles,
            "pruebas": len(estudio.trials), "estudio": estudio}


def optimizar_todos(datos: pd.DataFrame, variables: pd.DataFrame, desde, hasta, semilla: int = SEMILLA,
                    ext: dict | None = None) -> dict:
    """K-means ajustado con la historia hasta `hasta` y Optuna por régimen en [desde, hasta]."""
    modelo = KMedias().ajustar(variables.loc[:hasta])
    etiquetas = modelo.etiquetar(variables)
    por_regimen = {r: (optimizar_regimen(datos, etiquetas, r, desde, hasta, semilla + i, ext=ext) if r in OPERAR else
                       {"parametros": None, "calmar": np.nan, "calmar_meseta": np.nan, "es_argmax": True,
                        "factibles": 0, "pruebas": 0, "estudio": None})
                   for i, r in enumerate(REGIMENES)}
    return {"modelo": modelo, "etiquetas": etiquetas, "por_regimen": por_regimen,
            "parametros": {r: res["parametros"] for r, res in por_regimen.items()}}


# ------------------------------------------------------------------------------------ walk-forward
def ventanas_walk_forward(datos: pd.DataFrame, modo: str = "rolling", meses_train: int = MESES_TRAIN) -> list:
    """(inicio_train, fin_train, inicio_test, fin_test) con prueba de 1 mes calendario y paso mensual.

    rolling: los últimos `meses_train` meses; anchored: desde el inicio de los datos. El entrenamiento termina
    `EMBARGO` velas antes de la prueba.
    """
    meses = pd.Series(datos.index, index=datos.index).groupby(datos.index.tz_localize(None).to_period("M"))
    limites = [(g.iloc[0], g.iloc[-1]) for _, g in meses]
    salida = []
    for i in range(meses_train, len(limites)):
        inicio_test, fin_test = limites[i]
        fin_train = datos.index[datos.index.get_indexer([inicio_test])[0] - 1 - EMBARGO]
        inicio_train = datos.index[0] if modo == "anchored" else limites[i - meses_train][0]
        salida.append((inicio_train, fin_train, inicio_test, fin_test))
    return salida


def optimizar_ventana(datos, variables, a, b, c, d, semilla: int = SEMILLA, ext: dict | None = None) -> dict:
    """Una ventana: régimen y θ por régimen con [a, b]; evaluación congelada en [c, d]."""
    t0 = time.perf_counter()
    opt = optimizar_todos(datos.loc[:b], variables.loc[:b], a, b, semilla, ext=ext)
    etiquetas = opt["modelo"].etiquetar(variables.loc[:d])
    parametros = opt["parametros"]
    modo = confirmacion_de(ext)
    dentro = correr(datos, opt["etiquetas"], parametros, a, b, modo=modo)
    fuera = correr(datos, etiquetas, parametros, c, d, modo=modo)
    return {"inicio_train": a, "fin_train": b, "inicio_test": c, "fin_test": d, "parametros": parametros,
            "modelo": opt["modelo"].a_dict(),
            "calmar_por_regimen": {r: x["calmar"] for r, x in opt["por_regimen"].items()},
            "pruebas": sum(x["pruebas"] for x in opt["por_regimen"].values()),
            "dentro": resumen_metricas(dentro["equity"], dentro["operaciones"]),
            "fuera": resumen_metricas(fuera["equity"], fuera["operaciones"]),
            "velas_fuera": fuera["velas"], "salidas_fuera": fuera["salidas"],
            "segundos": time.perf_counter() - t0}


def _ventana_en_proceso(*args):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optimizar_ventana(*args)


def simular_encadenado(velas: list, salidas: list, costos: Costos = Costos()) -> dict:
    """Una sola simulación sobre los meses de prueba concatenados (la posición pasa de un mes al siguiente)."""
    v = pd.concat(velas)
    s = pd.concat(salidas).reindex(v.index).fillna(0).astype(int)
    return backtest(v, v["senal"], costos=costos, por_vela=v[list(COLUMNAS_OPERACION)], salida=s,
                    regimen=v["regimen"])


def walk_forward(datos: pd.DataFrame, modo: str = "rolling", meses_train: int = MESES_TRAIN,
                 n_jobs: int = -1, ext: dict | None = None) -> dict:
    """Todas las ventanas en paralelo y la curva fuera de muestra encadenada.

    Walk-forward efficiency = rendimiento anualizado fuera de muestra / promedio del anualizado dentro.
    """
    variables = calcular_variables_regimen(datos)
    vs = ventanas_walk_forward(datos, modo, meses_train)
    t0 = time.perf_counter()
    res = Parallel(n_jobs=n_jobs)(delayed(_ventana_en_proceso)(datos, variables, *w, SEMILLA + 10 * i, ext)
                                  for i, w in enumerate(vs))
    sim = simular_encadenado([r["velas_fuera"] for r in res], [r["salidas_fuera"] for r in res])
    resumen = pd.DataFrame([{
        "inicio_test": r["inicio_test"].date(), "calmar_dentro": r["dentro"]["calmar"],
        "calmar_fuera": r["fuera"]["calmar"], "retorno_dentro": r["dentro"]["retorno_total"],
        "retorno_fuera": r["fuera"]["retorno_total"], "anualizado_dentro": r["dentro"]["rendimiento_anualizado"],
        "operaciones_fuera": r["fuera"]["operaciones"],
        **{f"activo_{k}": v is not None for k, v in r["parametros"].items()},
    } for r in res])
    dentro = resumen["anualizado_dentro"].mean()
    fuera = rendimiento_anualizado(sim["equity"])
    return {"modo": modo, "ventanas": res, "resumen": resumen, "valor": sim["equity"], "posicion": sim["posicion"],
            "operaciones": sim["operaciones"], "regimen": pd.concat([r["velas_fuera"]["regimen"] for r in res]),
            "comisiones": sim["comisiones"], "deslizamiento": sim["deslizamiento"],
            "segundos": time.perf_counter() - t0, "configuraciones": sum(r["pruebas"] for r in res),
            "anualizado_dentro": dentro, "anualizado_fuera": fuera,
            "eficiencia": fuera / dentro if dentro and np.isfinite(dentro) else np.nan}


# ------------------------------------------------------------------------------- θ* final y congelado
def modelo_final(desarrollo: pd.DataFrame, modo: str = MODO_OFICIAL, meses_train: int = MESES_TRAIN) -> dict:
    """θ* con el mismo procedimiento que la última ventana del walk-forward oficial, con embargo de una sesión
    antes del inicio de validation. anchored: todo train + test; rolling: los últimos `meses_train` meses."""
    variables = calcular_variables_regimen(desarrollo)
    b = desarrollo.index[-1 - EMBARGO]
    a = desarrollo.index[0] if modo == "anchored" else \
        desarrollo.index[desarrollo.index >= b - pd.DateOffset(months=meses_train)][0]
    t0 = time.perf_counter()
    opt = optimizar_todos(desarrollo, variables, a, b)
    return {**opt, "desde": a, "hasta": b, "segundos": time.perf_counter() - t0,
            "pruebas": sum(x["pruebas"] for x in opt["por_regimen"].values())}


def guardar_theta(ruta, final: dict, extra: dict | None = None) -> dict:
    """Congela θ* por régimen y el K-means (escala y centroides) en un JSON."""
    congelado = {
        "optimizado_desde": str(final["desde"]), "optimizado_hasta": str(final["hasta"]),
        "parametros": final["parametros"],
        "calmar_entrenamiento": {r: x["calmar"] for r, x in final["por_regimen"].items()},
        "calmar_meseta": {r: x["calmar_meseta"] for r, x in final["por_regimen"].items()},
        "regimen_kmeans": final["modelo"].a_dict(),
        "ajustes": {"semilla": SEMILLA, "estilo": ESTILO, "modo": MODO_OFICIAL, "operar": list(OPERAR), "fijos": FIJOS, "umbral_meseta": UMBRAL_MESETA, "n_aleatorias": N_ALEATORIAS, "n_tpe": N_TPE,
                    "min_operaciones": MIN_OPERACIONES, "meses_train": MESES_TRAIN, "embargo": EMBARGO},
        **(extra or {}),
    }
    ruta.write_text(json.dumps(congelado, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return congelado


def cargar_theta(ruta) -> dict:
    d = json.loads(ruta.read_text(encoding="utf-8"))
    d["modelo"] = KMedias.desde_dict(d["regimen_kmeans"])
    return d


# ---------------------------------------------------------------------------------------- robustez
def importancia(estudio) -> pd.Series:
    """Importancia de cada parámetro (fANOVA de Optuna)."""
    return pd.Series(optuna.importance.get_param_importances(estudio, params=list(RANGOS))).sort_values(ascending=False)


def superficie(final: dict, datos: pd.DataFrame, regimen: str, ejes: tuple, puntos: int = 10) -> dict:
    """Calmar en una malla de dos parámetros con el resto de θ_r* fijo, en la ventana de entrenamiento final."""
    base = final["parametros"][regimen]
    xs = np.linspace(*RANGOS[ejes[0]][:2], puntos)
    ys = np.linspace(*RANGOS[ejes[1]][:2], puntos)
    Z = np.full((puntos, puntos), np.nan)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            p = dict(base)
            for eje, v in ((ejes[0], x), (ejes[1], y)):
                p[eje] = int(round(v)) if RANGOS[eje][2] is int else float(v)
            r = correr(datos, final["etiquetas"], {regimen: p}, final["desde"], final["hasta"])
            Z[i, j] = calmar(r["equity"]) if len(r["operaciones"]) >= MIN_OPERACIONES else np.nan
    return {"x": xs, "y": ys, "z": Z, "ejes": ejes, "base": base}


def _variar(valor, clave, factor):
    nuevo = valor * factor
    if clave == "fraccion":
        nuevo = min(nuevo, 1.0)  # sin apalancamiento
    if RANGOS[clave][2] is int:
        nuevo = max(1, int(round(nuevo)))
    return nuevo


def sensibilidad(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict, desde, hasta=None,
                 variacion: float = 0.20) -> pd.DataFrame:
    """Varía cada parámetro de cada régimen activo ±20% (uno a la vez) y mide el Calmar."""
    base = calmar(correr(datos, etiquetas, parametros, desde, hasta)["equity"])
    filas = []
    for r, p in parametros.items():
        if p is None:
            continue
        for clave in RANGOS:
            fila = {"parametro": f"{r}·{clave}", "base": base}
            for etiqueta, factor in (("-20%", 1 - variacion), ("+20%", 1 + variacion)):
                variado = {**parametros, r: {**p, clave: _variar(p[clave], clave, factor)}}
                fila[etiqueta] = calmar(correr(datos, etiquetas, variado, desde, hasta)["equity"])
            filas.append(fila)
    return pd.DataFrame(filas).set_index("parametro")[["-20%", "base", "+20%"]] if filas else pd.DataFrame()


def barrido_costos(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict, desde, hasta=None,
                   ida_vuelta_bps=np.arange(0, 101, 10)) -> pd.DataFrame:
    """Retorno neto, Sharpe y Calmar contra el costo total de ida y vuelta (comisión repartida en los dos lados,
    sin spread ni impacto, para aislar el efecto del costo)."""
    filas = []
    for bps in ida_vuelta_bps:
        r = correr(datos, etiquetas, parametros, desde, hasta, costos=Costos(bps / 2 / 1e4, 0.0, 0.0))
        filas.append({"ida_vuelta_bps": int(bps), "retorno_neto": r["equity"].iloc[-1] / r["equity"].iloc[0] - 1,
                      "sharpe": sharpe(r["equity"]), "calmar": calmar(r["equity"]),
                      "operaciones": len(r["operaciones"])})
    return pd.DataFrame(filas)
