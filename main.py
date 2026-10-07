"""Ejecuta el laboratorio completo: python main.py

NVS (Novartis AG, ADR) en velas de 1 hora, split cronológico 60/20/20 por sesiones: train, test y validation.
El walk-forward (anchored oficial, rolling como comparación) y θ* usan solo train + test. La primera corrida congela θ* en docs/theta_congelado.json y se
detiene; después de hacer commit de ese archivo, la siguiente corrida evalúa validation con θ* congelado
(sin reoptimizar nada) y genera todas las tablas y figuras. Validation se niega a correr si θ* no está en
un commit o si los datos cambiaron.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import Costos, buy_and_hold
from src.data import (RUTA_CRUDOS, TICKER, auditar_datos, cargar_datos, descargar, huella,
                      separar_train_test_validacion)
from src.metrics import (bootstrap_media, buy_hold_escalado, exposicion, metricas_por_regimen, prueba_diferencia_regimenes,
                         resumen_metricas, retorno_por_operacion, tablas_retornos)
from src.optimize import (EMBARGO, MESES_TRAIN, MIN_OPERACIONES, ESTILO, EXTENSIONES, MODO_OFICIAL, N_ALEATORIAS, N_TPE, OPERAR, RANGOS,
                          SEMILLA, barrido_costos,
                          cargar_theta, correr, guardar_theta, importancia, modelo_final, sensibilidad,
                          simular_encadenado, superficie, walk_forward)
from src.plots import (plot_anchored_rolling, plot_distribuciones_regimen, plot_historia, plot_importancia,
                       plot_indicadores, plot_linea_tiempo, plot_operaciones, plot_regimenes_precio,
                       plot_sensibilidad, plot_sensibilidad_costos, plot_slices, plot_superficie,
                       plot_tabla_retornos, plot_valor_drawdown, plot_valor_regimenes, plot_walk_forward)
from src.regimes import HMM, KMedias, Reglas, calcular_variables_regimen, comparar, validar_regimenes
from src.signals import MODOS, generar_senales

SEED = 42
FIGURES_DIR = Path("docs/figures")
RESULTS_DIR = Path("resultados")
THETA = Path("docs/theta_congelado.json")
COLUMNAS = ["retorno_total", "rendimiento_anualizado", "volatilidad", "sharpe", "sortino", "calmar",
            "max_drawdown", "win_rate", "operaciones", "turnover_anual"]


def imprimir(tabla):
    print("   " + tabla.to_string().replace("\n", "\n   "))


def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8")
    except FileNotFoundError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def metricas_bh(datos: pd.DataFrame) -> dict:
    valor, op = buy_and_hold(datos)
    m = resumen_metricas(valor, op)
    m["win_rate"] = np.nan
    return m


def tramo(valor: pd.Series, operaciones: pd.DataFrame, desde=None, hasta=None):
    """Parte de la curva encadenada (reescalada a $1,000,000) y las operaciones que entraron en ella."""
    mascara = np.ones(len(valor), dtype=bool)
    if desde is not None:
        mascara &= valor.index >= desde
    if hasta is not None:
        mascara &= valor.index < hasta
    v = valor[mascara]
    o = operaciones[(operaciones["entrada"] >= v.index[0]) & (operaciones["entrada"] <= v.index[-1])] \
        if len(operaciones) else operaciones
    return v / v.iloc[0] * 1e6, o


def comparar_con_bh(wf: dict, datos: pd.DataFrame, corte) -> pd.DataFrame:
    """Estrategia (walk-forward encadenado) contra buy & hold en el tramo de train, de test y completo."""
    filas = {}
    for nombre, a, b in (("train (WF)", None, corte), ("test (WF)", corte, None), ("WF completo", None, None)):
        v, o = tramo(wf["valor"], wf["operaciones"], a, b)
        filas[nombre] = resumen_metricas(v, o)
        filas[f"B&H {nombre}"] = metricas_bh(datos.loc[v.index[0]:v.index[-1]])
    return pd.DataFrame(filas).loc[COLUMNAS]


def main(hasta_test: bool = False, n_jobs: int = -1):
    np.random.seed(SEED)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 170)
    resultados = {}
    t_inicio = time.perf_counter()

    if THETA.exists():
        ajustes = json.loads(THETA.read_text(encoding="utf-8")).get("ajustes", {})
        if (ajustes.get("modo") != MODO_OFICIAL or ajustes.get("operar") != list(OPERAR) or ajustes.get("estilo") != ESTILO
                or ajustes.get("semilla") != SEMILLA):
            raise SystemExit(f"{THETA} se congeló con otra configuración de la estrategia. Quítalo del repositorio "
                             f"(git rm {THETA.as_posix()}), haz commit y vuelve a correr python main.py para "
                             "congelar el θ* de esta configuración.")

    print(f"1. Cargando datos de {TICKER} (velas de 1 hora, sesión regular)...")
    if not RUTA_CRUDOS.exists():
        print(f"   No existe {RUTA_CRUDOS}: descargando de Yahoo Finance (cambia el periodo según el día).")
        descargar()
    datos = cargar_datos()
    auditoria = auditar_datos(datos)
    train, test, validacion = separar_train_test_validacion(datos)
    desarrollo = pd.concat([train, test])
    print(f"   {auditoria['velas']} velas en {auditoria['dias']} sesiones; duplicados {auditoria['duplicados']}, "
          f"nulos {auditoria['nulos']}, OHLC incoherente {auditoria['ohlc_incoherente']}, "
          f"sesiones incompletas {auditoria['dias_incompletos']} (medias jornadas)")
    for nombre, d in (("Train", train), ("Test", test), ("Validation", validacion)):
        print(f"   {nombre}: {d.index[0]:%Y-%m-%d} a {d.index[-1]:%Y-%m-%d} ({len(d)} velas, {len(d) / len(datos):.0%})")
    resultados["auditoria"] = auditoria
    resultados["split"] = {n: [str(d.index[0]), str(d.index[-1]), len(d)]
                           for n, d in (("train", train), ("test", test), ("validation", validacion))}

    print("\n2. Regímenes: reglas, K-means y HMM filtrado, ajustados solo con train...")
    variables = calcular_variables_regimen(datos)
    f_train = variables.loc[train.index]
    clasificadores = {"reglas": Reglas().ajustar(f_train), "kmeans": KMedias().ajustar(f_train),
                      "hmm": HMM().ajustar(f_train)}
    etiq_clas = {n: m.etiquetar(variables.loc[desarrollo.index]) for n, m in clasificadores.items()}
    etiq_clas["hmm_viterbi"] = clasificadores["hmm"].etiquetar_viterbi(variables.loc[train.index])
    tabla_clas = pd.DataFrame({(n, c): comparar(variables.loc[d.index], e.reindex(d.index).fillna("sin_datos"))
                               for n, e in etiq_clas.items() for c, d in (("train", train), ("test", test))
                               if e.reindex(d.index).notna().any()}).T
    imprimir(tabla_clas.round(3))
    tabla_clas.to_csv(RESULTS_DIR / "clasificadores.csv")
    coincidencia = float((etiq_clas["hmm"].loc[train.index] == etiq_clas["hmm_viterbi"].loc[train.index]).mean())
    print(f"   Coincidencia HMM filtrado contra Viterbi en train: {coincidencia:.1%} · elegido: K-means")
    print("   Centroides de K-means (train):")
    imprimir(clasificadores["kmeans"].centros().round(3))
    resultados["clasificadores"] = {"/".join(k): v for k, v in tabla_clas.to_dict(orient="index").items()}
    resultados["coincidencia_filtrada_viterbi"] = coincidencia

    print(f"\n3. Señales ({ESTILO}) con parámetros base en train: efecto de la regla de confirmación...")
    conteo = pd.DataFrame({m: generar_senales(train, {"estilo": ESTILO}, modo=m)["senal"].value_counts().reindex([1, -1], fill_value=0)
                           for m in MODOS}).rename(index={1: "largos", -1: "cortos"})
    imprimir(conteo)
    resultados["senales_base"] = conteo.to_dict()

    otro = "rolling" if MODO_OFICIAL == "anchored" else "anchored"
    print(f"\n4. Walk-forward sobre train + test ({MODO_OFICIAL} oficial y {otro} como comparación; al menos "
          f"{MESES_TRAIN} meses de entrenamiento, 1 mes de prueba, paso mensual, embargo de {EMBARGO} velas; "
          f"{N_ALEATORIAS} aleatorias + {N_TPE} TPE por régimen y ventana; mínimo {MIN_OPERACIONES} operaciones; "
          f"regímenes que operan: {', '.join(OPERAR)})...")
    wf = walk_forward(desarrollo, MODO_OFICIAL, n_jobs=n_jobs)
    comparacion = walk_forward(desarrollo, otro, n_jobs=n_jobs)
    corte = test.index[0]
    tablas_wf = {}
    for w in (wf, comparacion):
        print(f"   {w['modo'].capitalize()}{' (oficial)' if w is wf else ' (comparación)'}: {len(w['resumen'])} ventanas, {w['configuraciones']} configuraciones en "
              f"{w['segundos']:.0f} s · anualizado dentro {w['anualizado_dentro']:.2%}, fuera {w['anualizado_fuera']:.2%}"
              f" · walk-forward efficiency {w['eficiencia']:.2f}")
        tablas_wf[w["modo"]] = comparar_con_bh(w, desarrollo, corte)
        imprimir(tablas_wf[w["modo"]].astype(float).round(3))
        tablas_wf[w["modo"]].to_csv(RESULTS_DIR / f"walk_forward_{w['modo']}_vs_bh.csv")
        w["resumen"].to_csv(RESULTS_DIR / f"walk_forward_{w['modo']}.csv", index=False)
        w["operaciones"].to_csv(RESULTS_DIR / f"walk_forward_{w['modo']}_operaciones.csv", index=False)
        activos = w["resumen"][[c for c in w["resumen"] if c.startswith("activo_")]].mean()
        print(f"   Fracción de ventanas con cada régimen activo: {activos.round(2).to_dict()}")
    resultados["walk_forward"] = {
        w["modo"]: {"ventanas": len(w["resumen"]), "configuraciones": w["configuraciones"], "segundos": w["segundos"],
                    "anualizado_dentro": w["anualizado_dentro"], "anualizado_fuera": w["anualizado_fuera"],
                    "eficiencia": w["eficiencia"], "tabla": tablas_wf[w["modo"]].to_dict()}
        for w in (wf, comparacion)}
    if hasta_test:
        print("\nValidation NO se evaluó (--hasta-test).")
        return

    print(f"\n5. θ*: modelo final con {'todo train + test' if MODO_OFICIAL == 'anchored' else f'los últimos {MESES_TRAIN} meses de train + test'}"
          " (mismo procedimiento que la última ventana del walk-forward oficial)...")
    final = modelo_final(desarrollo)
    for r, x in final["por_regimen"].items():
        estado = "apagado" if x["parametros"] is None else "activo"
        if x["estudio"] is None:
            print(f"   {r}: no se opera (regla del régimen)")
            continue
        print(f"   {r}: {x['pruebas']} pruebas, mejor Calmar {x['calmar']:.2f}, meseta {x['calmar_meseta']:.2f}, "
              f"θ* {'= argmax' if x['es_argmax'] else 'del centro de la meseta'} ({estado})")
    if not THETA.exists():
        import optuna, sklearn
        guardar_theta(THETA, final, {
            "activo": TICKER, "datos": RUTA_CRUDOS.as_posix(), "sha256_datos": huella(),
            "split": resultados["split"], "congelado_en": datetime.now().isoformat(timespec="seconds"),
            "versiones": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                          "optuna": optuna.__version__}})
        print(f"\n   θ* congelado en {THETA}. Validation NO se evaluó.")
        print(f"   Haz commit y push de θ* y de los datos, y vuelve a correr python main.py:")
        print(f"      git add {THETA.as_posix()} {RUTA_CRUDOS.as_posix()}")
        print('      git commit -m "Congela theta* antes de evaluar validation"')
        print("      git push")
        return

    congelado = cargar_theta(THETA)
    commit = _git("log", "-1", "--format=%h %cI", "--", THETA.as_posix())
    pendiente = _git("status", "--porcelain", "--", THETA.as_posix())
    if commit is None:
        raise SystemExit("Esta carpeta no es un repositorio git: no hay forma de probar que θ* se congeló antes de "
                         "ver validation. Corre el proyecto dentro del repositorio clonado.")
    if not commit or pendiente:
        raise SystemExit(f"{THETA} no está en un commit (o tiene cambios sin commit). Haz commit antes de validar.")
    if congelado.get("sha256_datos") != huella():
        raise SystemExit(f"{RUTA_CRUDOS} no es el mismo archivo con el que se congeló θ*. Usa el CSV del commit.")
    theta = congelado["parametros"]
    coincide = all((theta[r] is None) == (final["parametros"][r] is None) and
                   (theta[r] is None or all(np.isclose(theta[r][k], final["parametros"][r][k]) for k in RANGOS))
                   for r in theta)
    print(f"   θ* leído de {THETA} (commit {commit}); {'coincide' if coincide else 'NO coincide'} con el recalculado"
          f"{'' if coincide else ' (versiones de librerías distintas; validation usa siempre el congelado)'}.")
    resultados["theta_congelado"] = {k: v for k, v in congelado.items() if k != "modelo"}
    resultados["commit_theta"] = commit
    resultados["theta_coincide_recalculado"] = coincide

    print("\n6. Validation con θ* y K-means congelados (sin reoptimizar)...")
    etiquetas = congelado["modelo"].etiquetar(variables)
    inicio_val = validacion.index[0]
    val = correr(datos, etiquetas, theta, inicio_val)
    v_train, o_train = tramo(wf["valor"], wf["operaciones"], None, corte)
    v_test, o_test = tramo(wf["valor"], wf["operaciones"], corte, None)
    metricas = {
        "train (WF)": resumen_metricas(v_train, o_train), "B&H train": metricas_bh(train.loc[v_train.index[0]:]),
        "test (WF)": resumen_metricas(v_test, o_test), "B&H test": metricas_bh(test),
        "validation": resumen_metricas(val["equity"], val["operaciones"]), "B&H validation": metricas_bh(validacion),
    }
    tabla_metricas = pd.DataFrame(metricas).loc[COLUMNAS]
    imprimir(tabla_metricas.astype(float).round(4))
    tabla_metricas.to_csv(RESULTS_DIR / "metricas.csv")
    val["operaciones"].to_csv(RESULTS_DIR / "validation_operaciones.csv", index=False)
    print(f"   Comisiones en validation: ${val['comisiones']:,.0f} · spread e impacto: ${val['deslizamiento']:,.0f}")

    print("   Benchmark con la misma exposición media y bootstrap del retorno por operación:")
    filas_expo = {}
    for nombre, valor, posicion, cierre, ops in (
            ("walk-forward", wf["valor"], wf["posicion"], desarrollo["Close"], wf["operaciones"]),
            ("validation", val["equity"], val["posicion"], datos["Close"], val["operaciones"])):
        e = float(exposicion(posicion, cierre, valor).mean())
        bh_e = resumen_metricas(buy_hold_escalado(cierre.loc[valor.index[0]:valor.index[-1]], e), pd.DataFrame())
        est = resumen_metricas(valor, ops)
        boot = bootstrap_media(retorno_por_operacion(ops))
        filas_expo[nombre] = {"exposicion_media": e, "retorno_estrategia": est["retorno_total"],
                              "retorno_bh_misma_exposicion": bh_e["retorno_total"], "calmar_estrategia": est["calmar"],
                              "calmar_bh_misma_exposicion": bh_e["calmar"], "dd_bh_misma_exposicion": bh_e["max_drawdown"],
                              "media_por_operacion": boot["media"], "ic95_bajo": boot["ic_bajo"],
                              "ic95_alto": boot["ic_alto"], "p_valor_media_positiva": boot["p_valor"]}
    tabla_expo = pd.DataFrame(filas_expo)
    imprimir(tabla_expo.astype(float).round(4))
    tabla_expo.to_csv(RESULTS_DIR / "exposicion_bootstrap.csv")
    resultados["exposicion_bootstrap"] = filas_expo
    tablas = {"train (WF)": tablas_retornos(v_train), "test (WF)": tablas_retornos(v_test),
              "validation": tablas_retornos(val["equity"])}
    for nombre, t in tablas.items():
        pd.DataFrame({"retorno": t["mensual"]}).to_csv(RESULTS_DIR / f"retornos_mensuales_{nombre.split()[0]}.csv")
    regimenes_val = {n: validar_regimenes(congelado["modelo"], variables.loc[d.index], etiquetas.loc[d.index])
                     for n, d in (("train", train), ("test", test), ("validation", validacion))}
    print("   Validación del régimen (K-means congelado):")
    tabla_reg = pd.DataFrame(regimenes_val)
    imprimir(tabla_reg.drop(index="duracion_media_por_regimen").astype(float).round(3))
    print("   Duración media por régimen (horas de sesión):")
    imprimir(pd.DataFrame(tabla_reg.loc["duracion_media_por_regimen"].to_dict()).T)
    por_regimen = metricas_por_regimen(val["equity"], val["operaciones"], etiquetas)
    print("   Métricas por régimen en validation:")
    imprimir(por_regimen.astype(float).round(4))
    diferencia = prueba_diferencia_regimenes(pd.concat([wf["operaciones"], val["operaciones"]], ignore_index=True))
    print(f"   Kruskal-Wallis del retorno por operación entre regímenes (walk-forward + validation): "
          f"H = {diferencia['estadistico']:.2f}, p = {diferencia['p_valor']:.3f}")
    por_regimen.to_csv(RESULTS_DIR / "metricas_por_regimen.csv")
    resultados.update({"metricas": tabla_metricas.to_dict(), "regimenes": regimenes_val, "kruskal": diferencia,
                       "comisiones_validation": val["comisiones"], "deslizamiento_validation": val["deslizamiento"]})

    print("\n7. Robustez de θ* en validation: regla de confirmación, sensibilidad ±20% y costos...")
    reglas = {}
    for modo in MODOS:
        m = resumen_metricas(*(lambda r: (r["equity"], r["operaciones"]))(correr(datos, etiquetas, theta, inicio_val,
                                                                                    modo=modo)))
        reglas[modo] = {k: m[k] for k in ("operaciones", "calmar", "retorno_total", "win_rate", "turnover_anual")}
    tabla_reglas = pd.DataFrame(reglas).T
    imprimir(tabla_reglas.astype(float).round(4))
    tabla_sens = sensibilidad(datos, etiquetas, theta, inicio_val)
    if len(tabla_sens):
        imprimir(tabla_sens.round(3))
    costos_val = barrido_costos(datos, etiquetas, theta, inicio_val)
    costos_wf = []
    for bps in costos_val["ida_vuelta_bps"]:
        sim = simular_encadenado([x["velas_fuera"] for x in wf["ventanas"]], [x["salidas_fuera"] for x in wf["ventanas"]],
                                 Costos(bps / 2 / 1e4, 0.0, 0.0))
        e = sim["equity"]
        costos_wf.append({"ida_vuelta_bps": bps, "retorno_neto": e.iloc[-1] / e.iloc[0] - 1,
                          "sharpe": resumen_metricas(e, sim["operaciones"])["sharpe"],
                          "calmar": resumen_metricas(e, sim["operaciones"])["calmar"], "operaciones": len(sim["operaciones"])})
    costos_wf = pd.DataFrame(costos_wf)
    for nombre, c in (("validation", costos_val), ("walk-forward", costos_wf)):
        positivos = c[c["retorno_neto"] > 0]
        maximo = f"{positivos['ida_vuelta_bps'].max()} pb" if len(positivos) else "ninguno"
        print(f"   {nombre}: retorno neto sin costos {c['retorno_neto'].iloc[0]:.2%}; costo de ida y vuelta máximo con "
              f"retorno positivo: {maximo} (rejilla de 10 pb; el del lab es 25 pb de comisión más spread e impacto)")
    tabla_reglas.to_csv(RESULTS_DIR / "reglas.csv")
    tabla_sens.to_csv(RESULTS_DIR / "sensibilidad.csv")
    costos_val.to_csv(RESULTS_DIR / "costos_validation.csv", index=False)
    costos_wf.to_csv(RESULTS_DIR / "costos_walk_forward.csv", index=False)
    resultados["reglas"] = reglas

    print("\n8. Generando figuras...")
    activos = {r: x["estudio"] for r, x in final["por_regimen"].items() if x["estudio"] is not None}
    importancias = {r: importancia(e) for r, e in activos.items()}
    ejemplo_r = next((r for r in ("reversion", "tendencia", "crisis") if theta.get(r)), None)
    figures = {
        "03_valor_drawdown.png": plot_valor_drawdown(
            {"train (WF)": (v_train, buy_and_hold(train.loc[v_train.index[0]:])[0]),
             "test (WF)": (v_test, buy_and_hold(test)[0]),
             "validation": (val["equity"], buy_and_hold(validacion)[0])},
            titulo=f"{TICKER}: valor del portafolio y drawdown, estrategia contra buy & hold"),
        "04_tabla_retornos.png": plot_tabla_retornos(tablas),
        "06_costos.png": plot_sensibilidad_costos({"validation": costos_val, "walk-forward (train + test)": costos_wf},
                                                  marca_bps=25.0),
        "07_regimenes_precio.png": plot_regimenes_precio(datos["Close"], etiquetas, inicio_val,
                                                         titulo=f"Regímenes de K-means (congelado) sobre el precio de {TICKER}"),
        "08_distribuciones_regimen.png": plot_distribuciones_regimen(variables.loc[desarrollo.index],
                                                                     etiquetas.loc[desarrollo.index]),
        "09_valor_regimenes.png": plot_valor_regimenes(val["equity"], etiquetas, buy_and_hold(validacion)[0],
                                                       titulo="Valor del portafolio en validation con regímenes superpuestos"),
        "10_walk_forward.png": plot_walk_forward(wf["resumen"]),
        "11_historia_optimizacion.png": plot_historia(activos, N_ALEATORIAS),
        "12_importancia_parametros.png": plot_importancia(importancias),
        "15_anchored_rolling.png": plot_anchored_rolling(
            {f"{w['modo'].capitalize()}{' (oficial)' if w is wf else ''}": (w["valor"], w["eficiencia"]) for w in (wf, comparacion)},
            buy_and_hold(desarrollo.loc[wf["valor"].index[0]:])[0]),
        "16_hmm_filtrada_viterbi.png": plot_linea_tiempo(train["Close"], etiq_clas["hmm"].loc[train.index],
                                                         etiq_clas["hmm_viterbi"]),
    }
    if len(tabla_sens):
        figures["05_sensibilidad.png"] = plot_sensibilidad(tabla_sens, "Sensibilidad del Calmar ante ±20% en θ* (validation)")
    if ejemplo_r:
        p = theta[ejemplo_r]
        ventana = datos.loc[inicio_val:].iloc[:7 * 20]
        senales = generar_senales(datos.loc[:ventana.index[-1]], {k: p[k] for k in p if k in
                                                                  ("estilo", "bb_n", "bb_k", "rsi_bajo", "sto_bajo", "memoria", "ema_n")})
        figures["01_indicadores.png"] = plot_indicadores(senales.loc[ventana.index], p,
                                                         titulo=f"{TICKER}: indicadores con θ* del régimen {ejemplo_r}")
        figures["02_operaciones.png"] = plot_operaciones(senales.loc[ventana.index], val["operaciones"],
                                                         titulo=f"{TICKER}: operaciones de las primeras 4 semanas de validation")
    for r, e in activos.items():
        elegido = final["parametros"][r] or e.best_params
        figures[f"13_slices_{r}.png"] = plot_slices(e, elegido, r)
        if final["parametros"][r] is not None:
            ejes = tuple(importancias[r].index[:2])
            figures[f"14_superficie_{r}.png"] = plot_superficie(superficie(final, desarrollo, r, ejes), r)
    for filename, fig in figures.items():
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    resultados["segundos_totales"] = time.perf_counter() - t_inicio
    (RESULTS_DIR / "resumen.json").write_text(json.dumps(resultados, indent=2, default=str, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"   Figuras en {FIGURES_DIR}; tablas en {RESULTS_DIR}")
    print(f"\nProyecto ejecutado correctamente en {resultados['segundos_totales'] / 60:.1f} minutos.")


def extensiones(nombres: list | None = None, n_jobs: int = -1):
    """Walk-forward oficial (solo train + test) de las extensiones de EXTENSIONES. No toca θ* ni validation:
    cada fila se guarda en resultados/extensiones.csv en cuanto termina."""
    RESULTS_DIR.mkdir(exist_ok=True)
    datos = cargar_datos()
    train, test, _ = separar_train_test_validacion(datos)
    desarrollo = pd.concat([train, test])
    ruta = RESULTS_DIR / "extensiones.csv"
    tabla = pd.read_csv(ruta, index_col=0) if ruta.exists() else pd.DataFrame()
    columnas = ["descripcion", "retorno_total", "calmar", "max_drawdown", "sharpe", "win_rate", "operaciones",
                "turnover_anual", "exposicion_media", "eficiencia"]
    pendientes = {"oficial": None, **{n: EXTENSIONES[n] for n in (nombres or EXTENSIONES)}}
    print(f"Extensiones sobre el walk-forward {MODO_OFICIAL} de train + test (validation no se toca)...")
    for nombre, ext in pendientes.items():
        if nombre == "oficial" and "oficial" in tabla.index:
            continue
        w = walk_forward(desarrollo, MODO_OFICIAL, n_jobs=n_jobs, ext=ext)
        m = resumen_metricas(w["valor"], w["operaciones"])
        fila = {"descripcion": "diseño congelado" if ext is None else ext["descripcion"],
                **{k: m[k] for k in columnas[1:-2]},
                "exposicion_media": float(exposicion(w["posicion"], desarrollo["Close"], w["valor"]).mean()),
                "eficiencia": w["eficiencia"]}
        tabla = pd.concat([tabla.drop(index=nombre, errors="ignore"), pd.DataFrame({nombre: fila}).T])
        tabla.to_csv(ruta)
        print(f"   {nombre}: Calmar {m['calmar']:.2f}, retorno {m['retorno_total']:.2%}, drawdown {m['max_drawdown']:.2%}, "
              f"win rate {m['win_rate']:.1%}, {m['operaciones']} operaciones ({w['segundos']:.0f} s)")
    print(f"   Tabla en {ruta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laboratorio 02: estrategia de trading con análisis técnico (NVS 1h)")
    parser.add_argument("--hasta-test", action="store_true", help="solo walk-forward sobre train + test; no toca validation")
    parser.add_argument("--n-jobs", type=int, default=-1, help="procesos en paralelo para el walk-forward")
    parser.add_argument("--extensiones", nargs="*", metavar="NOMBRE",
                        help=f"walk-forward de las extensiones (todas o las nombradas: {', '.join(EXTENSIONES)}); no toca validation")
    args = parser.parse_args()
    if args.extensiones is not None:
        extensiones(args.extensiones, args.n_jobs)
    else:
        main(args.hasta_test, args.n_jobs)
