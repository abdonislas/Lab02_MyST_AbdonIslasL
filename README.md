# Laboratorio 02: Estrategia de trading con análisis técnico

- Abdon Islas

**Nivel de alcance: B** (estrategia multi-indicador, backtesting, optimización, walk-forward y detección dinámica de régimen de mercado).

## Descripción

Este proyecto desarrolla una estrategia sistemática de "ruptura" (seguimiento de tendencia) sobre Novartis AG (NVS) en velas de 1 hora de la sesión regular (noviembre de 2023 a octubre de 2026): compra cuando el precio rompe hacia arriba con fuerza dentro de una tendencia alcista, vende en corto cuando rompe hacia abajo dentro de una tendencia bajista, y cierra cuando el precio cruza de regreso su media. Tres votos (Bandas de Bollinger, RSI y Estocástico) deben coincidir al menos 2 de 3 y el precio debe estar del lado correcto de una EMA lenta. La estrategia se evalúa en un motor de backtesting event-driven con comisión de 0.125% por lado, spread, impacto de mercado, stop-loss y take-profit. Un K-means detecta tres regímenes (tendencia, reversión y crisis) y los parámetros de cada régimen se optimizan por separado con Optuna (random search y después TPE) maximizando el Calmar Ratio en un walk-forward anchored con 1 mes de prueba.

## Resultado principal

| Validation (10 de marzo a 6 de octubre de 2026) | Estrategia | Buy & hold |
|---|---:|---:|
| Retorno total | −5.9% | −13.1% |
| Máximo drawdown | −6.8% | −15.9% |
| Calmar | −1.47 | −1.36 |
| Volatilidad anualizada | 3.2% | 29.7% |

En validation la estrategia perdió menos de la mitad que buy & hold, con menos de la mitad de su drawdown; en Calmar quedan prácticamente empatados (con retornos negativos el Calmar penaliza un drawdown pequeño). La diferencia se debe a la baja exposición (7.8% del capital en promedio): contra un buy & hold con la misma exposición media, la estrategia queda abajo (−5.9% contra −0.9%). En el walk-forward sobre train + test no le ganó a buy & hold (Calmar 0.50 contra 1.63). La discusión completa está en `docs/reporte.pdf`.

## Elección del diseño (sesgo de selección)

El diseño se eligió solo con el walk-forward de train + test, entre cuatro variantes fijadas de antemano; θ* se congeló en un commit y validation se evaluó una sola vez. Elegir la mejor de cuatro infla el Calmar del walk-forward de la elegida, por eso se declara.

| Variante | Calmar walk-forward anchored |
|---|---:|
| Rebote, tres regímenes | −0.55 |
| Rebote, crisis sin operar | −0.52 |
| Ruptura, crisis sin operar | −0.14 |
| **Ruptura, tres regímenes (elegida)** | **0.50** |

Nota: el PDF asigna BTCUSDT de 5 minutos al Nivel B; el cambio de activo y de frecuencia se declara aquí y en el reporte. El walk-forward usa 1 mes de prueba con paso mensual y al menos 6 meses de entrenamiento, como el que el PDF especifica para acciones.

## Estructura del proyecto

```text
Lab02_MyST_Equipo6/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
│   └── nvs_1h.csv                 # datos crudos congelados
├── src/
│   ├── data.py                    # carga, validación y auditoría de datos
│   ├── signals.py                 # indicadores y regla de confirmación
│   ├── backtest.py                # motor event-driven con costos
│   ├── metrics.py                 # Sharpe, Sortino, Calmar, MDD, Win Rate, turnover
│   ├── optimize.py                # optimización por régimen y walk-forward
│   ├── regimes.py                 # reglas, K-means y HMM filtrado
│   └── plots.py
├── tests/
│   ├── conftest.py
│   ├── test_signals.py
│   ├── test_backtest.py
│   ├── test_metrics.py
│   ├── test_optimize.py
│   └── test_regimes.py
├── notebooks/
│   └── analysis.ipynb             # solo análisis y figuras, sin lógica
└── docs/
    ├── SPEC.md                    # especificación completa de la estrategia
    ├── theta_congelado.json       # θ* por régimen y K-means congelados
    ├── figures/
    ├── reporte.pdf
    └── presentacion.pdf
```

## Instalación

Requiere Python 3.10 o superior.

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproducción de resultados

Todo el proyecto (datos, regímenes, walk-forward anchored y rolling, θ*, validation, robustez y figuras) corre con un solo comando:

```bash
python main.py
```

Si `docs/theta_congelado.json` no existe, `main.py` corre el walk-forward, calcula θ* solo con train + test, lo guarda y se detiene sin tocar validation. Ese archivo se versiona en git (commit y push) y la siguiente corrida de `python main.py` evalúa validation con θ* congelado, sin reoptimizar nada. `main.py` se niega a evaluar validation si θ* no está en un commit, si tiene cambios sin commit, si se congeló con otra configuración de la estrategia o si `data/nvs_1h.csv` no es el mismo archivo con el que se congeló (se compara su hash SHA-256).

En el repositorio entregado θ* ya está congelado, así que `python main.py` corre todo de principio a fin. Tarda entre 5 y 15 minutos según el número de núcleos (el walk-forward corre en paralelo). Las figuras se guardan en `docs/figures/` y las tablas en `resultados/`. Después se puede abrir `notebooks/analysis.ipynb`.

Para correr solo el walk-forward sobre train + test, sin tocar validation:

```bash
python main.py --hasta-test
```

Pruebas automáticas:

```bash
python -m pytest -v
```

La semilla aleatoria es `42`, definida en `main.py`, `src/optimize.py` y `src/regimes.py`. Cada estudio de Optuna usa `TPESampler(n_startup_trials=50, seed=42 + desplazamiento)` por ventana y régimen, y K-means usa `random_state=42`.

## Datos

| Concepto | Valor |
|---|---|
| Activo | NVS (Novartis AG, ADR en NYSE) |
| Fuente | Yahoo Finance con `yfinance`, precios ajustados |
| Frecuencia | 1 hora, sesión regular 09:30 a 16:00 ET (7 velas por día; la última de 30 minutos) |
| Train | 7 de noviembre de 2023 a 7 de agosto de 2025 (3,046 velas, 60%) |
| Test | 8 de agosto de 2025 a 9 de marzo de 2026 (1,004 velas, 20%) |
| Validation | 10 de marzo a 6 de octubre de 2026 (1,022 velas, 20%) |
| Auditoría | 0 duplicados, 0 nulos, 0 velas con OHLC incoherente; 9 sesiones incompletas (medias jornadas de días festivos) |

Yahoo Finance solo conserva 730 días de velas de 1 hora y recorre esa ventana cada día, por eso los datos están congelados en `data/nvs_1h.csv` y nunca se vuelven a descargar si el archivo existe.

## Estrategia

| Voto | Indicador (familia) | Compra (+1) | Venta (−1) |
|---|---|---|---|
| v_bollinger | Bollinger (20, bb_k) (volatilidad) | Close > banda superior | Close < banda inferior |
| v_rsi | RSI 14 (momento) | RSI > 100 − rsi_bajo | RSI < rsi_bajo |
| v_estocastico | Estocástico %K (14, 3) (momento) | %K > 80 | %K < 20 |

Cada evento sigue vigente 3 velas. Regla de confirmación y filtro de dirección (EMA, familia de tendencia):

señal_t = +1 si Σ 1[v_i,t = +1] ≥ 2 y Close_t > EMA_t; −1 si Σ 1[v_i,t = −1] ≥ 2 y Close_t < EMA_t; 0 en otro caso.

Salidas: stop-loss en múltiplos de ATR, take-profit a 12 ATR, cruce de regreso de la media de Bollinger (un largo cierra si Close < media; un corto si Close > media), señal opuesta y holding máximo de 70 velas. La señal se calcula con información hasta el cierre de la vela t y la orden se ejecuta en la apertura de t+1, solo dentro de la misma sesión. Las posiciones se mantienen de un día a otro. Detalle completo en `docs/SPEC.md`.

## Parámetros fijos del problema

| Parámetro | Valor |
|---|---|
| Capital inicial | $1,000,000 |
| Comisión | 0.125% por lado, en cada apertura y cada cierre |
| Spread | Medio spread de $0.005 por acción en cada entrada y salida |
| Impacto de mercado | Ley de raíz cuadrada: precio · σ_diaria · √(acciones / volumen diario promedio de 20 sesiones) |
| Tamaño | `fraccion` del equity (optimizado por régimen), sin apalancamiento |
| Posiciones | Largas y cortas, una a la vez |
| Empate intrabar | Si stop y take-profit caen en la misma vela, se ejecuta primero el stop |
| Anualización | 7 velas × 252 sesiones = 1,764 velas por año, tasa libre de riesgo 0 |

## Detección de régimen

Variables sobre una ventana móvil de 1 semana (35 velas): ln(volatilidad realizada) y √(eficiencia de Kaufman). La clasificación se actualiza cada 4 horas de sesión. Se compararon tres clasificadores ajustados solo con train; el HMM etiqueta con probabilidades filtradas (algoritmo forward), y la ruta de Viterbi solo se calcula para compararla.

| Clasificador | Silhouette train / test | Duración media (h) train / test | Transiciones por mes train / test |
|---|---|---|---|
| Reglas (percentiles de train) | 0.331 / 0.267 | 20.1 / 20.1 | 7.3 / 7.2 |
| K-means | 0.358 / 0.225 | 16.2 / 17.0 | 9.0 / 8.5 |
| HMM filtrado | 0.261 / 0.189 | 24.5 / 31.4 | 6.0 / 4.5 |

El HMM filtrado coincide con Viterbi en el 93.8% de train. Se eligió K-means por la mayor separación en train (silhouette 0.358, por debajo del objetivo de 0.4). Nombres por centroides: crisis = mayor volatilidad; de los otros dos, tendencia = mayor eficiencia.

Regla de transición: un cambio de régimen no cierra la posición abierta; la posición conserva stop, objetivo, holding máximo y regla de salida del θ del régimen en que entró. Las entradas nuevas usan el θ del régimen vigente; un régimen apagado no abre posiciones.

## Optimización

| Elemento | Valor |
|---|---|
| Objetivo | θ_r* = argmax Calmar(backtest del régimen r), con comisión, spread e impacto |
| Parámetros por régimen | bb_k, rsi_bajo, ema_n, sl_atr y fraccion (`RANGOS` en `src/optimize.py`) |
| Fijos | Bollinger de 20 velas, Estocástico 20/80, memoria 3, take-profit 12 ATR, holding máximo 70 velas (`FIJOS`) |
| Búsqueda | 50 pruebas aleatorias y después 100 de TPE por régimen y ventana (150) |
| Restricción | Menos de 6 operaciones en la ventana → Calmar = −10 |
| Selección de θ* | Centro de la mejor meseta (vecindario de 10 trials), no el argmax |
| Régimen apagado | Si el mejor Calmar o el Calmar de la meseta del régimen en entrenamiento no es positivo (en θ* final: reversión) |

## Walk-forward

Anchored (oficial): el entrenamiento va desde el inicio de los datos hasta un día antes del mes de prueba (al menos 6 meses). Rolling (comparación): los últimos 6 meses. 1 mes de prueba, paso mensual, embargo de 1 sesión, sobre train + test. La curva fuera de muestra es una sola simulación encadenada: una posición abierta pasa al mes siguiente. θ* final se calcula con el mismo procedimiento que la última ventana anchored: todo train + test.

Walk-forward anchored completo (mayo de 2024 a marzo de 2026, todo fuera de muestra) contra buy & hold:

| Métrica | Estrategia (anchored) | Buy & hold |
|---|---:|---:|
| Retorno total | +5.6% | +67.3% |
| Rendimiento anualizado | +3.0% | +32.6% |
| Volatilidad anualizada | 5.8% | 20.1% |
| Sharpe | 0.54 | 1.50 |
| Sortino | 0.77 | 2.14 |
| Calmar | 0.50 | 1.63 |
| Máximo drawdown | −6.1% | −20.0% |
| Win rate | 34.6% |  |
| Operaciones | 78 | 1 |
| Turnover anual (× equity) | 31.5 | 1.2 |

Rendimiento anualizado dentro de muestra 1.2% y fuera 3.0% (walk-forward efficiency 2.45). Rolling: Calmar −0.27, WFE -0.39.

## Extensiones evaluadas solo en el walk-forward

Después de congelar θ* y evaluar validation, probamos tres cambios atacando los problemas observados (salidas prematuras, rupturas débiles y baja exposición). Se evalúan solo en el walk-forward anchored de train + test y no cambian θ* ni validation:

```bash
python main.py --extensiones
```

| Walk-forward anchored | Calmar | Retorno | Máx. drawdown | Win rate | Operaciones | Exposición media |
|---|---:|---:|---:|---:|---:|---:|
| **Diseño congelado** | 0.50 | +5.6% | −6.1% | 34.6% | 78 | 18.1% |
| Trailing stop en ATR (sin salida en la media) | 0.03 | +0.9% | −16.6% | 42.9% | 77 | 37.8% |
| Confirmación 3 de 3 | −0.36 | −5.9% | −9.2% | 34.6% | 52 | 11.8% |
| Fracción fija de 100% | −0.21 | −8.0% | −21.0% | 33.3% | 66 | 36.5% |

Ninguna mejora el diseño congelado. El trailing stop sube el win rate pero deja correr las pérdidas (el drawdown casi se triplica); la confirmación 3 de 3 deja pasar menos rupturas sin que las restantes sean mejores; la fracción fija de 100% amplifica las pérdidas.

## Resultados

Train (WF) y test (WF) son los tramos del walk-forward anchored fuera de muestra; validation se evalúa con θ* y K-means congelados (commit de `docs/theta_congelado.json`), sin reoptimizar nada.

| Métrica | Train (WF) | B&H train | Test (WF) | B&H test | Validation | B&H validation |
|---|---:|---:|---:|---:|---:|---:|
| Retorno total | +0.5% | +19.1% | +5.2% | +39.9% | −5.9% | −13.1% |
| Rendimiento anualizado | +0.4% | +15.0% | +9.3% | +80.4% | −10.0% | −21.5% |
| Volatilidad anualizada | 5.7% | 19.7% | 6.0% | 21.0% | 3.2% | 29.7% |
| Sharpe | 0.10 | 0.81 | 1.52 | 2.92 | −3.23 | −0.66 |
| Sortino | 0.14 | 1.13 | 2.43 | 4.24 | −3.69 | −0.85 |
| Calmar | 0.07 | 0.75 | 2.33 | 9.48 | −1.47 | −1.36 |
| Máximo drawdown | −6.1% | −20.0% | −4.0% | −8.5% | −6.8% | −15.9% |
| Win rate | 31.5% |  | 41.7% |  | 19.2% |  |
| Operaciones | 54 | 1 | 24 | 1 | 26 | 1 |
| Turnover anual (× equity) | 31.5 | 1.5 | 31.6 | 3.6 | 18.6 | 3.5 |

Benchmark con la misma exposición media y bootstrap del retorno por operación (10,000 remuestreos, semilla 42):

| | Walk-forward | Validation |
|---|---:|---:|
| Exposición media de la estrategia | 18.1% | 7.8% |
| Retorno de la estrategia | +5.6% | −5.9% |
| Buy & hold con la misma exposición | +10.4% | −0.9% |
| Retorno medio por operación | +0.2% | −1.1% |
| Intervalo de 95% (bootstrap) | [−0.3%, +0.7%] | [−1.7%, −0.5%] |
| p-valor de H0: media ≤ 0 | 0.233 | 0.999 |

En el walk-forward el retorno medio por operación no es distinguible de cero; en validation es significativamente negativo. Con nuestros números no hay evidencia de una ventaja real.

El análisis completo, con las respuestas a las preguntas del laboratorio, está en `docs/reporte.pdf`; la presentación en `docs/presentacion.pdf`. Las tablas quedan en `resultados/` y las figuras en `docs/figures/` al correr `python main.py`; `notebooks/analysis.ipynb` las muestra.

## Uso de inteligencia artificial

Se utilizó asistencia de IA (Claude) para: estructurar el proyecto según lo especificado, implementar el motor de backtesting con costos, la optimización por régimen con Optuna (random search y TPE), el walk-forward, la validación con θ* congelado y la detección de régimen, escribir las pruebas y las figuras, y redactar este README, el reporte y la presentación. La elección de indicadores, del activo y de la frecuencia fue del equipo. Todo el código y los resultados numéricos fueron revisados y ejecutados por los integrantes, quienes son responsables de explicar cualquier parte del proyecto entregado.
