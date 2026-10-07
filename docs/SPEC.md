# Especificación de la estrategia (NVS en velas de 1 hora, estilo ruptura)

## Universo y datos

- Activo: Novartis AG (NYSE: NVS), ADR líquido de farmacéutica.
- Velas de 1 hora de la sesión regular (09:30 a 16:00 de Nueva York): 7 velas por sesión, la última de 30 minutos.
- Fuente: Yahoo Finance (yfinance), precios ajustados por dividendos y splits; 730 días (límite de Yahoo para 1 hora).
- Datos congelados en `data/nvs_1h.csv`. El hash SHA-256 del archivo se guarda junto con θ* y validation se niega a
  correr si el archivo cambió.
- Split cronológico por sesiones completas: 60% train, 20% test, 20% validation.

## Indicadores y votos (al cierre de la vela t)

| Voto | Indicador (familia) | +1 (fuerza alcista) | −1 (fuerza bajista) |
|---|---|---|---|
| v_bollinger | Bandas de Bollinger (20, bb_k) (volatilidad) | Close > banda superior | Close < banda inferior |
| v_rsi | RSI 14 (momento) | RSI > 100 − rsi_bajo | RSI < rsi_bajo |
| v_estocastico | Estocástico %K (14, 3) (momento) | %K > 80 | %K < 20 |

Cada evento sigue vigente 3 velas (memoria). Filtro de dirección: EMA(ema_n) (tendencia). ATR 14 para stop y objetivo.

El código conserva el estilo "rebote" (los mismos votos con el signo invertido y salida al tocar la media), que se
comparó en el walk-forward (ver README).

## Entrada

preparación_t = +1 si Σ 1[v_i,t = +1] ≥ 2; −1 si Σ 1[v_i,t = −1] ≥ 2; 0 en otro caso.

señal_t = +1 si preparación_t = +1 y Close_t > EMA_t (comprar una ruptura alcista dentro de una tendencia alcista);
−1 si preparación_t = −1 y Close_t < EMA_t (vender en corto una ruptura bajista dentro de una tendencia bajista);
0 en otro caso.

La orden se ejecuta en la apertura de t+1, solo si t+1 es de la misma sesión que t.

## Salida (lo que ocurra primero)

1. Stop-loss a sl_atr · ATR y take-profit a 12 ATR del precio de entrada, revisados dentro de cada vela. Si ambos caen
   en la misma vela, se ejecuta el stop. Si la vela abre más allá del nivel, se llena a la apertura.
2. Pérdida de impulso: un largo se cierra en la apertura siguiente cuando Close < media de Bollinger; un corto, cuando
   Close > media.
3. Señal opuesta: cierra en la apertura siguiente y abre la contraria si la entrada está permitida.
4. Holding máximo de 70 velas (10 sesiones).
5. Fin de los datos.

Las posiciones se mantienen de un día a otro.

## Régimen y transiciones

- K-means (k = 3) sobre ln(volatilidad realizada) y √(eficiencia de Kaufman) de la última semana (35 velas),
  reclasificado cada 4 velas (4 horas de sesión). Nombres por centroides: crisis = mayor volatilidad; de los otros
  dos, tendencia = mayor eficiencia.
- θ distinto por régimen: θ_tendencia, θ_reversion y θ_crisis, cada uno con los 5 parámetros optimizados de abajo.
- En cada vela, la señal de entrada usa el θ del régimen vigente. Si un régimen quedó apagado (su mejor Calmar o el de
  su meseta en entrenamiento no fue positivo), no se abren posiciones en él. En el θ* final, reversión quedó apagado.
- Regla de transición: una posición abierta no se cierra por cambio de régimen. Conserva el stop, el objetivo, el
  holding máximo y la regla de salida del θ del régimen en que entró.

## Parámetros optimizados (por régimen)

| Parámetro | Rango | Significado |
|---|---|---|
| bb_k | 1.5 a 3.0 | ancho de las bandas de Bollinger (qué tan fuerte debe ser la ruptura) |
| rsi_bajo | 30 a 50 | fuerza alcista si RSI > 100 − rsi_bajo (50 a 70); bajista si RSI < rsi_bajo |
| ema_n | 50 a 300 | EMA del filtro de dirección |
| sl_atr | 2 a 10 | stop-loss en ATR |
| fraccion | 0.2 a 1.0 | tamaño de posición (fracción del equity) |

Fijos (iguales en todos los regímenes): Bollinger de 20 velas, Estocástico 20/80, memoria de 3 velas, take-profit a
12 ATR y holding máximo de 70 velas.

## Tamaño y costos

- Tamaño: fraccion · equity, sin apalancamiento; una sola posición a la vez; largos y cortos.
- Comisión de 0.125% del monto en cada apertura y cada cierre.
- Medio spread de $0.005 por acción en cada lado.
- Impacto de raíz cuadrada: precio · σ_diaria · √(acciones / volumen diario promedio de las 20 sesiones previas).
- Capital inicial: $1,000,000.

## Optimización y walk-forward

- Objetivo: Calmar del backtest del régimen en la ventana de entrenamiento; menos de 6 operaciones = −10.
- 50 pruebas aleatorias y después 100 de TPE (Optuna) por régimen y ventana, semilla 42.
- θ* = centro de la mejor meseta (promedio de los 10 vecinos más cercanos), no el argmax.
- Walk-forward: 1 mes de prueba, paso mensual, embargo de 7 velas. Anchored (oficial: entrenamiento desde el inicio
  de los datos, al menos 6 meses) y rolling (comparación: últimos 6 meses). La curva fuera de muestra es una sola
  simulación encadenada.
- θ* final: misma optimización que la última ventana anchored (todo train + test), congelado en
  `docs/theta_congelado.json` (con el K-means) y versionado en git antes de evaluar validation.

## Métricas

Retornos por vela de 1 hora, anualización con 7 × 252 = 1,764 velas, tasa libre de riesgo 0. Calmar =
rendimiento anualizado compuesto / |máximo drawdown|. Buy & hold paga la comisión de entrada y de salida.

## Extensiones evaluadas solo en el walk-forward

`python main.py --extensiones` corre el walk-forward anchored de train + test con tres cambios (EXTENSIONES en
`src/optimize.py`), sin tocar θ* ni validation: trailing stop a sl_atr ATR del cierre en lugar de la salida por la
media; confirmación 3 de 3; fracción fija de 100% del equity. Ninguno mejoró el diseño congelado (ver README).
