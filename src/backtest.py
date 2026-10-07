"""Motor de backtesting orientado a eventos (no vectorizado), con estado explícito.

`backtest()` es una función pura: recibe datos, señales y parámetros y regresa todo el estado en el tiempo
(efectivo, posición, equity) más la lista de operaciones. El estado vive en tres variables explícitas:
`efectivo`, `posicion` y `equity = efectivo + acciones · Close`.

Convenciones (docs/SPEC.md):
- La señal se decide al cierre de t y se ejecuta en la apertura de t+1.
- Las entradas solo ocurren dentro de la misma sesión en que se decidió la señal (una señal de las 15:30 no
  abre posición en la apertura del día siguiente). Las salidas sí cruzan la noche: las posiciones se mantienen
  varios días y pueden cerrar en la apertura siguiente.
- Stop-loss y take-profit a sl_atr y tp_atr ATR del precio de entrada. Si ambos caen dentro de la misma vela,
  gana el stop (convención conservadora). Si la vela abre más allá de un nivel (gap), se llena a la apertura.
- Salida por señal: la señal opuesta o la señal de salida (cruce de la media de Bollinger) del θ con el que se abrió la
  posición cierran en la apertura siguiente; tras una señal opuesta se abre la contraria si la entrada está
  permitida.
- Holding máximo de `max_velas` velas; una posición abierta en la última vela de los datos se cierra ahí.
- Trailing stop (opcional, `trailing` = 1; extensión evaluada solo en el walk-forward): al cierre de cada vela el
  stop se acerca a sl_atr ATR (ATR de la entrada) del cierre, nunca se aleja, y la señal de salida por la media se
  ignora; la posición cierra por stop, objetivo, señal opuesta u holding máximo.
- Tamaño: `fraccion` del equity, sin apalancamiento (nunca más que el efectivo disponible).
- Costos en cada apertura y cada cierre: comisión de 0.125% del monto, medio spread de $0.005 por acción e
  impacto de raíz cuadrada (precio · σ_diaria · √(acciones / volumen diario promedio de 20 sesiones previas)).
- Los parámetros de salida (stop, objetivo, holding máximo y regla de salida) se fijan al abrir la posición.
  Un cambio de régimen no cierra la posición: sigue con el θ del régimen en que entró.
- Nunca hay más de una posición abierta.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

THETA_OPERACION = {"sl_atr": 4.0, "tp_atr": 8.0, "max_velas": 35, "fraccion": 1.0, "trailing": 0.0}
COLUMNAS_OPERACION = tuple(THETA_OPERACION)
CAPITAL = 1_000_000.0
VELAS_POR_DIA = 7


@dataclass(frozen=True)
class Costos:
    """Comisión por lado (fracción del monto), medio spread (USD por acción) e impacto de raíz cuadrada."""
    comision: float = 0.00125
    medio_spread: float = 0.005
    coef_impacto: float = 1.0


SIN_COSTOS = Costos(0.0, 0.0, 0.0)


@dataclass
class Posicion:
    """Una posición abierta: lado (+1 largo, −1 corto), acciones, precio de entrada, stop y objetivo."""
    lado: int
    acciones: float
    precio_entrada: float
    stop_loss: float
    take_profit: float
    entrada: pd.Timestamp = None
    velas: int = 0
    max_velas: float = np.inf
    regimen: str | None = None
    trailing: bool = False
    distancia: float = 0.0

    def __post_init__(self):
        if self.lado not in (1, -1):
            raise ValueError("lado debe ser +1 (largo) o −1 (corto)")
        if self.acciones <= 0:
            raise ValueError("acciones debe ser positivo")
        if self.lado * (self.precio_entrada - self.stop_loss) <= 0:
            raise ValueError("el stop debe quedar del lado de la pérdida")
        if self.lado * (self.take_profit - self.precio_entrada) <= 0:
            raise ValueError("el objetivo debe quedar del lado de la ganancia")

    def salida_intrabar(self, alto: float, bajo: float, apertura: float):
        """Precio y motivo si la vela toca stop u objetivo; None si no. Stop primero si toca ambos."""
        if self.lado == 1:
            if bajo <= self.stop_loss:
                return min(apertura, self.stop_loss), "stop_loss"
            if alto >= self.take_profit:
                return max(apertura, self.take_profit), "take_profit"
        else:
            if alto >= self.stop_loss:
                return max(apertura, self.stop_loss), "stop_loss"
            if bajo <= self.take_profit:
                return min(apertura, self.take_profit), "take_profit"
        return None


def niveles(lado: int, precio: float, atr: float, sl_atr: float, tp_atr: float) -> tuple[float, float]:
    """Stop y objetivo a sl_atr y tp_atr ATR del precio de entrada."""
    return precio - lado * sl_atr * atr, precio + lado * tp_atr * atr


def tamano_por_fraccion(equity: float, precio: float, fraccion: float, comision: float = 0.0) -> float:
    """Acciones para invertir `fraccion` del equity (0 < fraccion <= 1), incluida la comisión de entrada."""
    if not 0 < fraccion <= 1:
        raise ValueError("fraccion debe estar en (0, 1]")
    return fraccion * equity / (precio * (1 + comision))


def _deslizamiento(costos: Costos, precio: float, acciones: float, sigma: float, adv: float) -> float:
    """USD por acción: medio spread + precio · coef · σ_diaria · √(acciones / volumen diario promedio)."""
    impacto = 0.0
    if costos.coef_impacto > 0 and adv > 0 and np.isfinite(sigma):
        impacto = precio * costos.coef_impacto * sigma * np.sqrt(acciones / adv)
    return costos.medio_spread + impacto


def _volumen_diario_previo(datos: pd.DataFrame) -> np.ndarray:
    """Volumen diario promedio de las 20 sesiones anteriores (sin la sesión en curso)."""
    if "Volume" not in datos:
        return np.zeros(len(datos))
    dia = pd.Series(datos.index.date, index=datos.index)
    por_dia = datos["Volume"].groupby(dia.to_numpy()).sum()
    previo = por_dia.rolling(20, min_periods=1).mean().shift(1)
    return dia.map(previo).fillna(0.0).to_numpy()


def backtest(datos: pd.DataFrame, senal: pd.Series, theta: dict | None = None, costos: Costos = Costos(),
             capital: float = CAPITAL, por_vela: pd.DataFrame | None = None,
             salida: pd.Series | pd.DataFrame | None = None, regimen: pd.Series | None = None) -> dict:
    """Simula la estrategia vela por vela.

    datos: Open, High, Low, Close, atr (Volume opcional, para el impacto).
    senal: {−1, 0, +1} decidida al cierre de cada vela.
    theta / por_vela: sl_atr, tp_atr, max_velas y fraccion, fijos o por vela (θ del régimen vigente).
    salida: {−1, 0, +1, 2} decidida al cierre: −1 cierra largos, +1 cierra cortos, 2 cierra cualquiera.
        Si es un DataFrame con una columna por régimen, cada posición usa la columna del régimen en que entró.
    regimen: etiqueta de régimen al cierre de cada vela (solo para registrar el régimen de entrada).
    """
    p = {**THETA_OPERACION, **(theta or {})}
    n = len(datos)
    o, h, l, c = (datos[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    atr = datos["atr"].to_numpy(float)
    s = senal.reindex(datos.index).fillna(0).to_numpy(int)
    sigma = atr / c * np.sqrt(VELAS_POR_DIA)
    adv = _volumen_diario_previo(datos)
    dia = np.asarray(datos.index.date)
    param = {k: (por_vela[k].to_numpy(float) if por_vela is not None and k in por_vela else np.full(n, float(p[k])))
             for k in COLUMNAS_OPERACION}
    reg = regimen.reindex(datos.index).to_numpy() if regimen is not None else np.full(n, None)
    if isinstance(salida, pd.DataFrame):
        sal = {k: salida[k].reindex(datos.index).fillna(0).to_numpy(int) for k in salida}
    else:
        unica = salida.reindex(datos.index).fillna(0).to_numpy(int) if salida is not None else np.zeros(n, int)
        sal = None

    efectivo, posicion = float(capital), None
    hist_efectivo, hist_posicion, hist_equity = np.empty(n), np.zeros(n), np.empty(n)
    operaciones, comisiones, deslizamientos = [], 0.0, 0.0

    def senal_salida(t):
        if sal is None:
            return unica[t]
        columna = sal.get(posicion.regimen)
        return 0 if columna is None else columna[t]

    def abrir(t, lado):
        nonlocal efectivo, posicion, comisiones, deslizamientos
        acciones = tamano_por_fraccion(efectivo, o[t], min(param["fraccion"][t - 1], 1.0), costos.comision)
        d = _deslizamiento(costos, o[t], acciones, sigma[t - 1], adv[t])
        precio = o[t] + lado * d
        stop, objetivo = niveles(lado, precio, atr[t - 1], param["sl_atr"][t - 1], param["tp_atr"][t - 1])
        acciones = min(acciones, efectivo / (precio * (1 + costos.comision)))  # sin apalancamiento
        comision = costos.comision * acciones * precio
        efectivo -= lado * acciones * precio + comision
        comisiones += comision
        deslizamientos += acciones * d
        posicion = Posicion(lado, acciones, precio, stop, objetivo, datos.index[t],
                            max_velas=param["max_velas"][t - 1], regimen=reg[t - 1],
                            trailing=bool(param["trailing"][t - 1] > 0), distancia=abs(precio - stop))
        operaciones.append({"lado": lado, "entrada": datos.index[t], "precio_entrada": precio, "acciones": acciones,
                            "stop_loss": stop, "take_profit": objetivo, "comision": comision,
                            "deslizamiento": acciones * d, "regimen": reg[t - 1]})

    def cerrar(t, precio_ref, motivo):
        nonlocal efectivo, posicion, comisiones, deslizamientos
        d = _deslizamiento(costos, precio_ref, posicion.acciones, sigma[t - 1] if t > 0 else np.nan, adv[t])
        precio = precio_ref - posicion.lado * d
        comision = costos.comision * posicion.acciones * precio
        efectivo += posicion.lado * posicion.acciones * precio - comision
        comisiones += comision
        deslizamientos += posicion.acciones * d
        op = operaciones[-1]
        op.update({"salida": datos.index[t], "precio_salida": precio, "motivo": motivo, "velas": posicion.velas})
        op["comision"] += comision
        op["deslizamiento"] += posicion.acciones * d
        op["pnl"] = posicion.lado * posicion.acciones * (precio - op["precio_entrada"]) - op["comision"]
        posicion = None

    for t in range(n):
        # 1) Órdenes decididas al cierre de t−1, ejecutadas en la apertura de t.
        if t > 0 and posicion is not None:
            if s[t - 1] == -posicion.lado:
                cerrar(t, o[t], "senal_opuesta")
            elif not posicion.trailing and senal_salida(t - 1) in (2, -posicion.lado):
                cerrar(t, o[t], "salida")
        if (t > 0 and posicion is None and s[t - 1] != 0 and dia[t] == dia[t - 1]
                and np.isfinite(atr[t - 1]) and atr[t - 1] > 0):
            abrir(t, int(s[t - 1]))
        # 2) Stop y objetivo dentro de la vela; holding máximo y fin de datos al cierre.
        if posicion is not None:
            fuera = posicion.salida_intrabar(h[t], l[t], o[t])
            if fuera is not None:
                cerrar(t, *fuera)
            else:
                posicion.velas += 1
                if posicion.velas >= posicion.max_velas:
                    cerrar(t, c[t], "holding_maximo")
                elif t == n - 1:
                    cerrar(t, c[t], "fin_de_datos")
                elif posicion.trailing:  # decidido al cierre de t, vale desde la vela t+1
                    if posicion.lado == 1:
                        posicion.stop_loss = max(posicion.stop_loss, c[t] - posicion.distancia)
                    else:
                        posicion.stop_loss = min(posicion.stop_loss, c[t] + posicion.distancia)
        hist_efectivo[t] = efectivo
        hist_posicion[t] = 0.0 if posicion is None else posicion.lado * posicion.acciones
        hist_equity[t] = efectivo + hist_posicion[t] * c[t]

    return {
        "equity": pd.Series(hist_equity, index=datos.index, name="equity"),
        "efectivo": pd.Series(hist_efectivo, index=datos.index, name="efectivo"),
        "posicion": pd.Series(hist_posicion, index=datos.index, name="posicion"),
        "operaciones": pd.DataFrame(operaciones),
        "comisiones": comisiones,
        "deslizamiento": deslizamientos,
    }


def buy_and_hold(datos: pd.DataFrame, costos: Costos = Costos(), capital: float = CAPITAL) -> tuple[pd.Series, pd.DataFrame]:
    """Compra todo en la apertura de la primera vela y vende al cierre de la última, con comisión en ambos lados
    (sin spread ni impacto). Regresa la curva de valor y la operación, para medir turnover igual que la estrategia."""
    entrada = datos["Open"].iloc[0]
    acciones = capital / (entrada * (1 + costos.comision))
    valor = capital - acciones * entrada * (1 + costos.comision) + acciones * datos["Close"]
    valor.iloc[-1] -= costos.comision * acciones * datos["Close"].iloc[-1]
    op = pd.DataFrame([{"lado": 1, "entrada": datos.index[0], "precio_entrada": entrada, "acciones": acciones,
                        "salida": datos.index[-1], "precio_salida": datos["Close"].iloc[-1],
                        "pnl": valor.iloc[-1] - capital, "regimen": None}])
    return valor.rename("buy_hold"), op
