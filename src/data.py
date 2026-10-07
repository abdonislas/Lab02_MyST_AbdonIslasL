"""Carga, validación y auditoría de datos.

Activo: Novartis AG (NYSE: NVS, ADR), velas de 1 hora de la sesión regular (09:30 a 16:00 de Nueva York),
precios ajustados por dividendos y splits, descargadas de Yahoo Finance con yfinance (el límite de Yahoo para
velas de 1 hora es de 730 días). Cada sesión tiene 7 velas: 09:30, 10:30, ..., 15:30 (la última dura 30 min).

Los datos crudos se congelan en data/nvs_1h.csv y se versionan en git: Yahoo recorre su ventana de 730 días
todos los días, así que una descarga nueva cambiaría el periodo y el split. El proyecto siempre lee el CSV
congelado; solo se descarga si el archivo no existe.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TICKER = "NVS"
ZONA = "America/New_York"
RUTA_CRUDOS = Path("data/nvs_1h.csv")
VELAS_POR_DIA = 7
FRACCIONES = (0.6, 0.2, 0.2)  # train, test, validation (por sesiones completas)


def descargar(ticker: str = TICKER, ruta: Path = RUTA_CRUDOS) -> Path:
    """Descarga unos 2 años de velas de 1 hora de Yahoo Finance y las guarda tal cual en `ruta`."""
    import yfinance as yf
    d = yf.download(ticker, interval="1h", period="730d", auto_adjust=True, prepost=False, progress=False)
    if d is None or len(d) == 0:
        raise RuntimeError(f"Yahoo Finance no regresó datos de {ticker}")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(ruta)
    return ruta


def cargar_datos(ruta: Path = RUTA_CRUDOS) -> pd.DataFrame:
    """Lee el CSV congelado y regresa OHLCV en hora de Nueva York, solo sesión regular, ordenado y sin duplicados."""
    df = pd.read_csv(ruta, index_col=0)
    df = df[pd.to_numeric(df["Close"], errors="coerce").notna()]
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(ZONA)
    df.index.name = "Datetime"
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df[~df.index.duplicated()].sort_index()
    return df.between_time("09:30", "15:30")


def huella(ruta: Path = RUTA_CRUDOS) -> str:
    """SHA-256 del archivo de datos: prueba de que validation usa los mismos datos que se congelaron.

    Se normalizan los fines de línea (CRLF → LF) porque git los convierte al clonar en Windows o en Linux/Mac.
    """
    return hashlib.sha256(Path(ruta).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def auditar_datos(df: pd.DataFrame, velas_por_dia: int = VELAS_POR_DIA) -> dict:
    """Conteos de calidad: duplicados, nulos, OHLC incoherente, sesiones incompletas y huecos."""
    dias = pd.Series(df.index.date)
    por_dia = dias.value_counts()
    incoherente = (df["High"] < df[["Open", "Close"]].max(axis=1)) | (df["Low"] > df[["Open", "Close"]].min(axis=1))
    return {
        "velas": int(len(df)),
        "dias": int(dias.nunique()),
        "inicio": str(df.index[0]),
        "fin": str(df.index[-1]),
        "duplicados": int(df.index.duplicated().sum()),
        "nulos": int(df.isna().sum().sum()),
        "ohlc_incoherente": int(incoherente.sum()),
        "volumen_cero": int((df["Volume"] <= 0).sum()),
        "dias_incompletos": int((por_dia < velas_por_dia).sum()),
        "retorno_maximo_abs": float(np.log(df["Close"]).diff().abs().max()),
    }


def separar_train_test_validacion(df: pd.DataFrame, fracciones=FRACCIONES):
    """Split cronológico 60/20/20 por sesiones completas (ninguna sesión queda partida)."""
    dias = np.array(sorted(set(df.index.date)))
    a = int(round(len(dias) * fracciones[0]))
    b = int(round(len(dias) * (fracciones[0] + fracciones[1])))
    d = np.asarray(df.index.date)
    return df[d < dias[a]], df[(d >= dias[a]) & (d < dias[b])], df[d >= dias[b]]
