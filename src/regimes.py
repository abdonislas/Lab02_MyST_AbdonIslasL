"""Act 07 · Detección de régimen con tres clasificadores: reglas, K-means y HMM.

Features (`regime_features`), en una ventana móvil de 1 semana (35 velas de 1 hora):
- volatilidad / log_volatilidad: desviación estándar de los log-retornos, escalada a un día (√7).
- eficiencia: eficiencia de Kaufman |ln P_t − ln P_{t−35}| / Σ |Δ ln P| (1 = línea recta, 0 = va y viene).
- raiz_eficiencia: √eficiencia (menos sesgada para agrupar).
- autocorrelacion: correlación de los retornos con su rezago de una vela (negativa = reversión).
- retorno_semana: ln P_t − ln P_{t−35}.

Los tres clasificadores se ajustan solo con datos pasados y se reclasifican cada 4 velas (4 horas) en
posiciones fijas desde el inicio de la serie; la etiqueta se mantiene entre actualizaciones.
Nombres por centroides: crisis = mayor volatilidad; de los otros dos, tendencia = mayor eficiencia.
El HMM etiqueta con probabilidades filtradas P(s_t | x_1..x_t) (algoritmo forward), nunca con Viterbi,
que reetiqueta el pasado usando el futuro; Viterbi solo se calcula para compararlo.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

VENTANA = 35
CADA = 4
VELAS_POR_HORA = 1
VELAS_POR_DIA = 7
SEMILLA = 42
REGIMENES = ("tendencia", "reversion", "crisis")
SIN_DATOS = "sin_datos"
MODELO = ["log_volatilidad", "raiz_eficiencia"]


def regime_features(df: pd.DataFrame, ventana: int = VENTANA) -> pd.DataFrame:
    """Features de régimen en cada vela con la semana previa, incluida la vela t (causales)."""
    lp = np.log(df["Close"])
    r = lp.diff()
    vol = r.rolling(ventana).std() * np.sqrt(VELAS_POR_DIA)
    ef = lp.diff(ventana).abs() / r.abs().rolling(ventana).sum()
    return pd.DataFrame({
        "volatilidad": vol,
        "log_volatilidad": np.log(vol),
        "eficiencia": ef,
        "raiz_eficiencia": np.sqrt(ef),
        "autocorrelacion": r.rolling(ventana).corr(r.shift(1)),
        "retorno_semana": lp.diff(ventana),
    }, index=df.index)


def puntos(features: pd.DataFrame, cada: int = CADA) -> pd.DataFrame:
    """Velas de reclasificación: posiciones múltiplo de `cada` con features completas."""
    pos = np.arange(len(features))
    return features[(pos % cada == 0) & features[MODELO].notna().all(axis=1).to_numpy()]


def _nombrar(centros: pd.DataFrame) -> dict:
    crisis = int(centros["log_volatilidad"].idxmax())
    resto = centros.drop(index=crisis)
    tendencia = int(resto["raiz_eficiencia"].idxmax())
    reversion = int(resto.index.drop(tendencia)[0])
    return {tendencia: "tendencia", reversion: "reversion", crisis: "crisis"}


def _a_vela(features: pd.DataFrame, etiquetas_puntos: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=features.index, dtype=object)
    out.loc[etiquetas_puntos.index] = etiquetas_puntos.to_numpy()
    return out.ffill().fillna(SIN_DATOS).rename("regimen")


@dataclass
class Reglas:
    """Crisis si la volatilidad supera su percentil 75 de train; si no, tendencia si la eficiencia
    supera su percentil 60 de train; reversión en otro caso."""
    q_vol: float = 0.75
    q_ef: float = 0.60
    umbrales: dict = field(default_factory=dict)

    def ajustar(self, f_train: pd.DataFrame):
        p = puntos(f_train)
        self.umbrales = {"log_volatilidad": p["log_volatilidad"].quantile(self.q_vol),
                         "raiz_eficiencia": p["raiz_eficiencia"].quantile(self.q_ef)}
        return self

    def etiquetar(self, features: pd.DataFrame) -> pd.Series:
        p = puntos(features)
        e = np.where(p["log_volatilidad"] > self.umbrales["log_volatilidad"], "crisis",
                     np.where(p["raiz_eficiencia"] > self.umbrales["raiz_eficiencia"], "tendencia", "reversion"))
        return _a_vela(features, pd.Series(e, index=p.index))


@dataclass
class KMedias:
    """K-means (k = 3) sobre las features estandarizadas con train."""
    escalador: StandardScaler = None
    modelo: KMeans = None
    nombres: dict = None

    def ajustar(self, f_train: pd.DataFrame):
        X = puntos(f_train)[MODELO]
        self.escalador = StandardScaler().fit(X)
        self.modelo = KMeans(3, n_init=10, random_state=SEMILLA).fit(self.escalador.transform(X))
        self.nombres = _nombrar(pd.DataFrame(self.escalador.inverse_transform(self.modelo.cluster_centers_), columns=MODELO))
        return self

    def centros(self) -> pd.DataFrame:
        c = pd.DataFrame(self.escalador.inverse_transform(self.modelo.cluster_centers_), columns=MODELO)
        return c.rename(index=self.nombres)

    def etiquetar(self, features: pd.DataFrame) -> pd.Series:
        """Cada punto de reclasificación toma el centroide más cercano en el espacio estandarizado."""
        p = puntos(features)
        z = (p[MODELO].to_numpy() - self.escalador.mean_) / self.escalador.scale_
        g = np.argmin(((z[:, None, :] - self.modelo.cluster_centers_[None, :, :]) ** 2).sum(axis=2), axis=1)
        return _a_vela(features, pd.Series([self.nombres[int(i)] for i in g], index=p.index))

    def a_dict(self) -> dict:
        """Lo necesario para etiquetar sin volver a ajustar: escala y centroides (para congelar θ*)."""
        return {"media": self.escalador.mean_.tolist(), "escala": self.escalador.scale_.tolist(),
                "centros": self.modelo.cluster_centers_.tolist(),
                "nombres": {str(k): v for k, v in self.nombres.items()}}

    @classmethod
    def desde_dict(cls, d: dict) -> "KMedias":
        m = cls()
        m.escalador = StandardScaler()
        m.escalador.mean_, m.escalador.scale_ = np.array(d["media"]), np.array(d["escala"])
        m.modelo = KMeans(3)
        m.modelo.cluster_centers_ = np.array(d["centros"])
        m.nombres = {int(k): v for k, v in d["nombres"].items()}
        return m


@dataclass
class HMM:
    """HMM gaussiano de 3 estados (covarianza completa) ajustado con todas las velas de train.

    Se ajusta con 8 inicializaciones y se queda la de mayor verosimilitud: con una sola, EM cae en
    soluciones degeneradas (dos estados casi iguales que se alternan). Las probabilidades filtradas se
    calculan vela por vela y la etiqueta se toma en los puntos de actualización, como en los otros métodos.
    """
    escalador: StandardScaler = None
    modelo: GaussianHMM = None
    nombres: dict = None
    reinicios: int = 8

    def ajustar(self, f_train: pd.DataFrame):
        X = f_train[MODELO].dropna()
        self.escalador = StandardScaler().fit(X)
        Z = self.escalador.transform(X)
        mejor = None
        for semilla in range(SEMILLA, SEMILLA + self.reinicios):
            m = GaussianHMM(3, covariance_type="full", n_iter=300, random_state=semilla).fit(Z)
            if mejor is None or m.score(Z) > mejor.score(Z):
                mejor = m
        self.modelo = mejor
        self.nombres = _nombrar(pd.DataFrame(self.escalador.inverse_transform(mejor.means_), columns=MODELO))
        return self

    def centros(self) -> pd.DataFrame:
        c = pd.DataFrame(self.escalador.inverse_transform(self.modelo.means_), columns=MODELO)
        return c.rename(index=self.nombres)

    def probabilidades_filtradas(self, features: pd.DataFrame) -> pd.DataFrame:
        """Algoritmo forward vela por vela: α_t ∝ (α_{t−1} A) ⊙ b(x_t), normalizado; solo usa x_1..x_t."""
        validas = features[MODELO].dropna()
        X = self.escalador.transform(validas)
        m = self.modelo
        log_b = np.column_stack([multivariate_normal(m.means_[k], m.covars_[k]).logpdf(X) for k in range(3)])
        alfa = np.empty_like(log_b)
        for t in range(len(X)):
            prior = m.startprob_ if t == 0 else alfa[t - 1] @ m.transmat_
            log_a = np.log(prior + 1e-300) + log_b[t]
            a = np.exp(log_a - log_a.max())
            alfa[t] = a / a.sum()
        return pd.DataFrame(alfa, index=validas.index, columns=[self.nombres[k] for k in range(3)])

    def etiquetar(self, features: pd.DataFrame) -> pd.Series:
        prob = self.probabilidades_filtradas(features)
        p = puntos(features)
        return _a_vela(features, prob.loc[p.index].idxmax(axis=1))

    def etiquetar_viterbi(self, features: pd.DataFrame) -> pd.Series:
        """Ruta de Viterbi sobre toda la secuencia: usa el futuro, solo para comparar."""
        validas = features[MODELO].dropna()
        g = pd.Series([self.nombres[i] for i in self.modelo.predict(self.escalador.transform(validas))], index=validas.index)
        p = puntos(features)
        return _a_vela(features, g.loc[p.index])


def duraciones(etiquetas: pd.Series) -> pd.DataFrame:
    """Rachas consecutivas de cada régimen con su duración en horas de sesión."""
    v = etiquetas[etiquetas != SIN_DATOS]
    racha = (v != v.shift()).cumsum()
    t = v.groupby(racha).agg(["first", "size"])
    t.columns = ["regimen", "velas"]
    t["horas"] = t["velas"] / VELAS_POR_HORA
    return t.reset_index(drop=True)


def comparar(features: pd.DataFrame, etiquetas: pd.Series) -> dict:
    """Silhouette, duración media, transiciones por mes y participación de cada régimen."""
    p = puntos(features)
    X = StandardScaler().fit_transform(p[MODELO])
    lab = etiquetas.reindex(p.index).to_numpy()
    ok = lab != SIN_DATOS
    rachas = duraciones(etiquetas)
    v = etiquetas[etiquetas != SIN_DATOS]
    meses = len(v) / (VELAS_POR_DIA * 21)
    return {
        "silhouette": float(silhouette_score(X[ok], lab[ok])) if len(set(lab[ok])) > 1 else np.nan,
        "duracion_media_h": float(rachas["horas"].mean()),
        "transiciones_por_mes": float((len(rachas) - 1) / meses),
        **{f"part_{r}": float((v == r).mean()) for r in REGIMENES},
    }


# Interfaz que usa el resto del lab: K-means es el clasificador elegido (ver notebook y reporte).
def calcular_variables_regimen(df: pd.DataFrame, ventana: int = VENTANA) -> pd.DataFrame:
    return regime_features(df, ventana)


def ajustar_modelo(variables: pd.DataFrame) -> KMedias:
    return KMedias().ajustar(variables)


def etiquetar(modelo, variables: pd.DataFrame) -> pd.Series:
    return modelo.etiquetar(variables)


def validar_regimenes(modelo, variables: pd.DataFrame, etiquetas: pd.Series) -> dict:
    """Métricas de validación de la Act 07 más la duración media por régimen."""
    rachas = duraciones(etiquetas)
    return {**comparar(variables, etiquetas),
            "duracion_media_por_regimen": rachas.groupby("regimen")["horas"].mean().round(1).to_dict()}
