"""
calculos.py
Funciones de estadística para la app de distribuciones muestrales.
Todas las funciones son cortas y están comentadas para que se entiendan.
"""
from itertools import product, combinations
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------
# 1. POBLACIÓN Y MUESTRAS
# ---------------------------------------------------------------
def parametros_poblacion(poblacion):
    """Devuelve media, varianza y desviación de la población (dividiendo entre N)."""
    datos = np.array(poblacion, dtype=float)
    media = datos.mean()
    varianza = ((datos - media) ** 2).mean()   # varianza poblacional: entre N
    desviacion = np.sqrt(varianza)
    return media, varianza, desviacion


def todas_las_muestras(poblacion, n, con_reposicion=True):
    """
    Lista todas las muestras posibles de tamaño n.
    - Con reposición: el orden importa -> N^n muestras (36 si N=6 y n=2)
    - Sin reposición: no importa el orden -> C(N,n) muestras (15 si N=6 y n=2)
    """
    if con_reposicion:
        return list(product(poblacion, repeat=n))
    return list(combinations(poblacion, n))


def distribucion_muestral_de_medias(poblacion, n, con_reposicion=True):
    """
    Calcula la media de cada muestra y las características
    de la distribución muestral de medias.
    """
    muestras = todas_las_muestras(poblacion, n, con_reposicion)
    medias = np.array([np.mean(m) for m in muestras])

    media_medias = medias.mean()
    varianza_medias = ((medias - media_medias) ** 2).mean()
    desviacion_medias = np.sqrt(varianza_medias)

    tabla = pd.DataFrame({
        "Muestra": [str(m) for m in muestras],
        "Media muestral": medias,
    })
    return tabla, media_medias, varianza_medias, desviacion_medias


def formulas_teoricas(poblacion, n, con_reposicion=True):
    """
    Valores teóricos para comparar con lo calculado:
    - Con reposición:  σx̄² = σ² / n
    - Sin reposición:  σx̄² = (σ² / n) * (N - n) / (N - 1)
    """
    mu, var, _ = parametros_poblacion(poblacion)
    N = len(poblacion)
    var_medias = var / n
    if not con_reposicion:
        var_medias = var_medias * (N - n) / (N - 1)
    return mu, var_medias, np.sqrt(var_medias)


# ---------------------------------------------------------------
# 2. PROBABILIDAD CON LA NORMAL (MEDIAS)
# ---------------------------------------------------------------
def probabilidad_normal(mu, sigma_error, tipo, a, b=None):
    """
    Calcula una probabilidad usando la normal N(mu, sigma_error).
    tipo: 'menor' -> P(X < a) | 'mayor' -> P(X > a) | 'entre' -> P(a < X < b)
    """
    dist = stats.norm(mu, sigma_error)
    if tipo == "menor":
        return dist.cdf(a)
    if tipo == "mayor":
        return 1 - dist.cdf(a)
    return dist.cdf(b) - dist.cdf(a)


def error_estandar_medias(sigma, n, N=None):
    """Error estándar de la media. Si se da N (población finita, sin reposición) se corrige."""
    error = sigma / np.sqrt(n)
    if N is not None:
        error *= np.sqrt((N - n) / (N - 1))
    return error


# ---------------------------------------------------------------
# 3. PROPORCIONES
# ---------------------------------------------------------------
def error_estandar_proporcion(p, n):
    """Desviación de la distribución muestral de proporciones."""
    return np.sqrt(p * (1 - p) / n)


# ---------------------------------------------------------------
# 4. DIFERENCIAS Y SUMAS
# ---------------------------------------------------------------
def diferencia_de_medias(mu1, s1, n1, mu2, s2, n2):
    """Media y error estándar de x̄1 - x̄2 (poblaciones independientes)."""
    media = mu1 - mu2
    error = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    return media, error


def suma_de_medias(mu1, s1, n1, mu2, s2, n2):
    """Media y error estándar de x̄1 + x̄2 (poblaciones independientes)."""
    media = mu1 + mu2
    error = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    return media, error


def diferencia_de_proporciones(p1, n1, p2, n2):
    """Media y error estándar de p̂1 - p̂2."""
    media = p1 - p2
    error = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return media, error


def suma_de_proporciones(p1, n1, p2, n2):
    """Media y error estándar de p̂1 + p̂2."""
    media = p1 + p2
    error = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return media, error
