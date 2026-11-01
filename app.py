"""
app.py  -  Aplicación de Distribuciones Muestrales (Ingeniería Industrial)
Para ejecutarla:   streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

import calculos as c

st.set_page_config(page_title="Distribuciones muestrales", layout="wide")
st.title("📊 Distribuciones muestrales")
st.caption("Aplicación de apoyo para Estadística – Ingeniería Industrial")


# ---------------------------------------------------------------
# Función auxiliar: dibuja la curva normal y sombrea la probabilidad
# ---------------------------------------------------------------
def graficar_normal(media, error, tipo, a, b=None, etiqueta="x"):
    x = np.linspace(media - 4 * error, media + 4 * error, 400)
    y = stats.norm.pdf(x, media, error)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, y, color="navy")

    if tipo == "menor":
        zona = x <= a
    elif tipo == "mayor":
        zona = x >= a
    else:
        zona = (x >= a) & (x <= b)

    ax.fill_between(x, y, where=zona, color="orange", alpha=0.6)
    ax.axvline(media, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(etiqueta)
    ax.set_ylabel("Densidad")
    st.pyplot(fig)


def pedir_evento(clave, valor_a, valor_b=None):
    """Controles para elegir el evento: menor, mayor o entre."""
    tipo_txt = st.radio("Evento", ["Menor que", "Mayor que", "Entre"],
                        horizontal=True, key=clave + "_tipo")
    tipo = {"Menor que": "menor", "Mayor que": "mayor", "Entre": "entre"}[tipo_txt]
    a = st.number_input("Valor a", value=float(valor_a), key=clave + "_a", format="%.4f")
    b = None
    if tipo == "entre":
        b = st.number_input("Valor b", value=float(valor_b if valor_b is not None else valor_a),
                            key=clave + "_b", format="%.4f")
    return tipo, a, b


def texto_evento(tipo, a, b, simbolo):
    if tipo == "menor":
        return f"P({simbolo} < {a:g})"
    if tipo == "mayor":
        return f"P({simbolo} > {a:g})"
    return f"P({a:g} < {simbolo} < {b:g})"


# ---------------------------------------------------------------
# Menú lateral
# ---------------------------------------------------------------
seccion = st.sidebar.radio("Elige una sección", [
    "1. Población y muestras (con / sin reposición)",
    "2. Problema de medias + normal",
    "3. Distribución muestral de proporciones",
    "4. Diferencias y sumas",
])


# ===============================================================
# SECCIÓN 1
# ===============================================================
if seccion.startswith("1"):
    st.header("1. Población y todas las muestras posibles")

    texto = st.text_input("Población (números separados por coma)", "1, 9, 12, 14, 16, 20")
    n = st.number_input("Tamaño de la muestra (n)", min_value=1, value=2, step=1)

    try:
        poblacion = [float(v) for v in texto.replace(";", ",").split(",") if v.strip() != ""]
    except ValueError:
        st.error("Escribe solo números separados por comas.")
        st.stop()

    if n > len(poblacion):
        st.error("n no puede ser mayor que el tamaño de la población.")
        st.stop()

    mu, var, sigma = c.parametros_poblacion(poblacion)
    st.subheader("Parámetros de la población")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("N", len(poblacion))
    col2.metric("Media μ", f"{mu:.4f}")
    col3.metric("Varianza σ²", f"{var:.4f}")
    col4.metric("Desviación σ", f"{sigma:.4f}")

    for titulo, con_rep in [("a) CON reposición", True), ("b) SIN reposición", False)]:
        st.subheader(titulo)
        tabla, m_med, v_med, d_med = c.distribucion_muestral_de_medias(poblacion, int(n), con_rep)
        t_mu, t_var, t_sig = c.formulas_teoricas(poblacion, int(n), con_rep)

        st.write(f"Número de muestras posibles: **{len(tabla)}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Media de las medias μx̄", f"{m_med:.4f}")
        c2.metric("Varianza de las medias σ²x̄", f"{v_med:.4f}")
        c3.metric("Desviación de las medias σx̄", f"{d_med:.4f}")

        st.caption(f"Comprobación con fórmula teórica → μx̄ = {t_mu:.4f}, "
                   f"σ²x̄ = {t_var:.4f}, σx̄ = {t_sig:.4f}")

        izq, der = st.columns([1, 1])
        with izq:
            st.dataframe(tabla, height=300)
        with der:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(tabla["Media muestral"], bins=10, color="steelblue", edgecolor="white")
            ax.axvline(m_med, color="red", linestyle="--", label="μx̄")
            ax.set_xlabel("Media muestral")
            ax.set_ylabel("Frecuencia")
            ax.legend()
            st.pyplot(fig)


# ===============================================================
# SECCIÓN 2
# ===============================================================
elif seccion.startswith("2"):
    st.header("2. Problema de medias")
    st.info("**Ejemplo:** una máquina llena botellas con media μ = 500 ml y desviación "
            "σ = 12 ml. Se toma una muestra de n = 36 botellas. ¿Cuál es la probabilidad "
            "de que la media muestral supere 503 ml?")

    col1, col2, col3 = st.columns(3)
    mu = col1.number_input("Media poblacional μ", value=500.0)
    sigma = col2.number_input("Desviación poblacional σ", value=12.0, min_value=0.0001)
    n = col3.number_input("Tamaño de muestra n", value=36, min_value=1, step=1)

    finita = st.checkbox("Población finita y muestreo sin reposición (usar factor de corrección)")
    N = None
    if finita:
        N = st.number_input("Tamaño de la población N", value=500, min_value=int(n) + 1, step=1)

    error = c.error_estandar_medias(sigma, int(n), N)
    st.write(f"**Media esperada de x̄:** {mu:.4f}")
    st.write(f"**Desviación esperada (error estándar) de x̄:** {error:.4f}")
    st.write(f"**Varianza esperada de x̄:** {error**2:.4f}")

    tipo, a, b = pedir_evento("s2", 503.0, 506.0)
    prob = c.probabilidad_normal(mu, error, tipo, a, b)

    if tipo == "menor":
        z_txt = f"z = {(a - mu) / error:.4f}"
    elif tipo == "mayor":
        z_txt = f"z = {(a - mu) / error:.4f}"
    else:
        z_txt = f"z1 = {(a - mu) / error:.4f} y z2 = {(b - mu) / error:.4f}"

    st.success(f"{texto_evento(tipo, a, b, 'x̄')} = **{prob:.4f}**  ({prob*100:.2f} %)   |   {z_txt}")
    graficar_normal(mu, error, tipo, a, b, "Media muestral x̄")


# ===============================================================
# SECCIÓN 3
# ===============================================================
elif seccion.startswith("3"):
    st.header("3. Distribución muestral de proporciones")
    st.info("**Ejemplo:** el 8 % de las piezas de un proceso salen defectuosas (p = 0,08). "
            "Se inspeccionan n = 200 piezas. ¿Cuál es la probabilidad de que la proporción "
            "de defectuosas en la muestra sea mayor que 0,10?")

    col1, col2 = st.columns(2)
    p = col1.number_input("Proporción poblacional p", value=0.08, min_value=0.0001,
                          max_value=0.9999, format="%.4f")
    n = col2.number_input("Tamaño de muestra n", value=200, min_value=1, step=1)

    error = c.error_estandar_proporcion(p, int(n))
    st.write(f"**Media esperada de p̂:** {p:.4f}")
    st.write(f"**Desviación (error estándar) de p̂:** {error:.4f}")
    st.write(f"**Varianza de p̂:** {error**2:.6f}")

    if n * p < 5 or n * (1 - p) < 5:
        st.warning("Cuidado: n·p o n·(1−p) es menor que 5; la aproximación normal no es buena.")

    tipo, a, b = pedir_evento("s3", 0.10, 0.12)
    prob = c.probabilidad_normal(p, error, tipo, a, b)
    st.success(f"{texto_evento(tipo, a, b, 'p̂')} = **{prob:.4f}**  ({prob*100:.2f} %)")
    graficar_normal(p, error, tipo, a, b, "Proporción muestral p̂")


# ===============================================================
# SECCIÓN 4
# ===============================================================
else:
    st.header("4. Distribución muestral de diferencias y sumas")
    modo = st.radio("¿Con qué trabajas?", ["Medias", "Proporciones"], horizontal=True)
    operacion = st.radio("Operación", ["Diferencia", "Suma"], horizontal=True)

    if modo == "Medias":
        st.info("**Ejemplo:** tiempo de ensamble. Línea A: μ₁ = 50 s, σ₁ = 6, n₁ = 40. "
                "Línea B: μ₂ = 47 s, σ₂ = 5, n₂ = 35. ¿Probabilidad de que x̄₁ − x̄₂ > 5?")
        col1, col2 = st.columns(2)
        mu1 = col1.number_input("μ₁", value=50.0)
        s1 = col1.number_input("σ₁", value=6.0, min_value=0.0001)
        n1 = col1.number_input("n₁", value=40, min_value=1, step=1)
        mu2 = col2.number_input("μ₂", value=47.0)
        s2 = col2.number_input("σ₂", value=5.0, min_value=0.0001)
        n2 = col2.number_input("n₂", value=35, min_value=1, step=1)

        if operacion == "Diferencia":
            media, error = c.diferencia_de_medias(mu1, s1, int(n1), mu2, s2, int(n2))
            simbolo = "x̄₁ − x̄₂"
            valor_por_defecto = 5.0
        else:
            media, error = c.suma_de_medias(mu1, s1, int(n1), mu2, s2, int(n2))
            simbolo = "x̄₁ + x̄₂"
            valor_por_defecto = 100.0
    else:
        st.info("**Ejemplo:** defectuosos. Proceso 1: p₁ = 0,10, n₁ = 150. "
                "Proceso 2: p₂ = 0,06, n₂ = 200. ¿Probabilidad de que p̂₁ − p̂₂ > 0,05?")
        col1, col2 = st.columns(2)
        p1 = col1.number_input("p₁", value=0.10, min_value=0.0001, max_value=0.9999, format="%.4f")
        n1 = col1.number_input("n₁", value=150, min_value=1, step=1)
        p2 = col2.number_input("p₂", value=0.06, min_value=0.0001, max_value=0.9999, format="%.4f")
        n2 = col2.number_input("n₂", value=200, min_value=1, step=1)

        if operacion == "Diferencia":
            media, error = c.diferencia_de_proporciones(p1, int(n1), p2, int(n2))
            simbolo = "p̂₁ − p̂₂"
            valor_por_defecto = 0.05
        else:
            media, error = c.suma_de_proporciones(p1, int(n1), p2, int(n2))
            simbolo = "p̂₁ + p̂₂"
            valor_por_defecto = 0.20

    st.write(f"**Media esperada de {simbolo}:** {media:.4f}")
    st.write(f"**Desviación (error estándar):** {error:.4f}")
    st.write(f"**Varianza:** {error**2:.6f}")

    tipo, a, b = pedir_evento("s4", valor_por_defecto, valor_por_defecto * 1.5)
    prob = c.probabilidad_normal(media, error, tipo, a, b)
    st.success(f"{texto_evento(tipo, a, b, simbolo)} = **{prob:.4f}**  ({prob*100:.2f} %)")
    graficar_normal(media, error, tipo, a, b, simbolo)
