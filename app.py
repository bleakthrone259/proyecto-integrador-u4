"""
=============================================================================
 PROYECTO INTEGRADOR - UNIDAD 4
 "Optimizacion Inferencial del Rendimiento Agricola y Eficiencia de Riego
  en Cultivos de Invernadero"
=============================================================================
 Asignatura : Analisis y Exploracion de Datos (INC-2505)
 Alumno     : Torres Hernandez Dorian Idari  -  No. control: 23680452
 Docente    : Aristides Caballero Alfaro
 Instituto  : Instituto Tecnologico de Cuautla - TecNM

 Dashboard web (Fase 5 / Unidad 6). Ejecutar con:
     streamlit run app.py
=============================================================================
"""

import math

import numpy as np
import pandas as pd
import streamlit as st

from analisis import (
    ALPHA,
    ETIQUETA_A,
    ETIQUETA_B,
    N_PARCELAS,
    SEMILLA,
    ajustar_ols,
    ajustar_ols_extendido,
    generar_telemetria,
    matriz_correlacion,
    prueba_t_tratamientos,
)
from graficas import (
    grafica_boxplot,
    grafica_dispersion_ols,
    grafica_heatmap,
    grafica_residuos,
)

# ---------------------------------------------------------------------------
# Configuracion de la pagina
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Optimización Inferencial del Rendimiento Agrícola",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

ESTILO = """
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
  div[data-testid="stMetric"] {
      background: #ffffff;
      border: 1px solid rgba(11,11,11,0.10);
      border-radius: 12px;
      padding: 16px 18px;
  }
  div[data-testid="stMetricLabel"] p { font-size: 0.82rem; color: #52514e; }
  div[data-testid="stMetricValue"] { font-size: 1.9rem; color: #0b0b0b; }
  h1 { font-size: 1.85rem !important; letter-spacing: -0.01em; }
  .subtitulo { color: #52514e; font-size: 1rem; margin-top: -0.5rem; }
  .pie { color: #898781; font-size: 0.82rem; }
</style>
"""
st.markdown(ESTILO, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Datos (se simulan una sola vez y quedan en cache)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Simulando telemetría del invernadero…")
def cargar_datos() -> pd.DataFrame:
    return generar_telemetria()


df_completo = cargar_datos()


# ---------------------------------------------------------------------------
# Barra lateral: controles interactivos
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌱 Panel de control")
    st.caption(
        "Los filtros recalculan en tiempo real las métricas, el modelo de "
        "regresión y la prueba de hipótesis."
    )

    tratamientos = st.multiselect(
        "Tipo de tratamiento",
        options=[ETIQUETA_A, ETIQUETA_B],
        default=[ETIQUETA_A, ETIQUETA_B],
        format_func=lambda s: s.replace("_", " "),
        help="Tratamiento A = método estándar · Tratamiento B = bio-estimulante foliar",
    )

    # Se redondea hacia afuera (floor/ceil) para que los extremos del slider
    # siempre contengan todos los registros y ninguna parcela quede excluida.
    riego_min = math.floor(df_completo["volumen_riego"].min())
    riego_max = math.ceil(df_completo["volumen_riego"].max())
    rango_riego = st.slider(
        "Volumen de riego (m³/ha)",
        min_value=float(riego_min), max_value=float(riego_max),
        value=(float(riego_min), float(riego_max)), step=1.0,
    )

    fert_min = math.floor(df_completo["fertilizante_kg"].min())
    fert_max = math.ceil(df_completo["fertilizante_kg"].max())
    rango_fert = st.slider(
        "Fertilizante aplicado (kg/ha)",
        min_value=float(fert_min), max_value=float(fert_max),
        value=(float(fert_min), float(fert_max)), step=1.0,
    )

    st.divider()
    st.markdown("#### Ficha del proyecto")
    st.markdown(
        f"""
        <div class="pie">
        <b>Alumno:</b> Torres Hernández Dorian Idari<br>
        <b>No. de control:</b> 23680452<br>
        <b>Asignatura:</b> Análisis y Exploración de Datos (INC-2505)<br>
        <b>Docente:</b> Arístides Caballero Alfaro<br>
        <b>Semilla:</b> <code>setseed({SEMILLA})</code> · N = {N_PARCELAS} parcelas
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Aplicacion de los filtros
# ---------------------------------------------------------------------------
df = df_completo[
    df_completo["tipo_tratamiento"].isin(tratamientos)
    & df_completo["volumen_riego"].between(*rango_riego)
    & df_completo["fertilizante_kg"].between(*rango_fert)
].copy()


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("Optimización inferencial del rendimiento agrícola")
st.markdown(
    '<p class="subtitulo">Invernaderos inteligentes de jitomate · '
    "análisis de regresión, correlación e inferencia sobre telemetría de riego "
    "y fertilización</p>",
    unsafe_allow_html=True,
)
st.write("")

if df.empty or len(df) < 3:
    st.warning(
        "Los filtros actuales dejan menos de 3 parcelas. "
        "Amplía los rangos para poder ajustar el modelo."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Calculos sobre el subconjunto filtrado
# ---------------------------------------------------------------------------
ols = ajustar_ols(df)
prueba = prueba_t_tratamientos(df)
correlaciones = matriz_correlacion(df)


# ---------------------------------------------------------------------------
# Tarjetas KPI
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Rendimiento promedio global",
        f"{df['rendimiento_ton'].mean():.2f} ton/ha",
        help="Media del rendimiento sobre las parcelas que pasan el filtro.",
    )

with k2:
    st.metric(
        "R² ajustado del modelo",
        f"{ols.r2_ajustado * 100:.2f} %",
        help=(
            "Porcentaje de la variabilidad de la cosecha explicado por el "
            "riego y el fertilizante, penalizado por el número de predictores."
        ),
    )

with k3:
    if prueba.valida:
        st.metric(
            "P-valor de la prueba inferencial",
            f"{prueba.p_unilateral:.2e}",
            delta="Rechaza H₀" if prueba.rechaza_h0 else "No rechaza H₀",
            delta_color="normal" if prueba.rechaza_h0 else "off",
            help="Prueba t de dos muestras independientes, unilateral (H₁: μ_B > μ_A).",
        )
    else:
        st.metric("P-valor de la prueba inferencial", "n/d",
                  help=prueba.mensaje)

with k4:
    st.metric(
        "Parcelas analizadas",
        f"{len(df)} de {len(df_completo)}",
        help="Registros que pasan los filtros activos.",
    )

st.write("")

# ---------------------------------------------------------------------------
# Veredicto inferencial
# ---------------------------------------------------------------------------
if prueba.valida:
    if prueba.rechaza_h0:
        st.success(
            f"**Veredicto de ingeniería — se rechaza H₀.** "
            f"El Tratamiento B incrementa el rendimiento medio en "
            f"**{prueba.diferencia:.2f} ton/ha** respecto al Tratamiento A "
            f"(p = {prueba.p_unilateral:.2e} < α = {ALPHA}). "
            f"El intervalo de confianza al 95 % para la diferencia es "
            f"[{prueba.ic_inferior:.2f}, {prueba.ic_superior:.2f}] ton/ha y no "
            f"incluye el cero: **se autoriza escalar el bio-estimulante a todo "
            f"el invernadero.**"
        )
    else:
        st.error(
            f"**Veredicto de ingeniería — no se rechaza H₀.** "
            f"Con p = {prueba.p_unilateral:.4f} ≥ α = {ALPHA} no hay evidencia "
            f"suficiente de que el Tratamiento B incremente el rendimiento. "
            f"No se recomienda autorizar la inversión."
        )
else:
    st.info(f"**Prueba de hipótesis no disponible.** {prueba.mensaje}")

st.write("")

# ---------------------------------------------------------------------------
# Cuadricula 2x2 de graficas
# ---------------------------------------------------------------------------
fila1_izq, fila1_der = st.columns(2, gap="large")
with fila1_izq:
    st.pyplot(grafica_heatmap(correlaciones), use_container_width=True)
    st.caption(
        "El riego domina la productividad. La temperatura se incluye como "
        "variable ambiental de control y su correlación cercana a cero confirma "
        "que no interviene en la ecuación generadora del rendimiento."
    )
with fila1_der:
    st.pyplot(grafica_dispersion_ols(df), use_container_width=True)
    st.caption(
        "Cada punto es una parcela. La recta es el ajuste por mínimos cuadrados "
        "sobre el riego y la banda sombreada es su intervalo de confianza al 95 %."
    )

st.write("")

fila2_izq, fila2_der = st.columns(2, gap="large")
with fila2_izq:
    st.pyplot(grafica_boxplot(df), use_container_width=True)
    st.caption(
        "Soporte visual de la prueba t: la separación entre las medianas y las "
        "medias (rombos) es lo que la prueba evalúa formalmente."
    )
with fila2_der:
    st.pyplot(grafica_residuos(ols.residuos), use_container_width=True)
    st.caption(
        "Validación del supuesto inferencial: los errores deben distribuirse "
        "de forma aproximadamente normal alrededor de cero."
    )

st.write("")
st.divider()

# ---------------------------------------------------------------------------
# Detalle estadistico
# ---------------------------------------------------------------------------
st.markdown("### Detalle estadístico")

det1, det2 = st.columns(2, gap="large")

with det1:
    st.markdown("#### Modelo de regresión lineal múltiple")
    st.latex(
        r"\text{Rendimiento} = \beta_0 + \beta_1(\text{Riego})"
        r" + \beta_2(\text{Fertilizante}) + \epsilon"
    )
    st.markdown(
        f"""
| Parámetro | Valor | Valor p |
|---|---|---|
| β₀ (intercepto) | {ols.beta_0:.4f} ton/ha | — |
| β₁ (riego) | {ols.beta_riego:.5f} ton por m³/ha | {ols.p_riego:.3e} |
| β₂ (fertilizante) | {ols.beta_fertilizante:.5f} ton por kg/ha | {ols.p_fertilizante:.3e} |
| R² | {ols.r2:.4f} | — |
| R² ajustado | {ols.r2_ajustado:.4f} | — |
| RMSE | {ols.rmse:.4f} ton/ha | — |
"""
    )
    with st.expander("Ver resumen completo de statsmodels (OLS Regression Results)"):
        st.code(str(ols.modelo.summary()), language="text")

with det2:
    st.markdown("#### Prueba de hipótesis sobre los tratamientos")
    st.latex(r"H_0:\ \mu_B - \mu_A \leq 0 \qquad H_1:\ \mu_B - \mu_A > 0")
    if prueba.valida:
        st.markdown(
            f"""
| Estadístico | Valor |
|---|---|
| n (Tratamiento A / B) | {prueba.n_a} / {prueba.n_b} |
| Media A / Media B | {prueba.media_a:.3f} / {prueba.media_b:.3f} ton/ha |
| Diferencia observada (B − A) | **{prueba.diferencia:.3f} ton/ha** |
| Estadístico t | {prueba.t_stat:.4f} |
| Grados de libertad | {prueba.gl:.2f} |
| Valor p (unilateral) | **{prueba.p_unilateral:.4e}** |
| IC 95 % de la diferencia | [{prueba.ic_inferior:.3f}, {prueba.ic_superior:.3f}] ton/ha |
| Levene (homogeneidad de varianzas) | p = {prueba.levene_p:.4f} |
| Tamaño del efecto (d de Cohen) | {prueba.cohen_d:.4f} |
"""
        )
    else:
        st.info(prueba.mensaje)

st.write("")

# ---------------------------------------------------------------------------
# Diagnostico de supuestos y modelo extendido de control
# ---------------------------------------------------------------------------
ols_ext = ajustar_ols_extendido(df)

with st.expander("Diagnóstico de los supuestos del modelo y modelo extendido de control"):
    st.markdown(
        f"""
La **Gráfica 4** muestra que los residuos del modelo base no forman una campana
única, sino **dos modas separadas**. La prueba de Shapiro-Wilk lo confirma:
**p = {ols.shapiro_p:.3e}**, por debajo de α = {ALPHA}, de modo que se rechaza la
normalidad de los errores.

El diagnóstico es una **variable omitida**. El modelo que especifica el enunciado
incluye el riego y el fertilizante, pero no el tratamiento, y el bio-estimulante
desplaza el rendimiento de casi la mitad de las parcelas de forma sistemática. Ese
desplazamiento no puede ir al ajuste, así que termina en los residuos y los parte
en dos grupos.
"""
    )

    if ols_ext is not None:
        beta_trat = float(ols_ext.modelo.params["trat_B"])
        p_trat = float(ols_ext.modelo.pvalues["trat_B"])

        st.markdown("**Modelo extendido de control** (agrega el tratamiento como variable ficticia):")
        st.latex(
            r"\text{Rendimiento} = \beta_0 + \beta_1(\text{Riego})"
            r" + \beta_2(\text{Fertilizante}) + \beta_3(\text{Tratamiento\_B}) + \epsilon"
        )
        st.markdown(
            f"""
| Métrica | Modelo base (enunciado) | Modelo extendido |
|---|---|---|
| R² ajustado | {ols.r2_ajustado * 100:.2f} % | **{ols_ext.r2_ajustado * 100:.2f} %** |
| RMSE | {ols.rmse:.4f} ton/ha | **{ols_ext.rmse:.4f} ton/ha** |
| Shapiro-Wilk (residuos) | p = {ols.shapiro_p:.3e} | p = {ols_ext.shapiro_p:.4f} |
| Durbin-Watson | {ols.durbin_watson:.3f} | {ols_ext.durbin_watson:.3f} |
| β₀ (intercepto) | {ols.beta_0:.4f} | **{ols_ext.beta_0:.4f}** |
| β₃ (bono del Tratamiento B) | — | **{beta_trat:.4f} ton/ha** (p = {p_trat:.2e}) |
"""
        )
        st.markdown(
            f"""
Al incorporar el tratamiento, la bimodalidad desaparece, el **R² ajustado sube
{(ols_ext.r2_ajustado - ols.r2_ajustado) * 100:.2f} puntos** y el RMSE se reduce a
menos de la mitad. Más revelador aún: el intercepto baja de {ols.beta_0:.2f} a
{ols_ext.beta_0:.2f} ton/ha, muy cerca del valor real de diseño de 5.0, porque en el
modelo base **el intercepto estaba absorbiendo el efecto promedio del tratamiento**.
El coeficiente β₃ = {beta_trat:.2f} ton/ha recupera por su cuenta el bono de diseño
de +1.8 ton/ha, y coincide con la diferencia de medias que estimó la prueba t.

Queda una observación fina: incluso el modelo extendido mantiene
p = {ols_ext.shapiro_p:.4f} en Shapiro-Wilk. No es un defecto del modelo, sino del
proceso generador: el enunciado define el ruido como una variación **uniforme**
entre −0.8 y +0.8 ton/ha, no gaussiana. Una distribución uniforme es platicúrtica y
Shapiro-Wilk la distingue de una normal con n = {len(df)}. Con ese tamaño de muestra,
tanto la prueba t como los errores estándar del OLS son robustos a esa desviación
por el teorema del límite central, de modo que las conclusiones inferenciales se
sostienen.
"""
        )
    else:
        st.info(
            "El modelo extendido necesita parcelas de los dos tratamientos. "
            "Activa ambos en el filtro lateral para verlo."
        )

st.write("")

with st.expander(f"Ver tabla de telemetría ({len(df)} parcelas filtradas)"):
    st.dataframe(
        df.rename(columns={
            "parcela_id": "Parcela",
            "volumen_riego": "Riego (m³/ha)",
            "fertilizante_kg": "Fertilizante (kg/ha)",
            "tipo_tratamiento": "Tratamiento",
            "temperatura_c": "Temperatura (°C)",
            "rendimiento_ton": "Rendimiento (ton/ha)",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Descargar CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="telemetria_parcelas.csv",
        mime="text/csv",
    )

st.divider()
st.markdown(
    '<p class="pie">Instituto Tecnológico de Cuautla · TecNM · '
    "Proyecto Integrador Unidad 4 · Análisis y Exploración de Datos (INC-2505) · "
    "Torres Hernández Dorian Idari — 23680452</p>",
    unsafe_allow_html=True,
)
