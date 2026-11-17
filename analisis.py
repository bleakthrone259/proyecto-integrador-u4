"""
=============================================================================
 PROYECTO INTEGRADOR - UNIDAD 4
 "Optimizacion Inferencial del Rendimiento Agricola y Eficiencia de Riego
  en Cultivos de Invernadero"
=============================================================================
 Asignatura : Analisis y Exploracion de Datos (INC-2505)
 Alumno     : Torres Hernandez Dorian Idari  -  No. control: 23680452
 Docente    : Aristides Caballero Alfaro
=============================================================================
 Motor analitico del proyecto. Separa la logica estadistica de la capa de
 presentacion (app.py), de modo que el analisis pueda ejecutarse y validarse
 sin levantar la interfaz web.

   - Fase 1: ingesta determinista en DuckDB (Hito 1)
   - Fase 2: correlacion de Pearson y modelo OLS (Subtema 4.2)
   - Fase 3: prueba t de dos muestras independientes (Subtema 4.3)
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ---------------------------------------------------------------------------
# Parametros de diseno de la simulacion (fijados por el enunciado)
# ---------------------------------------------------------------------------
SEMILLA = 0.42
N_PARCELAS = 100

BETA_0 = 5.0          # Intercepto base (ton/ha)
BETA_RIEGO = 0.035    # Sensibilidad al riego  (100 m3 -> +3.5 ton)
BETA_FERT = 0.020     # Sensibilidad al fertilizante (50 kg -> +1.0 ton)
BONO_TRATAMIENTO_B = 1.8   # Bono directo del bio-estimulante (ton/ha)
RUIDO_MIN, RUIDO_MAX = -0.8, 0.8

ALPHA = 0.05          # Nivel de significancia

ETIQUETA_A = "Tratamiento_A"
ETIQUETA_B = "Tratamiento_B"


# ===========================================================================
#  FASE 1 — INGESTA DETERMINISTA EN DUCKDB
# ===========================================================================

SQL_TELEMETRIA = f"""
CREATE OR REPLACE TABLE telemetria_parcelas AS
WITH aleatorios AS (
    -- Cada random() es una llamada independiente, por eso se materializan
    -- primero todos los valores uniformes de la parcela. Asi la etiqueta de
    -- tratamiento y el bono que se suma al rendimiento provienen del MISMO
    -- sorteo, que es lo que exige el algoritmo del enunciado.
    SELECT
        parcela_id,
        random() AS u_riego,
        random() AS u_fertilizante,
        random() AS u_tratamiento,
        random() AS u_temperatura,
        random() AS u_ruido
    FROM generate_series(1, {N_PARCELAS}) AS t(parcela_id)
),
campos AS (
    SELECT
        parcela_id,
        -- Paso 3: volumen de riego uniforme entre 100.0 y 300.0 m3/ha
        ROUND(100.0 + u_riego * 200.0, 2) AS volumen_riego,
        -- Paso 4: fertilizante uniforme entre 50.0 y 150.0 kg/ha
        ROUND(50.0 + u_fertilizante * 100.0, 2) AS fertilizante_kg,
        -- Paso 5: asignacion aleatoria del grupo de prueba
        CASE WHEN u_tratamiento < 0.50 THEN '{ETIQUETA_A}'
             ELSE '{ETIQUETA_B}'
        END AS tipo_tratamiento,
        -- Variable ambiental del caso de uso, necesaria para el mapa de calor.
        -- NO entra en la ecuacion generadora del rendimiento (ver informe).
        ROUND(18.0 + u_temperatura * 14.0, 2) AS temperatura_c,
        -- Ruido aleatorio uniforme entre -0.8 y +0.8 ton/ha
        ROUND({RUIDO_MIN} + u_ruido * {RUIDO_MAX - RUIDO_MIN}, 4) AS ruido
    FROM aleatorios
)
SELECT
    parcela_id,
    volumen_riego,
    fertilizante_kg,
    tipo_tratamiento,
    temperatura_c,
    -- Paso 6: ecuacion fisico-matematica de la variable de respuesta
    ROUND(
        {BETA_0}
        + {BETA_RIEGO} * volumen_riego
        + {BETA_FERT} * fertilizante_kg
        + CASE WHEN tipo_tratamiento = '{ETIQUETA_B}' THEN {BONO_TRATAMIENTO_B} ELSE 0.0 END
        + ruido
    , 2) AS rendimiento_ton
FROM campos
ORDER BY parcela_id;
"""


def generar_telemetria() -> pd.DataFrame:
    """Fase 1. Simula las 100 parcelas en DuckDB y devuelve un DataFrame."""
    con = duckdb.connect(database=":memory:")
    con.execute(f"SELECT setseed({SEMILLA});")
    con.execute(SQL_TELEMETRIA)

    # Paso 1.3: limpieza y extraccion SQL previa al motor analitico de Python.
    # Se descartan registros incompletos o fisicamente imposibles antes de
    # construir el DataFrame.
    df = con.execute("""
        SELECT *
        FROM telemetria_parcelas
        WHERE volumen_riego   IS NOT NULL AND volumen_riego   > 0
          AND fertilizante_kg IS NOT NULL AND fertilizante_kg > 0
          AND rendimiento_ton IS NOT NULL AND rendimiento_ton > 0
        ORDER BY parcela_id;
    """).df()
    con.close()
    return df


# ===========================================================================
#  FASE 2 — CORRELACION Y MODELO OLS (SUBTEMA 4.2)
# ===========================================================================

COLUMNAS_NUMERICAS = [
    "volumen_riego",
    "fertilizante_kg",
    "temperatura_c",
    "rendimiento_ton",
]

NOMBRES_BONITOS = {
    "volumen_riego": "Riego",
    "fertilizante_kg": "Fertilizante",
    "temperatura_c": "Temperatura",
    "rendimiento_ton": "Rendimiento",
}


def matriz_correlacion(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 2.1. Matriz de correlacion de Pearson entre las variables numericas."""
    sub = df[COLUMNAS_NUMERICAS].rename(columns=NOMBRES_BONITOS)
    return sub.corr(method="pearson")


@dataclass
class ResultadoOLS:
    """Resultados del modelo de regresion lineal multiple."""
    modelo: sm.regression.linear_model.RegressionResultsWrapper
    r2: float
    r2_ajustado: float
    rmse: float
    beta_0: float
    beta_riego: float
    beta_fertilizante: float
    p_riego: float
    p_fertilizante: float
    residuos: np.ndarray
    predichos: np.ndarray
    shapiro_p: float          # Normalidad de los residuos (Shapiro-Wilk)
    durbin_watson: float      # Independencia de los residuos


def ajustar_ols(df: pd.DataFrame) -> ResultadoOLS:
    """Paso 2.2. Ajusta  Rendimiento = b0 + b1*Riego + b2*Fertilizante + e."""
    X = df[["volumen_riego", "fertilizante_kg"]]
    X = sm.add_constant(X)            # Agrega el intercepto b0
    y = df["rendimiento_ton"]

    modelo = sm.OLS(y, X).fit()
    predichos = modelo.fittedvalues.to_numpy()
    residuos = modelo.resid.to_numpy()
    rmse = float(np.sqrt(np.mean(residuos ** 2)))

    # Diagnostico de los supuestos del modelo
    shapiro_p = float(stats.shapiro(residuos).pvalue) if len(residuos) >= 3 else float("nan")
    dw = float(sm.stats.stattools.durbin_watson(residuos)) if len(residuos) >= 3 else float("nan")

    return ResultadoOLS(
        modelo=modelo,
        r2=float(modelo.rsquared),
        r2_ajustado=float(modelo.rsquared_adj),
        rmse=rmse,
        beta_0=float(modelo.params["const"]),
        beta_riego=float(modelo.params["volumen_riego"]),
        beta_fertilizante=float(modelo.params["fertilizante_kg"]),
        p_riego=float(modelo.pvalues["volumen_riego"]),
        p_fertilizante=float(modelo.pvalues["fertilizante_kg"]),
        residuos=residuos,
        predichos=predichos,
        shapiro_p=shapiro_p,
        durbin_watson=dw,
    )


def ajustar_ols_extendido(df: pd.DataFrame) -> ResultadoOLS | None:
    """
    Modelo de control que AGREGA la variable de tratamiento como variable
    ficticia (dummy):

        Rendimiento = b0 + b1*Riego + b2*Fertilizante + b3*(Tratamiento_B) + e

    No forma parte del modelo que pide el enunciado; se ajusta unicamente para
    diagnosticar por que los residuos del modelo base resultan bimodales.
    Devuelve None si el subconjunto no contiene los dos tratamientos.
    """
    if df["tipo_tratamiento"].nunique() < 2:
        return None

    X = df[["volumen_riego", "fertilizante_kg"]].copy()
    X["trat_B"] = (df["tipo_tratamiento"] == ETIQUETA_B).astype(float)
    X = sm.add_constant(X)
    y = df["rendimiento_ton"]

    modelo = sm.OLS(y, X).fit()
    predichos = modelo.fittedvalues.to_numpy()
    residuos = modelo.resid.to_numpy()
    rmse = float(np.sqrt(np.mean(residuos ** 2)))
    shapiro_p = float(stats.shapiro(residuos).pvalue) if len(residuos) >= 3 else float("nan")
    dw = float(sm.stats.stattools.durbin_watson(residuos)) if len(residuos) >= 3 else float("nan")

    return ResultadoOLS(
        modelo=modelo,
        r2=float(modelo.rsquared),
        r2_ajustado=float(modelo.rsquared_adj),
        rmse=rmse,
        beta_0=float(modelo.params["const"]),
        beta_riego=float(modelo.params["volumen_riego"]),
        beta_fertilizante=float(modelo.params["fertilizante_kg"]),
        p_riego=float(modelo.pvalues["volumen_riego"]),
        p_fertilizante=float(modelo.pvalues["fertilizante_kg"]),
        residuos=residuos,
        predichos=predichos,
        shapiro_p=shapiro_p,
        durbin_watson=dw,
    )


# ===========================================================================
#  FASE 3 — PRUEBA DE HIPOTESIS SOBRE LOS TRATAMIENTOS (SUBTEMA 4.3)
# ===========================================================================

@dataclass
class ResultadoPruebaT:
    """Resultados de la prueba t de dos muestras independientes."""
    valida: bool
    mensaje: str = ""
    n_a: int = 0
    n_b: int = 0
    media_a: float = float("nan")
    media_b: float = float("nan")
    diferencia: float = float("nan")
    t_stat: float = float("nan")
    p_unilateral: float = float("nan")
    ic_inferior: float = float("nan")
    ic_superior: float = float("nan")
    gl: float = float("nan")
    levene_p: float = float("nan")
    cohen_d: float = float("nan")
    rechaza_h0: bool = False


def prueba_t_tratamientos(df: pd.DataFrame) -> ResultadoPruebaT:
    """
    Paso 3.2-3.3.  H0: mu_B - mu_A <= 0   contra   H1: mu_B - mu_A > 0

    Prueba t de dos muestras independientes, unilateral por la derecha sobre
    el grupo B. Devuelve ademas el intervalo de confianza del 95% para la
    diferencia de medias.
    """
    a = df.loc[df["tipo_tratamiento"] == ETIQUETA_A, "rendimiento_ton"].to_numpy()
    b = df.loc[df["tipo_tratamiento"] == ETIQUETA_B, "rendimiento_ton"].to_numpy()

    if len(a) < 2 or len(b) < 2:
        return ResultadoPruebaT(
            valida=False,
            n_a=len(a),
            n_b=len(b),
            mensaje=(
                "La prueba t necesita al menos 2 parcelas de cada tratamiento. "
                "Ajusta los filtros para incluir ambos grupos."
            ),
        )

    # Prueba de Levene: decide si las varianzas pueden suponerse homogeneas
    _, levene_p = stats.levene(a, b)
    varianzas_iguales = levene_p > ALPHA

    t_stat, p_unilateral = stats.ttest_ind(
        b, a, equal_var=varianzas_iguales, alternative="greater"
    )

    diferencia = float(b.mean() - a.mean())

    # Intervalo de confianza del 95% (bilateral) para mu_B - mu_A
    n_a, n_b = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)

    if varianzas_iguales:
        gl = n_a + n_b - 2
        s_pool = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / gl)
        error_est = s_pool * np.sqrt(1 / n_a + 1 / n_b)
        d_cohen = diferencia / s_pool if s_pool else float("nan")
    else:
        # Grados de libertad de Welch-Satterthwaite
        error_est = np.sqrt(var_a / n_a + var_b / n_b)
        gl = (var_a / n_a + var_b / n_b) ** 2 / (
            (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        )
        s_prom = np.sqrt((var_a + var_b) / 2)
        d_cohen = diferencia / s_prom if s_prom else float("nan")

    t_critico = stats.t.ppf(1 - ALPHA / 2, df=gl)
    margen = t_critico * error_est

    return ResultadoPruebaT(
        valida=True,
        n_a=n_a,
        n_b=n_b,
        media_a=float(a.mean()),
        media_b=float(b.mean()),
        diferencia=diferencia,
        t_stat=float(t_stat),
        p_unilateral=float(p_unilateral),
        ic_inferior=float(diferencia - margen),
        ic_superior=float(diferencia + margen),
        gl=float(gl),
        levene_p=float(levene_p),
        cohen_d=float(d_cohen),
        rechaza_h0=bool(p_unilateral < ALPHA),
    )


# ===========================================================================
#  EJECUCION DIRECTA — validacion del motor analitico sin interfaz web
# ===========================================================================

if __name__ == "__main__":
    pd.set_option("display.width", 120)

    df = generar_telemetria()
    print("=" * 74)
    print(" FASE 1 — TELEMETRIA SIMULADA EN DUCKDB")
    print("=" * 74)
    print(df.head(10).to_string(index=False))
    print(f"\nRegistros: {len(df)}")
    print(f"Rendimiento: min={df['rendimiento_ton'].min():.2f} "
          f"max={df['rendimiento_ton'].max():.2f} ton/ha")
    print(df["tipo_tratamiento"].value_counts().to_string())

    print("\n" + "=" * 74)
    print(" FASE 2 — CORRELACION DE PEARSON Y MODELO OLS")
    print("=" * 74)
    print(matriz_correlacion(df).round(4).to_string())

    ols = ajustar_ols(df)
    print(f"\nR2 = {ols.r2:.4f} | R2 ajustado = {ols.r2_ajustado:.4f} "
          f"| RMSE = {ols.rmse:.4f} ton/ha")
    print(f"Ecuacion: Rendimiento = {ols.beta_0:.4f} "
          f"+ {ols.beta_riego:.5f}*Riego + {ols.beta_fertilizante:.5f}*Fertilizante")
    print(f"p(Riego) = {ols.p_riego:.4e} | p(Fertilizante) = {ols.p_fertilizante:.4e}")

    print("\n" + "=" * 74)
    print(" FASE 3 — PRUEBA t DE DOS MUESTRAS INDEPENDIENTES")
    print("=" * 74)
    t = prueba_t_tratamientos(df)
    print(f"n_A = {t.n_a} | n_B = {t.n_b}")
    print(f"Media A = {t.media_a:.4f} | Media B = {t.media_b:.4f} "
          f"| Diferencia = {t.diferencia:.4f} ton/ha")
    print(f"Levene p = {t.levene_p:.4f} "
          f"({'varianzas homogeneas' if t.levene_p > ALPHA else 'varianzas distintas'})")
    print(f"t = {t.t_stat:.4f} | gl = {t.gl:.2f} | p (unilateral) = {t.p_unilateral:.4e}")
    print(f"IC 95% para mu_B - mu_A: [{t.ic_inferior:.4f}, {t.ic_superior:.4f}] ton/ha")
    print(f"d de Cohen = {t.cohen_d:.4f}")
    print(f"Veredicto: {'RECHAZAR H0' if t.rechaza_h0 else 'NO RECHAZAR H0'}")
