"""
=============================================================================
 PROYECTO INTEGRADOR - UNIDAD 4
 Capa de visualizacion: las 4 graficas clave del dashboard (Unidad 6)
=============================================================================
 Alumno: Torres Hernandez Dorian Idari  -  No. control: 23680452

 Grafica 1: Mapa de calor de correlacion de Pearson   (divergente)
 Grafica 2: Dispersion con ajuste OLS y banda de confianza
 Grafica 3: Diagrama de cajas comparativo por tratamiento
 Grafica 4: Histograma y densidad de residuos del modelo
=============================================================================
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from analisis import ETIQUETA_A, ETIQUETA_B, NOMBRES_BONITOS

# ---------------------------------------------------------------------------
# Paleta del tablero. Categoricos asignados en orden fijo, nunca ciclados.
# ---------------------------------------------------------------------------
SERIE_1 = "#2a78d6"   # azul   -> Tratamiento_A / serie unica
SERIE_2 = "#eb6834"   # naranja-> Tratamiento_B
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUTADO = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
NEUTRO = "#f0efec"
ROJO_DIV = "#d03b3b"

COLOR_TRATAMIENTO = {ETIQUETA_A: SERIE_1, ETIQUETA_B: SERIE_2}

# Rampa divergente azul <-> rojo con punto medio gris neutro.
# La correlacion es polaridad (-1 a +1), por eso divergente y no secuencial.
# Azul = asociacion positiva, rojo = negativa, gris neutro en r = 0.
CMAP_CORRELACION = LinearSegmentedColormap.from_list(
    "corr_div", [ROJO_DIV, NEUTRO, SERIE_1], N=256
)


def _aplicar_estilo(ax, titulo: str = "", xlabel: str = "", ylabel: str = ""):
    """Cromo recesivo: rejilla fina, sin marcos, tinta apagada en los ejes."""
    ax.set_facecolor(SUPERFICIE)
    ax.figure.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EJE)
        ax.spines[lado].set_linewidth(1.0)
    ax.grid(True, color=REJILLA, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTADO, labelsize=9, length=0)
    if titulo:
        ax.set_title(titulo, color=TINTA, fontsize=12, fontweight="600",
                     pad=12, loc="left")
    ax.set_xlabel(xlabel, color=TINTA_2, fontsize=10)
    ax.set_ylabel(ylabel, color=TINTA_2, fontsize=10)


def _figura(ancho=6.2, alto=4.3):
    fig, ax = plt.subplots(figsize=(ancho, alto), dpi=130)
    fig.patch.set_facecolor(SUPERFICIE)
    return fig, ax


# ===========================================================================
#  GRAFICA 1 — MAPA DE CALOR DE CORRELACION DE PEARSON
# ===========================================================================

def grafica_heatmap(matriz: pd.DataFrame):
    fig, ax = _figura(6.2, 4.6)
    sns.heatmap(
        matriz,
        annot=True, fmt=".3f",
        cmap=CMAP_CORRELACION, vmin=-1, vmax=1, center=0,
        linewidths=2, linecolor=SUPERFICIE,   # separador de 2px entre celdas
        square=True, cbar_kws={"shrink": 0.78, "label": "r de Pearson"},
        annot_kws={"fontsize": 10, "fontweight": "600"},
        ax=ax,
    )
    _aplicar_estilo(ax, "Mapa de calor de correlación de Pearson")
    ax.grid(False)
    ax.tick_params(labelsize=9.5)
    plt.setp(ax.get_yticklabels(), rotation=0, color=TINTA_2)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", color=TINTA_2)
    barra = ax.collections[0].colorbar
    barra.ax.tick_params(colors=MUTADO, labelsize=8, length=0)
    barra.outline.set_visible(False)
    barra.set_label("r de Pearson", color=TINTA_2, fontsize=9)
    fig.tight_layout()
    return fig


# ===========================================================================
#  GRAFICA 2 — DISPERSION CON AJUSTE OLS Y BANDA DE CONFIANZA
# ===========================================================================

def grafica_dispersion_ols(df: pd.DataFrame):
    fig, ax = _figura()

    for etiqueta in (ETIQUETA_A, ETIQUETA_B):
        sub = df[df["tipo_tratamiento"] == etiqueta]
        if sub.empty:
            continue
        ax.scatter(
            sub["volumen_riego"], sub["rendimiento_ton"],
            s=46, color=COLOR_TRATAMIENTO[etiqueta],
            edgecolor=SUPERFICIE, linewidth=1.6,   # anillo de 2px al solaparse
            alpha=0.92, label=etiqueta.replace("_", " "), zorder=3,
        )

    # Recta de minimos cuadrados sobre el riego, con banda de confianza al 95%
    if len(df) >= 3:
        x = df["volumen_riego"].to_numpy()
        y = df["rendimiento_ton"].to_numpy()
        modelo = sm.OLS(y, sm.add_constant(x)).fit()

        malla = np.linspace(x.min(), x.max(), 160)
        pred = modelo.get_prediction(sm.add_constant(malla))
        resumen = pred.summary_frame(alpha=0.05)

        ax.fill_between(
            malla, resumen["mean_ci_lower"], resumen["mean_ci_upper"],
            color=TINTA, alpha=0.10, zorder=1, label="Banda de confianza 95%",
        )
        ax.plot(malla, resumen["mean"], color=TINTA, linewidth=2.0,
                zorder=2, label="Ajuste OLS")

        ax.annotate(
            f"$R^2$ = {modelo.rsquared:.3f}",
            xy=(0.98, 0.06), xycoords="axes fraction", ha="right",
            fontsize=11, color=TINTA, fontweight="600",
        )

    _aplicar_estilo(
        ax,
        "Rendimiento frente a volumen de riego",
        "Volumen de riego (m³/ha)",
        "Rendimiento (ton/ha)",
    )
    leyenda = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for texto in leyenda.get_texts():
        texto.set_color(TINTA_2)
    fig.tight_layout()
    return fig


# ===========================================================================
#  GRAFICA 3 — DIAGRAMA DE CAJAS COMPARATIVO POR TRATAMIENTO
# ===========================================================================

def grafica_boxplot(df: pd.DataFrame):
    fig, ax = _figura()

    presentes = [e for e in (ETIQUETA_A, ETIQUETA_B)
                 if (df["tipo_tratamiento"] == e).any()]
    datos = [df.loc[df["tipo_tratamiento"] == e, "rendimiento_ton"].to_numpy()
             for e in presentes]

    if datos:
        cajas = ax.boxplot(
            datos, patch_artist=True, widths=0.45,
            tick_labels=[e.replace("_", " ") for e in presentes],
            medianprops=dict(color=TINTA, linewidth=2.0),
            whiskerprops=dict(color=EJE, linewidth=1.4),
            capprops=dict(color=EJE, linewidth=1.4),
            flierprops=dict(marker="o", markersize=5, markerfacecolor=MUTADO,
                            markeredgecolor=SUPERFICIE, alpha=0.8),
        )
        for caja, etiqueta in zip(cajas["boxes"], presentes):
            caja.set_facecolor(COLOR_TRATAMIENTO[etiqueta])
            caja.set_alpha(0.28)
            caja.set_edgecolor(COLOR_TRATAMIENTO[etiqueta])
            caja.set_linewidth(2.0)

        # La media de cada grupo, que es lo que compara la prueba t
        for i, (serie, etiqueta) in enumerate(zip(datos, presentes), start=1):
            media = float(np.mean(serie))
            ax.scatter([i], [media], s=70, marker="D",
                       color=COLOR_TRATAMIENTO[etiqueta],
                       edgecolor=SUPERFICIE, linewidth=1.6, zorder=4)
            ax.annotate(f"media {media:.2f}", xy=(i, media),
                        xytext=(12, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TINTA_2)

    _aplicar_estilo(
        ax,
        "Rendimiento por tipo de tratamiento",
        "",
        "Rendimiento (ton/ha)",
    )
    plt.setp(ax.get_xticklabels(), color=TINTA_2, fontsize=10)
    fig.tight_layout()
    return fig


# ===========================================================================
#  GRAFICA 4 — HISTOGRAMA Y DENSIDAD DE RESIDUOS
# ===========================================================================

def grafica_residuos(residuos: np.ndarray):
    fig, ax = _figura()

    if len(residuos) >= 2:
        ax.hist(
            residuos, bins=18, density=True,
            color=SERIE_1, alpha=0.30,
            edgecolor=SUPERFICIE, linewidth=1.5, zorder=2,
        )
        if len(residuos) >= 5 and np.std(residuos) > 0:
            kde = stats.gaussian_kde(residuos)
            malla = np.linspace(residuos.min(), residuos.max(), 220)
            ax.plot(malla, kde(malla), color=SERIE_1, linewidth=2.0,
                    zorder=3, label="Densidad observada (KDE)")

            # Curva normal teorica N(0, sigma^2) como referencia del supuesto
            normal = stats.norm.pdf(malla, 0.0, float(np.std(residuos, ddof=1)))
            ax.plot(malla, normal, color=TINTA, linewidth=1.8,
                    linestyle="--", zorder=4, label="Normal teórica N(0, σ²)")

        ax.axvline(0.0, color=EJE, linewidth=1.4, zorder=1)

    _aplicar_estilo(
        ax,
        "Distribución de los residuos del modelo",
        "Residuo = Y real − Y predicho (ton/ha)",
        "Densidad",
    )
    manejadores, _ = ax.get_legend_handles_labels()
    if manejadores:
        leyenda = ax.legend(frameon=False, fontsize=9, loc="upper right")
        for texto in leyenda.get_texts():
            texto.set_color(TINTA_2)
    fig.tight_layout()
    return fig
