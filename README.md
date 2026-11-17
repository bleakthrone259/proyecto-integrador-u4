# Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego

Proyecto Integrador — Unidad 4
**Análisis y Exploración de Datos (INC-2505)**
Instituto Tecnológico de Cuautla · TecNM

**Alumno:** Torres Hernández Dorian Idari — No. de control **23680452**
**Docente:** Arístides Caballero Alfaro

---

## Qué hace

Tablero analítico sobre telemetría de invernaderos inteligentes de jitomate. A partir de
100 parcelas simuladas de forma determinista en DuckDB, responde dos preguntas de
ingeniería agroindustrial:

1. **¿Qué factor agronómico predice mejor la cosecha?** → correlación de Pearson y
   regresión lineal múltiple (OLS).
2. **¿El nuevo bio-estimulante foliar incrementa el rendimiento?** → prueba *t* de dos
   muestras independientes con intervalo de confianza al 95 %.

## Resultados principales

| Indicador | Valor |
|---|---|
| Correlación riego ↔ rendimiento | r = 0.8756 |
| Correlación fertilizante ↔ rendimiento | r = 0.3300 |
| R² ajustado del modelo | 80.43 % |
| RMSE | 1.0589 ton/ha |
| Diferencia de rendimiento (B − A) | 1.988 ton/ha |
| Valor p (unilateral) | 1.08 × 10⁻⁵ |
| IC 95 % de la diferencia | [1.104, 2.873] ton/ha |
| **Veredicto** | **Se rechaza H₀ — se autoriza escalar el Tratamiento B** |

## Estructura

```
.
├── app.py                 # Interfaz Streamlit: KPIs, cuadrícula 2x2, filtros
├── analisis.py            # Motor analítico: DuckDB, OLS, prueba t, diagnóstico
├── graficas.py            # Las 4 gráficas clave (matplotlib + seaborn)
├── requirements.txt       # Dependencias
├── README.md
└── .streamlit/
    └── config.toml        # Tema visual
```

El motor analítico está desacoplado de la interfaz, de modo que puede ejecutarse y
validarse sin levantar la app:

```bash
python analisis.py
```

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

La app queda en `http://localhost:8501`.

## Desplegar en Streamlit Community Cloud

1. Subir este repositorio a GitHub como **público**.
2. Entrar a <https://share.streamlit.io> e iniciar sesión con GitHub.
3. **Create app** → seleccionar el repositorio, rama `main`, archivo principal `app.py`.
4. Personalizar la URL y pulsar **Deploy**.

El primer despliegue tarda unos minutos porque instala SciPy y statsmodels. Cada `push`
a `main` redespliega automáticamente.

## Nota de reproducibilidad

`setseed(0.42)` fija la secuencia de `random()` dentro de una misma versión del motor,
pero el generador pseudoaleatorio de DuckDB ha cambiado entre versiones. Las cifras de
este README corresponden a **DuckDB 2.0.0**. Los veredictos inferenciales no dependen de
ese detalle: ambos valores p son varios órdenes de magnitud menores que α = 0.05.
