# Modelo de frontera eficiente para clientes de tarjetas de crédito

Este proyecto contiene una aplicación en **Python + Streamlit** para optimizar la mezcla de productos/clientes de tarjeta de crédito mediante un modelo de **frontera eficiente tipo Markowitz**.

El objetivo es estimar qué combinación de grupos de clientes genera la mayor rentabilidad agregada ajustada por riesgo, usando variables comerciales, financieras y de riesgo crediticio de una base de datos de TDC.

## Estructura del proyecto

```text
streamlit_markowitz_tdc/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── Base_Datos_TDC_Ejemplo.xlsx
```

## Archivo Excel esperado

La aplicación espera un archivo Excel con, al menos, estas hojas:

- `Clientes`
- `Historico_Mensual`

Opcionalmente puede incluir:

- `Rendimientos_Mensuales`
- `Diccionario_Datos`

## Variables utilizadas

### Hoja `Clientes`

- `Cliente_ID`
- `Segmento`
- `Edad_años`
- `Ingreso_Mensual_MXN`
- `Antigüedad_meses`
- `Productos_Banco`
- `Estatus`
- `Límite_Crédito_MXN`
- `Saldo_Promedio_MXN`
- `Utilización_Línea_pct`
- `Tasa_Interés_Anual_pct`
- `Score_Crediticio`
- `PD_12m_pct`
- `LGD_pct`
- `Cliente_Revolvente`
- `Ingreso_Intereses_Anual_MXN`
- `Costo_Fondeo_Anual_MXN`
- `Pérdida_Esperada_Anual_MXN`
- `Margen_Neto_Anual_MXN`
- `Rentabilidad_Estimada_Anual_pct`

### Hoja `Historico_Mensual`

- `Fecha`
- `Cliente_ID`
- `Segmento`
- `Límite_Crédito_MXN`
- `Saldo_Promedio_MXN`
- `Utilización_Línea_pct`
- `Compras_Mes_MXN`
- `Pago_Mes_MXN`
- `Mora_Días`
- `Ingreso_Intereses_Mes_MXN`
- `Comisiones_Mes_MXN`
- `Costo_Fondeo_Mes_MXN`
- `Pérdida_Crédito_Mes_MXN`
- `Margen_Neto_Mes_MXN`
- `Rendimiento_Mensual_pct`

## Lógica del modelo

La aplicación convierte los clientes en “activos” optimizables. Por defecto, cada activo corresponde a la combinación:

```text
Segmento + Productos_Banco
```

Ejemplos:

```text
Premium | 3
Oro | 2
Clásica | 1
Básica | 4
```

Para cada grupo, el modelo calcula:

- rendimiento mensual ponderado por saldo promedio;
- rendimiento esperado anualizado;
- matriz de covarianzas anualizada;
- volatilidad anual;
- penalización por riesgo de crédito usando `PD_12m × LGD`;
- pesos óptimos de la mezcla.

## Objetivos disponibles

La aplicación permite elegir entre tres objetivos:

1. **Máximo Sharpe ajustado por riesgo**  
   Maximiza la relación entre rentabilidad esperada ajustada y volatilidad.

2. **Máxima rentabilidad con límite de riesgo**  
   Maximiza la rentabilidad esperada sin superar una volatilidad anual máxima.

3. **Mínima volatilidad**  
   Encuentra la mezcla menos riesgosa bajo las restricciones configuradas.

## Restricciones incluidas

- Los pesos deben sumar 100%.
- No se permiten posiciones cortas.
- Se puede definir un peso máximo por grupo.
- Se puede filtrar solo clientes activos.
- Se puede exigir un mínimo de clientes y meses por grupo.

## Instalación local

1. Clonar o descargar el repositorio.
2. Crear un entorno virtual.
3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecutar la aplicación:

```bash
streamlit run app.py
```

5. Abrir la URL local que indique Streamlit.

## Cómo subir a GitHub

Desde la carpeta del proyecto:

```bash
git init
git add .
git commit -m "Modelo Streamlit frontera eficiente TDC"
git branch -M main
git remote add origin https://github.com/USUARIO/NOMBRE_REPOSITORIO.git
git push -u origin main
```

Reemplazar `USUARIO` y `NOMBRE_REPOSITORIO` por los datos de tu cuenta y repositorio.

## Cómo correrlo en Streamlit Community Cloud

1. Subir el proyecto completo a GitHub.
2. Entrar a Streamlit Community Cloud.
3. Crear una nueva app.
4. Seleccionar el repositorio de GitHub.
5. Indicar como archivo principal:

```text
app.py
```

6. Desplegar la aplicación.

## Salidas del dashboard

La aplicación genera:

- métricas de rentabilidad esperada, riesgo anual y Sharpe;
- gráfica de frontera eficiente;
- tabla de mezcla óptima por grupo;
- gráfica de pesos óptimos;
- diagnóstico del universo optimizable;
- matriz de correlación;
- archivos CSV descargables con resultados.

## Nota metodológica

Este modelo usa una adaptación de Markowitz para una cartera de grupos de clientes de tarjeta de crédito. En lugar de acciones o instrumentos financieros, los “activos” son segmentos comerciales o combinaciones de productos bancarios. El rendimiento se toma de la variable `Rendimiento_Mensual_pct` y se pondera por `Saldo_Promedio_MXN`.

La rentabilidad se ajusta por riesgo de crédito mediante una penalización basada en `PD_12m_pct × LGD_pct`, lo que permite favorecer mezclas con mejor rentabilidad ajustada por pérdida esperada.

## Advertencia

El modelo es analítico y educativo. No sustituye políticas internas de riesgo, capital regulatorio, apetito de riesgo, normativa aplicable ni validaciones independientes del área de riesgos.
