## Challenge TelecomX-2-ML – Análisis Predictivo de Churn

### Objetivo

En esta segunda etapa del Challenge de Data Science de Alura, el enfoque se centra en el **modelado predictivo** para anticipar la evasión de clientes (Churn) en TelecomX.
Luego de completar el proceso de **ETL y Análisis Exploratorio (EDA)**, el objetivo ahora es construir, evaluar y optimizar modelos de Machine Learning que permitan:

- Predecir qué clientes tienen mayor probabilidad de abandonar el servicio.
- Identificar las variables más influyentes en la decisión de churn.
- Proporcionar recomendaciones estratégicas basadas en evidencia predictiva.

---

## Implementación de Modelado Predictivo

Esta etapa se enfoca en preparar el dataset para el entrenamiento de modelos, garantizando la calidad de los datos y creando variables con alto poder predictivo.

### 1. Limpieza de Datos

Se realizó un proceso exhaustivo para asegurar la consistencia del dataset, obteniendo los siguientes resultados clave:

- **Diagnóstico Inicial**: 7,043 registros y 22 columnas.
- **Estandarización de Categorías**: Se homologaron valores en servicios adicionales (ej. "No internet service" → "No") para reducir ruido.
- **Eliminación de Identificadores**: Se removió la columna `CustomerID` por carecer de valor predictivo.
- **Gestión de Duplicados**: Tras eliminar el ID, se detectaron y eliminaron **22 registros duplicados**, garantizando la integridad de las muestras.
- **Estado Final**: **7,021 registros** y **21 columnas** listos para el análisis.

---

### 2. Ingeniería de Características (Feature Engineering)

En esta fase se optimizaron las variables existentes y se crearon nuevos indicadores:

- **Análisis de Correlación**:
  - Se identificó una redundancia crítica entre `ChargesDaily` y `ChargesMonthly` (**99.99%** de correlación).
  - Se detectó que `Tenure` es la variable con mayor correlación individual con el **Churn (35.15%)**.
- **Eliminación de Redundancias**: Se eliminó la variable `ChargesDaily` para evitar multicolinealidad.
- **Creación de `num_services`**: Nueva variable numérica que cuantifica la cantidad de servicios activos contratados (rango de 1 a 9). Esta variable captura la intensidad de la relación comercial con el cliente.

---

### 3. Análisis Categórico y Test Chi²

Se realizó un análisis de proporciones y se aplicó el test estadístico **Chi-cuadrado (Chi²)** para evaluar la asociación entre las variables categóricas y el Churn.

- **Ranking de Relevancia Estadística**:
  - **Variables más determinantes**: `Contract` (Chi² = 1108.89), `OnlineSecurity`, `SeniorCitizen`, `TechSupport` y `Dependents`. Todas con p-value < 0.05.
  - **Variables irrelevantes**: `Gender` y `PhoneService` no mostraron asociación significativa con el abandono.
- **Hallazgos Clave**:
  - **Contratos**: Los clientes con contrato mensual (_Month-to-month_) presentan un riesgo de fuga de **42.64%**, frente al **2.83%** de los contratos a dos años.
  - **Servicio de Internet**: La fibra óptica (_Fiber optic_) registra un churn del **41.78%**, significativamente superior al DSL (**18.89%**).
  - **Método de Pago**: El cheque electrónico (_Electronic check_) es el método con mayor deserción (**45.15%**).

---

### 4. Análisis Visual Integrado

Investigación de patrones clave antes del modelado:

- **Tenure, Contract y ChargesTotal vs Churn**: Análisis conjunto (boxplots, gráficos de barras y scatter plots) para detectar cómo el tiempo de contrato y el gasto acumulado impactan los segmentos con mayor riesgo de cancelación.

---

### 5. Preprocesamiento de Datos y División del Dataset

Preparación técnica de los datos para su uso en Machine Learning, incluyendo la separación estratégica de muestras:

- **Codificación de la Variable Objetivo (`Churn`)**:
  - Transformación manual mediante mapeo directo: `No` → 0, `Yes` → 1.
  - Garantiza que el valor 1 represente el evento de interés (abandono).

- **División del Dataset (Estratificada)**:
  - Se realizó una separación con `stratify=y` para asegurar la representatividad del Churn en todos los cortes:
  - **Train (~70%)**: Entrenamiento del modelo.
  - **Validation (~15%)**: Ajuste de hiperparámetros y selección del mejor modelo.
  - **Test (~15%)**: Aislamiento estricto para la evaluación final.

- **Preprocesamiento de Variables Predictoras (X)**:
  - **Variables Categóricas**: `OneHotEncoder` con `drop='first'` y `handle_unknown='ignore'`.
  - **Variables Numéricas**: `StandardScaler` aplicado a `Tenure`, `ChargesMonthly`, `ChargesTotal` y `num_services`.

- **Pipeline de Preprocesamiento**:
  - Implementación mediante `ColumnTransformer` y `Pipeline` para encapsular transformaciones.
  - **Aislamiento de Datos**: El preprocesador se ajusta (`fit`) **exclusivamente con el conjunto de entrenamiento**, eliminando riesgos de _data leakage_.

---

### 6. Implementación de Modelos: Modelo Base

Se estableció un punto de partida (_benchmark_) utilizando un pipeline completo:

- **Algoritmo**: Regresión Logística (max_iter=1000)
- **Resultados en Validation**:
  - **Accuracy (Exactitud)**: ~80.3%
  - **Recall (Sensibilidad para 'Yes')**: ~55%
  - **F1-Score (Clase 'Yes')**: ~60%
- _Diagnóstico_: El modelo previene adecuadamente a los clientes retenidos, pero requiere algoritmos más avanzados o afinación para capturar la mayor cantidad posible de fugas.

---

### 7. Próximos Modelos y Validación Cruzada

Se probarán algoritmos avanzados soportados por **K-Fold Cross Validation**:

- **Random Forest**: Manejo de relaciones no lineales.
- **Gradient Boosting (XGBoost / LightGBM)**: Alto poder predictivo para patrones complejos y estructurados.

---

### 8. Evaluación de Métricas

Se priorizan las siguientes métricas dado el posible desbalance:

- **Matriz de Confusión**: FP y FN.
- **Recall (Sensibilidad)**: Métrica crítica en churn.
- **ROC-AUC**: Evaluación global del poder discriminatorio.

---

### 9. Optimización de Hiperparámetros

- **GridSearchCV / RandomizedSearchCV** sobre el modelo con mejor desempeño.
- Ajuste de parámetros clave (profundidad, estimadores, learning rate).

---

### 10. Modelo Final y Análisis de Importancia

- Entrenamiento final con mejores hiperparámetros.
- Evaluación sobre conjunto de prueba aislado.
- Análisis de importancia de variables e interpretación estratégica.

---

## Resultados Esperados

- Modelo robusto y validado.
- Identificación de variables clave en churn.
- Base técnica para estrategias de retención.

---

## Impacto Estratégico

- Implementar campañas de retención dirigidas.
- Optimizar recursos comerciales y reducir tasa de churn.
- Incrementar el Customer Lifetime Value (CLV).
