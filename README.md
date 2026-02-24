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
  - **Soporte Técnico**: La ausencia de servicios de seguridad y soporte eleva la tasa de churn por encima del **30%**.

---

### 4 Preprocesamiento de Datos (Data Preprocessing)

Preparación técnica de los datos para su uso en modelos de Machine Learning.

- **Codificación de Variables Categóricas**:
  - One-Hot Encoding para variables nominales (ej. `PaymentMethod`).
  - Label Encoding para la variable objetivo (`Churn`: 0 = No, 1 = Sí).

- **Escalado de Variables Numéricas**:
  - Estandarización (StandardScaler) o Normalización (MinMaxScaler).
  - Aplicado a variables como:
    - `Tenure`
    - `ChargesMonthly`
    - `num_services`

- **Pipeline de Preprocesamiento**:
  Implementación de `Pipeline` para evitar data leakage y garantizar reproducibilidad.

---

### 5 División del Dataset

Separación estratégica para validar el desempeño real del modelo.

- **Definición de variables**:
  - `X` → Variables predictoras.
  - `y` → Variable objetivo (Churn).

- **Train/Test Split**:
  - 80% entrenamiento
  - 20% prueba
  - Uso de `stratify=y` para mantener proporción de churn.

---

### 6 Validación Cruzada (Cross-Validation)

Antes de evaluar el modelo final, se implementa validación cruzada:

- **K-Fold Cross Validation (k=5 o 10)**
  Permite:
  - Reducir varianza en la estimación del rendimiento.
  - Detectar sobreajuste.
  - Obtener métricas promedio más robustas.

Esta etapa es clave para asegurar estabilidad del modelo, se aplica sólo alconjunto de entrenamiento.

---

### 7 Selección y Entrenamiento de Modelos

Se exploran diferentes algoritmos para comparar desempeño:

- **Regresión Logística**
  - Modelo base interpretable.
  - Buena referencia inicial.

- **Random Forest**
  - Maneja relaciones no lineales.
  - Robusto ante outliers y ruido.

- **Gradient Boosting (ej. XGBoost o similares)**
  - Alto poder predictivo.
  - Captura patrones complejos.

- **LightGBM**
  - Más rápido y eficiente que otros boosting

  - Excelente rendimiento en datasets estructurados

  - Maneja bien variables categóricas codificadas

Cada modelo se entrena utilizando validación cruzada para comparación objetiva.

---

### 8 Evaluación de Métricas

Dado que el problema es de clasificación binaria con posible desbalance, se priorizan las siguientes métricas:

- **Matriz de Confusión**
  - Falsos Positivos (FP)
  - Falsos Negativos (FN)

- **Recall (Sensibilidad)**
  - Métrica crítica en churn: identificar correctamente a quienes se irán.

- **Precision**
  - Minimizar falsas alarmas.

- **F1-Score**
  - Balance entre Precision y Recall.

- **ROC-AUC**
  - Evaluación global del poder discriminatorio del modelo.

* En `variables` como churn, **Recall** tiene mayor relevancia en resultados.

---

### 9 Optimización de Hiperparámetros

Para maximizar el rendimiento:

- **GridSearchCV**
- **RandomizedSearchCV**

Aplicado sobre el modelo con mejor desempeño preliminar.

Objetivo:

- Ajustar profundidad de árboles.
- Número de estimadores.
- Regularización.
- Learning rate (en boosting).

Todo validado mediante Cross-Validation.

---

### 10 Modelo Final y Análisis de Importancia

- Entrenamiento final con mejores hiperparámetros.
- Evaluación sobre conjunto de prueba.
- Análisis de importancia de variables.
- Interpretación estratégica de resultados.

---

## Resultados Esperados

- Modelo robusto y validado.
- Identificación de variables clave en churn.
- Segmentación de clientes en riesgo.
- Base técnica para implementar estrategias de retención.

---

## Impacto Estratégico

Este modelo predictivo permite:

- Implementar campañas de retención dirigidas.
- Optimizar recursos comerciales.
- Reducir tasa de churn.
- Incrementar el Customer Lifetime Value (CLV).
