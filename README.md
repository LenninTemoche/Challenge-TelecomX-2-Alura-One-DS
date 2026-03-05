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

### 6. Implementación de Modelos Base (Etapa 1)

Se evaluaron cinco algoritmos en su configuración inicial (sin balanceo de clases) para establecer un _benchmark_ de rendimiento utilizando un `Pipeline` integral.

#### Resultados y Ranking (Validation Set)

| Modelo                  |  F1-Score  |   Recall   | Accuracy | Diagnóstico           |
| :---------------------- | :--------: | :--------: | :------: | :-------------------- |
| **Logistic Regression** | **0.5965** | **0.5504** |  80.30%  | 🟢 Bien generalizado  |
| **LightGBM**            |   0.5737   |   0.5180   |  75.64%  | 🟡 Leve sobreajuste   |
| **XGBoost**             |   0.5668   |   0.5036   |  75.83%  | 🟡 Leve sobreajuste   |
| **MLP**                 |   0.5424   |   0.4712   |  74.69%  | 🟢 Bien generalizado  |
| **Random Forest**       |   0.5342   |   0.4640   |  78.59%  | 🔴 Sobreajuste severo |

### 7. Implementación de Modelos Balanceados (Etapa 2)

Dado el desbalance de clases, se aplicaron técnicas de balanceo para priorizar el **Recall** (detección de abandonos) y corregir el sobreajuste.

- **Técnicas aplicadas:** `class_weight='balanced'` (LR, RF, LGBM), `scale_pos_weight` (XGBoost) y **SMOTE** (MLP).
- **Ajustes:** Se limitó la profundidad en modelos de árboles (`max_depth=8`) y se aumentó la regularización (`C=0.1`) para mejorar la generalización.

#### Comparativa de Modelos Optimizados (Validation Set)

| Modelo                  |   Recall   |  F1-Score  |     Diferencia vs Base     |
| :---------------------- | :--------: | :--------: | :------------------------: |
| **Logistic Regression** | **0.7842** |   0.6158   | 🟢 +42% de Recall mejorado |
| **MLP (SMOTE)**         |   0.7590   |   0.6134   | 🟢 +61% de Recall mejorado |
| **Random Forest**       |   0.7518   | **0.6353** | 🟢 +62% de Recall mejorado |
| **XGBoost**             |   0.7410   |   0.6186   | 🟢 +47% de Recall mejorado |
| **LightGBM**            |   0.7302   |   0.6133   | 🟢 +41% de Recall mejorado |

#### Validación Cruzada (Robustez)

Para asegurar que el rendimiento no dependa de una partición aleatoria, se aplicó **StratifiedKFold (5 folds)**:

- **Regresión Logística:** Obtuvo un **CV Recall de 0.7985 (±0.03)**, confirmando su estabilidad como el mejor sensor de abandono.
- **ROC-AUC Global:** Todos los modelos superaron el **0.84**, demostrando un alto poder de separación de clases.
- **Estabilidad:** La baja desviación estándar (±0.01) en las métricas garantiza una alta fiabilidad para producción.

### 8. Evaluación de Métricas y Diagnóstico Final

La optimización permitió alcanzar un equilibrio superior entre precisión y sensibilidad:

- **Identificación Proactiva:** Se logran capturar casi 8 de cada 10 posibles casos de churn.
- **Generalización Ganada:** El _gap_ entre entrenamiento y validación se redujo sustancialmente en Random Forest (de 21% a 2% de diferencia en Accuracy).

| Métrica Crítica              | Modelo Ganador      |   Valor    |
| :--------------------------- | :------------------ | :--------: |
| **Mejor Recall (Detección)** | Logistic Regression |   0.7842   |
| **Mejor F1-Score (Balance)** | Random Forest       |   0.6353   |
| **Mejor Consistencia (CV)**  | LightGBM            | F1: 0.6276 |

### 9. Recomendación Estratégica de Negocio

Se sugieren dos caminos de acción basados en el perfil del modelo:

1.  **Estrategia de Retención Masiva (Prioridad: No perder clientes):**
    - Utilizar **Regresión Logística (Balanced)**. Es el modelo más sensible para detectar el churn (**82.1% Recall** en Test Set), ideal para campañas preventivas amplias.
2.  **Estrategia de Eficiencia Operativa (Prioridad: Costo-Beneficio):**
    - Utilizar **Random Forest (Balanced)**. Ofrece el mejor balance global (**0.635 F1-Score**) reduciendo la cantidad de clientes contactados innecesariamente (menos falsos positivos).

### 10. Evaluación en el Conjunto de Prueba (Test Set)

Para validar la capacidad de generalización del modelo elegido (**Regresión Logística**), se evaluó en el conjunto de datos de prueba (`Test Set`), el cual fue aislado desde el inicio del proyecto.

| Métrica       |   Valor   | Interpretación                                                |
| :------------ | :-------: | :------------------------------------------------------------ |
| **Accuracy**  |   76.0%   | Proporción global de predicciones correctas.                  |
| **Recall**    | **82.1%** | **Capacidad de detectar 8 de cada 10 fugas reales.**          |
| **Precision** |   53.0%   | Precisión en las alarmas de riesgo generadas.                 |
| **F1-Score**  |   64.4%   | Balance armónico entre precisión y detección.                 |
| **ROC-AUC**   | **85.1%** | Excelente capacidad de separar clientes leales de desertores. |

---

## Impacto Estratégico

El uso de estos modelos permite a TelecomX optimizar sus recursos al dirigir sus esfuerzos de fidelización específicamente a los segmentos de alto riesgo, incrementando el **Customer Lifetime Value (CLV)** y reduciendo la tasa de deserción mediante intervenciones oportunas.

---

_Este proyecto sigue un flujo de MLOps utilizando Pipelines para asegurar la reproducibilidad y facilidad de despliegue._
