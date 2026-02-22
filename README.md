## Challenge TelecomX-2-ML – Análisis Predictivo de Churn

### Objetivo

En esta segunda etapa del Challenge de Data Science de Alura, el enfoque se centra en el **modelado predictivo** para anticipar la evasión de clientes (Churn) en TelecomX.
Luego de completar el proceso de **ETL y Análisis Exploratorio (EDA)**, el objetivo ahora es construir, evaluar y optimizar modelos de Machine Learning que permitan:

* Predecir qué clientes tienen mayor probabilidad de abandonar el servicio.
* Identificar las variables más influyentes en la decisión de churn.
* Proporcionar recomendaciones estratégicas basadas en evidencia predictiva.

---

## Implementación de Modelado Predictivo

### 1 Ingeniería de Características (Feature Engineering)

En esta fase se optimizan y crean variables para mejorar la capacidad predictiva del modelo.

* **Creación de `num_services`**: Variable que cuantifica la intensidad de uso del cliente. Un mayor número de servicios puede reflejar mayor fidelización o mayor complejidad de abandono.
* **Variables derivadas**: Posible creación de ratios como gasto promedio por servicio.
* **Eliminación de variables irrelevantes o redundantes**:

  * `CustomerID` (identificador sin valor predictivo).
  * Análisis de variables altamente correlacionadas que puedan generar multicolinealidad.
* **Análisis de importancia preliminar** para validar aporte de nuevas variables.

---

### 2 Preprocesamiento de Datos (Data Preprocessing)

Preparación técnica de los datos para su uso en modelos de Machine Learning.

* **Codificación de Variables Categóricas**:

  * One-Hot Encoding para variables nominales (ej. `PaymentMethod`).
  * Label Encoding para la variable objetivo (`Churn`: 0 = No, 1 = Sí).

* **Escalado de Variables Numéricas**:

  * Estandarización (StandardScaler) o Normalización (MinMaxScaler).
  * Aplicado a variables como:

    * `Tenure`
    * `ChargesMonthly`
    * `num_services`

* **Pipeline de Preprocesamiento**:
  Implementación de `Pipeline` para evitar data leakage y garantizar reproducibilidad.

---

### 3 División del Dataset

Separación estratégica para validar el desempeño real del modelo.

* **Definición de variables**:

  * `X` → Variables predictoras.
  * `y` → Variable objetivo (Churn).

* **Train/Test Split**:

  * 80% entrenamiento
  * 20% prueba
  * Uso de `stratify=y` para mantener proporción de churn.

---

### 4 Validación Cruzada (Cross-Validation)

Antes de evaluar el modelo final, se implementa validación cruzada:

* **K-Fold Cross Validation (k=5 o 10)**
  Permite:

  * Reducir varianza en la estimación del rendimiento.
  * Detectar sobreajuste.
  * Obtener métricas promedio más robustas.

Esta etapa es clave para asegurar estabilidad del modelo, se aplica sólo alconjunto de entrenamiento.

---

### 5 Selección y Entrenamiento de Modelos

Se exploran diferentes algoritmos para comparar desempeño:

* **Regresión Logística**

  * Modelo base interpretable.
  * Buena referencia inicial.

* **Random Forest**

  * Maneja relaciones no lineales.
  * Robusto ante outliers y ruido.

* **Gradient Boosting (ej. XGBoost o similares)**

  * Alto poder predictivo.
  * Captura patrones complejos.

* **LightGBM**

  * Más rápido y eficiente que otros boosting

  * Excelente rendimiento en datasets estructurados

  * Maneja bien variables categóricas codificadas


Cada modelo se entrena utilizando validación cruzada para comparación objetiva.

---

### 6 Evaluación de Métricas

Dado que el problema es de clasificación binaria con posible desbalance, se priorizan las siguientes métricas:

* **Matriz de Confusión**

  * Falsos Positivos (FP)
  * Falsos Negativos (FN)

* **Recall (Sensibilidad)**

  * Métrica crítica en churn: identificar correctamente a quienes se irán.

* **Precision**

  * Minimizar falsas alarmas.

* **F1-Score**

  * Balance entre Precision y Recall.

* **ROC-AUC**

  * Evaluación global del poder discriminatorio del modelo.

- En `variables` como churn, **Recall** tiene mayor relevancia en resultados.

---

### 7 Optimización de Hiperparámetros

Para maximizar el rendimiento:

* **GridSearchCV**
* **RandomizedSearchCV**

Aplicado sobre el modelo con mejor desempeño preliminar.

Objetivo:

* Ajustar profundidad de árboles.
* Número de estimadores.
* Regularización.
* Learning rate (en boosting).

Todo validado mediante Cross-Validation.

---

### 8 Modelo Final y Análisis de Importancia

* Entrenamiento final con mejores hiperparámetros.
* Evaluación sobre conjunto de prueba.
* Análisis de importancia de variables.
* Interpretación estratégica de resultados.

---

## Resultados Esperados

* Modelo robusto y validado.
* Identificación de variables clave en churn.
* Segmentación de clientes en riesgo.
* Base técnica para implementar estrategias de retención.

---

## Impacto Estratégico

Este modelo predictivo permite:

* Implementar campañas de retención dirigidas.
* Optimizar recursos comerciales.
* Reducir tasa de churn.
* Incrementar el Customer Lifetime Value (CLV).

