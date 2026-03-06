# INFORME EJECUTIVO

## Modelo de Predicción de Churn de Clientes

**Proyecto:** Sistema de detección temprana de abandono de clientes  
**Modelo:** Regresión Logística (`class_weight='balanced'`)  
**Fecha de evaluación:** 06 de marzo de 2026  
**Estado del proyecto:** Modelo validado y listo para producción

---

# 1. Resumen Ejecutivo

El presente informe describe los resultados obtenidos en el desarrollo de un **modelo de Machine Learning para la predicción de churn (abandono de clientes)**.

El objetivo del modelo es **identificar clientes con alta probabilidad de abandonar el servicio**, permitiendo a la organización implementar **acciones de retención tempranas y estrategias comerciales dirigidas**.

El modelo seleccionado corresponde a una **Regresión Logística con balanceo de clases**, optimizada para priorizar la **detección de clientes que efectivamente abandonarán el servicio**.

Los resultados obtenidos evidencian una **alta capacidad de detección de churn**, logrando identificar más del **82% de los clientes que abandonarán la compañía**, lo cual lo convierte en una herramienta útil para procesos de **retención preventiva y gestión de clientes en riesgo**.

---

# 2. Desempeño del Modelo

## 2.1 Métricas de Evaluación (Conjunto de Prueba)

| Métrica   | Valor  |
| --------- | ------ |
| Accuracy  | 77.38% |
| Recall    | 82.08% |
| Precision | 52.93% |
| F1-Score  | 64.39% |
| ROC-AUC   | 84.91% |

## 2.2 Interpretación de Métricas

**Recall (82.08%)**  
El modelo detecta correctamente aproximadamente **8 de cada 10 clientes que abandonarán el servicio**.

**Precision (52.93%)**  
De todos los clientes identificados como riesgo de abandono, aproximadamente **la mitad realmente abandonan el servicio**.

**ROC-AUC (84.91%)**  
El modelo presenta una **buena capacidad de discriminación** entre clientes que abandonan y los que permanecen. Un valor superior al 80% indica que el ordenamiento de probabilidad de riesgo es sumamente confiable para el negocio.

## 2.3 Robustez y Validación Cruzada

Para garantizar que estos resultados no fueran fruto del azar, se aplicó una validación cruzada estratificada (5-folds):

- **Estabilidad del Recall:** Se mantuvo en un promedio de **79.85% (±3%)**, confirmando que el modelo es consistente a través de diferentes subconjuntos de datos.
- **Baja Varianza:** La desviación estándar de las métricas fue inferior al 0.02, lo que asegura que el modelo no sufrirá degradaciones drásticas al enfrentarse a nuevos datos.

---

# 3. Interpretación de Resultados desde el Negocio

| Indicador                                          | Resultado |
| -------------------------------------------------- | --------- |
| Clientes que se van detectados correctamente       | **82.1%** |
| Clientes marcados como riesgo que realmente se van | **52.9%** |

Estos resultados indican que el modelo está **optimizado para maximizar la detección de churn**, lo cual es particularmente útil en estrategias de retención donde **es preferible identificar más clientes potencialmente en riesgo aunque existan algunos falsos positivos**.

---

# 4. Factores que Influyen en el Abandono de Clientes

El análisis de los coeficientes del modelo permite identificar los **principales factores que incrementan o reducen la probabilidad de churn**.

---

## 4.1 Factores que Aumentan la Probabilidad de Churn

| Variable                       | Coeficiente |
| ------------------------------ | ----------- |
| InternetService_Fiber optic    | 0.501       |
| ChargesTotal                   | 0.399       |
| PaymentMethod_Electronic check | 0.361       |
| ChargesMonthly                 | 0.354       |
| PaperlessBilling_Yes           | 0.310       |

### Interpretación

Los clientes con estas características presentan **mayor probabilidad de abandono**:

- Uso de **servicio de internet por fibra óptica**
- **Altos cargos mensuales o acumulados**
- Método de pago mediante **cheque electrónico**
- Uso de **facturación electrónica**

Esto puede estar asociado a **sensibilidad al precio, percepción de valor del servicio o experiencia del cliente**.

---

## 4.2 Factores que Reducen la Probabilidad de Churn

| Variable           | Coeficiente |
| ------------------ | ----------- |
| Contract_Two year  | -1.218      |
| Tenure             | -1.090      |
| PhoneService_Yes   | -0.659      |
| Contract_One year  | -0.653      |
| InternetService_No | -0.563      |

### Interpretación

Los clientes con las siguientes características presentan **menor probabilidad de abandonar el servicio**:

- Contratos de **largo plazo**
- **Mayor antigüedad como cliente**
- Uso de **servicio telefónico**
- Contratos de **un año o más**

Esto confirma que **la fidelización aumenta significativamente con la permanencia y los contratos prolongados**.

---

# 5. Recomendaciones Estratégicas

A partir de los resultados obtenidos se proponen las siguientes acciones estratégicas:

| Recomendación                                        | Objetivo                                                     |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| Implementar el modelo en el CRM                      | Generar scoring automático de churn                          |
| Generar alertas para clientes con probabilidad > 60% | Intervención temprana                                        |
| Diseñar campañas para contratos _Month-to-Month_     | Reducir abandono                                             |
| Ofrecer incentivos a clientes de alto gasto          | Retener clientes de alto valor                               |
| Monitorear desempeño del modelo mensualmente         | Control de calidad del modelo                                |
| Reentrenar el modelo trimestralmente                 | Mantener precisión predictiva frente a cambios en el mercado |

---

# 6. Implementación Operativa y Despliegue

El modelo puede integrarse en los sistemas de negocio mediante los siguientes procesos:

- **Evaluación periódica de clientes mediante scoring automático**
- **Segmentación de clientes según riesgo de abandono**
- **Activación automática de campañas de retención**
- **Monitoreo continuo del desempeño del modelo**

La implementación permitirá evolucionar de un enfoque **reactivo a uno predictivo en la gestión de clientes**.

## 6.1 Artefactos de Despliegue

Se han generado los siguientes archivos para la puesta en producción:

- **`churn_model_final.pkl`**: Pipeline completo (escalador + modelo) serializado.
- **`model_info.pkl`**: Metadatos, versión y métricas de referencia.

---

# 7. Resumen del Modelo Final

| Modelo                                        | Fecha      | Recall | Precision | ROC-AUC |
| --------------------------------------------- | ---------- | ------ | --------- | ------- |
| Logistic Regression (class_weight='balanced') | 2026-03-06 | 82.1%  | 52.9%     | 84.9%   |

---

# 8. Conclusión

El modelo desarrollado demuestra **una alta capacidad para identificar clientes con riesgo de abandono**, permitiendo a la organización implementar **estrategias proactivas de retención**.

Su aplicación en los procesos operativos puede contribuir significativamente a:

- **Reducir la pérdida de clientes**
- **Optimizar campañas de retención**
- **Mejorar la rentabilidad del negocio**

El sistema se encuentra **validado y preparado para su implementación en producción**, sujeto a monitoreo continuo y reentrenamiento periódico para mantener su desempeño a lo largo del tiempo.

---

**Estado final del proyecto:**  
✔ Modelo validado  
✔ Interpretación de negocio realizada  
✔ Recomendaciones estratégicas definidas  
✔ Listo para implementación en producción
