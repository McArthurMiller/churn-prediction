# Predicción de abandono de clientes (Customer Churn)

Modelo predictivo exploratorio para identificar qué clientes de una empresa de telecomunicaciones tienen mayor probabilidad de abandonar el servicio, con foco en el **proceso** y en **explicar los resultados al negocio**.

## Dataset
[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 clientes, 21 columnas. ~73% No / ~27% Yes (clases desbalanceadas).
Descárgalo y colócalo en `dataset/Telco-Customer-Churn.csv` (no se incluye en el repositorio).

## Proceso
1. Limpieza (`TotalCharges` a numérico; 11 nulos de clientes con `tenure = 0` imputados con 0)
2. EDA: `tenure`, `Contract`, `InternetService` y `MonthlyCharges` se relacionan claramente con el abandono
3. One-hot encoding, split 80/20 estratificado, escalado (ajustado solo con train)
4. Modelos: regresión logística, Random Forest, regresión logística con `class_weight='balanced'`
5. Selección del umbral de decisión con validación cruzada estratificada (sin usar el test)
6. Evaluación final única sobre el conjunto de prueba

## Resultados (test, clase "abandona")
| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logística (umbral 0.5) | 0.806 | 0.659 | 0.559 | 0.605 |
| Random Forest | 0.804 | 0.664 | 0.529 | 0.589 |
| Logística balanced | 0.738 | 0.504 | 0.783 | 0.614 |
| **Logística (umbral 0.35)** | 0.765 | 0.545 | 0.703 | 0.614 |

**Modelo final:** regresión logística con umbral 0.35. Detecta al 70% de los clientes que abandonan (vs. 56% con 0.5). Frente a 0.5, detecta 54 churners más a cambio de 112 falsas alarmas más (≈ 3 contactos por churner adicional). Es rentable si el costo de retención `c < p·V/3` (V = valor de un cliente retenido, p = probabilidad de que la acción funcione).

## Limitaciones
- Un solo split train/test
- Random Forest probado con una sola configuración
- El escalado se ajustó antes de la validación cruzada (fuga menor; se evitaría con un `Pipeline`)
- Los costos de negocio son supuestos, no datos reales

## Cómo reproducirlo
1. Instala las dependencias: `pip install -r requirements.txt`
2. Descarga el dataset de Kaggle y colócalo en `dataset/`
3. Abre `churn_prediction.ipynb` y ejecuta todas las celdas

## Autor
Arturo Jiménez Hernández · [GitHub](https://github.com/McArthurMiller)