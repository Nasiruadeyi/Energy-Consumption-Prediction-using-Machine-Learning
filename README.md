# ⚡ Energy Consumption Prediction

## 📌 Project Overview  
This project explores **building energy consumption** and develops a machine learning model that predicts daily energy usage based on key environmental and operational factors. It includes full **EDA, feature analysis, model experimentation**, and performance evaluation.

Linear Regression delivered the best performance with a **Mean Absolute Error (MAE) of 4.11**, outperforming Lasso, KNN, Random Forest, and XGBoost.

---

## 🗂️ Dataset  
The dataset was sourced locally and contains building-level energy usage with associated environmental and operational metrics.

### **Features**
- `Temperature`
- `Humidity`
- `SquareFootage`
- `Occupancy`
- `HVACUsage`
- `LightingUsage`
- `RenewableEnergy`
- `DayOfWeek`
- `Holiday`
- `EnergyConsumption` *(target variable)*

All columns are complete (`0` missing values).

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights include:

- **Humidity patterns across the week** using area plots  
- Relationship between **Square Footage and Energy Consumption**  
- Distribution of **Renewable Energy usage** across different days  
- Correlation patterns between building operations and consumption  

**Example visualizations (to be added):**
- `/images/humidity_by_day.png`
- `/images/sqft_vs_energy.png`
- `/images/renewable_energy_dist.png`

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

- **Linear Regression**
- **Lasso Regression**
- **Random Forest Regressor**
- **K-Nearest Neighbors Regressor**
- **XGBoost Regressor**

### **Model Input**
```python
x = df_energy.drop(['EnergyConsumption'], axis=1)
y = df_energy['EnergyConsumption']
