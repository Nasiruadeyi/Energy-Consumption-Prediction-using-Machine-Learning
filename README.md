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
```

# 📊 Model Performance

Metrics shown below are computed on the hold-out **test set**.  
If you re-run the scripts/notebook with a different random seed or hyperparameters, numbers may change slightly.

### Mean Absolute Error (MAE), Mean Squared Error (MSE), and R²

| Model              | MAE  | MSE (test)                        | R² (test)                           |
|-------------------|------|-----------------------------------|--------------------------------------|
| Linear Regression | 4.11 | — (compute via `train_model`)     | — (compute via `train_model`)        |
| Lasso Regression  | 4.14 | —                                 | —                                    |
| KNN Regressor     | 6.69 | —                                 | —                                    |
| XGBoost Regressor | 4.78 | —                                 | —                                    |
| Random Forest     | ~5.30| —                                 | —                                    |

**How to fill MSE / R²:**  
Run:

```python
src/train_model.py
```

## 🧪 Workflow (Step-by-step)

### 1. Load data
- **File:** `data/Energy_consumption.csv`  
- **Functions:** `load_data()` and `preprocess_data()` in `src/preprocessing.py`

---

### 2. Preprocessing
- Map `DayOfWeek` → integers (Monday = 1 … Sunday = 7)  
- Encode `HVACUsage` & `LightingUsage` (On = 1, Off = 0)  
- Encode `Holiday` (Yes = 1, No = 0)  
- Drop `Timestamp` if present  

---

### 3. Train/Test Split
```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Model Training
- Linear Regression  
- Lasso Regression  
- Random Forest (n = 100)  
- KNN (k = 5)  
- XGBoost  

---

### 5. Evaluation
- Compute **MAE, MSE, R²** on the test set  
- Save plots to the `images/` directory  

---

### 6. Presentation
- `notebooks/01_eda.ipynb` — Exploratory Data Analysis  
- `notebooks/modeling.ipynb` — Runs `train_model.py` and displays saved plots  

---

## 📝 Files in Repository
```python
energy-consumption-prediction/
|
|-- data/
| |-- Energy_consumption.csv
|
|-- notebooks/
| |-- 01_eda.ipynb
| |-- modeling.ipynb
|
|-- src/
| |-- preprocessing.py
| |-- train_model.py
| |-- evaluate.py
| |-- utils.py
|
|-- images/
| |-- humidity_by_day.png
| |-- sqft_vs_energy.png
| |-- models_mae.png
| |-- models_r2.png
|
|-- results/
| |-- final_metrics.json
|
|-- requirements.txt
|-- README.md
```
---

## 🚀 How to Run (Reproducible Steps)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/energy-consumption-prediction.git
cd energy-consumption-prediction
```

  ### 2. Add your dataset
```bash
Energy_consumption.csv
```
(or update the path in src/preprocessing.py).

  ### 3. Install dependencies
```bash
pip install -r requirements.txt
```

  ### 4. Run the modeling script
```bash
python src/train_model.py
```

  ### 5. Open the notebook
```bash
jupyter notebook
notebooks/modeling.ipynb
```



