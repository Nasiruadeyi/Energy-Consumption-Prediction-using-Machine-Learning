import sys
import os

# Add the repo root folder to Python path
repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)
import pandas as pd
from IPython.display import Image, display
from src.preprocessing import load_data, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def train_and_evaluate():
    df = load_data()
    df = preprocess_data(df)

    X = df.drop("EnergyConsumption", axis=1)
    y = df["EnergyConsumption"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results['Linear Regression'] = {
        'MAE': mean_absolute_error(y_test, lr_pred),
        'MSE': mean_squared_error(y_test, lr_pred),
        'R2': r2_score(y_test, lr_pred)
    }

    # Lasso Regression
    lasso = Lasso(alpha=0.2)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    results['Lasso Regression'] = {
        'MAE': mean_absolute_error(y_test, lasso_pred),
        'MSE': mean_squared_error(y_test, lasso_pred),
        'R2': r2_score(y_test, lasso_pred)
    }

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['Random Forest'] = {
        'MAE': mean_absolute_error(y_test, rf_pred),
        'MSE': mean_squared_error(y_test, rf_pred),
        'R2': r2_score(y_test, rf_pred)
    }

    # KNN
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    results['KNN'] = {
        'MAE': mean_absolute_error(y_test, knn_pred),
        'MSE': mean_squared_error(y_test, knn_pred),
        'R2': r2_score(y_test, knn_pred)
    }

    # XGBoost
    xgb = XGBRegressor(random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results['XGBoost'] = {
        'MAE': mean_absolute_error(y_test, xgb_pred),
        'MSE': mean_squared_error(y_test, xgb_pred),
        'R2': r2_score(y_test, xgb_pred)
    }

    # Create bar plots for MAE and R2
    mae_values = [results[m]['MAE'] for m in results]
    r2_values = [results[m]['R2'] for m in results]

    if not os.path.exists("images"):
        os.makedirs("images")

    plt.figure(figsize=(8,5))
    plt.bar(results.keys(), mae_values, color=['blue','green','red','yellow','tan'])
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE of Different Models')
    plt.ylim(0, max(mae_values)+1)
    plt.savefig("images/models_mae.png")
    plt.close()

    plt.figure(figsize=(8,5))
    plt.bar(results.keys(), r2_values, color=['blue','green','red','yellow','tan'])
    plt.ylabel('R2 Score')
    plt.title('R2 of Different Models')
    plt.ylim(0,1)
    plt.savefig("images/models_r2.png")
    plt.close()

    return results

if __name__ == "__main__":
    res = train_and_evaluate()
    for model, metrics in res.items():
        print(f"{model}: MAE={metrics['MAE']:.2f}, MSE={metrics['MSE']:.2f}, R2={metrics['R2']:.2f}")
