import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Step 1: Create Real-Time Like Dataset
# -------------------------------
data = {
    "Area": [500, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000],
    "Price": [15, 25, 30, 36, 45, 55, 60, 65, 75, 90]
}

df = pd.DataFrame(data)

# -------------------------------
# Step 2: Feature & Target Split
# -------------------------------
X = df[["Area"]]      # Independent variable
y = df["Price"]      # Dependent variable

# -------------------------------
# Step 3: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Step 4: Train Linear Regression Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Predictions
# -------------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# -------------------------------
# Step 6: Evaluation Metrics Function
# -------------------------------
def evaluate_model(y_true, y_pred, p):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return r2, adj_r2, mae, mse, rmse

# Number of features
p = X.shape[1]

# -------------------------------
# Step 7: Evaluate on Training & Testing Data
# -------------------------------
train_metrics = evaluate_model(y_train, y_train_pred, p)
test_metrics = evaluate_model(y_test, y_test_pred, p)

# -------------------------------
# Step 8: Display Results Clearly
# -------------------------------
print("\n========== Model Evaluation (House Price Prediction) ==========\n")

print("Training Data Performance:")
print(f"R-Squared (R²)          : {train_metrics[0]:.4f}")
print(f"Adjusted R-Squared     : {train_metrics[1]:.4f}")
print(f"Mean Absolute Error    : {train_metrics[2]:.4f}")
print(f"Mean Squared Error     : {train_metrics[3]:.4f}")
print(f"Root Mean Squared Error: {train_metrics[4]:.4f}")

print("\nTesting Data Performance:")
print(f"R-Squared (R²)          : {test_metrics[0]:.4f}")
print(f"Adjusted R-Squared     : {test_metrics[1]:.4f}")
print(f"Mean Absolute Error    : {test_metrics[2]:.4f}")
print(f"Mean Squared Error     : {test_metrics[3]:.4f}")
print(f"Root Mean Squared Error: {test_metrics[4]:.4f}")

# -------------------------------
# Step 9: Model Parameters
# -------------------------------
print("\nModel Parameters:")
print("Intercept:", round(model.intercept_, 4))
print("Coefficient (Slope):", round(model.coef_[0], 6))