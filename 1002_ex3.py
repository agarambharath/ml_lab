import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc

# Fix for Tkinter / GUI error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------
# Step 1: Create Real-Time Like Dataset
# -------------------------------------------------
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 4, 6, 7, 2, 9],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 60, 78, 82, 58, 92],
    "Result": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1]   # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# -------------------------------------------------
# Step 2: Feature & Target Split
# -------------------------------------------------
X = df[["Hours", "Attendance"]]   # Independent variables
y = df["Result"]                 # Dependent variable

# -------------------------------------------------
# Step 3: Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------------------------
# Step 4: Train Classification Model (Logistic Regression)
# -------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------------------------
# Step 5: Predictions
# -------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]   # Probabilities for ROC

# -------------------------------------------------
# Step 6: Confusion Matrix
# -------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("\n========== Confusion Matrix ==========\n")
print(cm)

print("\nTP (True Positive) :", TP)
print("TN (True Negative) :", TN)
print("FP (False Positive):", FP)
print("FN (False Negative):", FN)

# -------------------------------------------------
# Step 7: Evaluation Metrics
# -------------------------------------------------
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\n========== Model Performance Metrics ==========\n")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# -------------------------------------------------
# Step 8: ROC Curve and AUC (Saved as Image)
# -------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="ROC Curve (AUC = %.3f)" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Student Result Prediction")
plt.legend()

plt.savefig("roc_curve.png")
print("\nROC curve saved as roc_curve.png")

print("\n========== Experiment Completed Successfully ==========\n")
