# offline_train_and_save.py (run locally)
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

df_train = pd.read_csv("path/to/labeled.csv")
target = "Conversion_Rate"
X = df_train.drop(columns=[target])
y = df_train[target]

# Fit encoders for object columns
encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    le.fit(X[col].astype(str).tolist())
    encoders[col] = le
    X[col] = le.transform(X[col].astype(str))

# Train
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model and encoders
os.makedirs("core/sales_model/encoders", exist_ok=True)
joblib.dump(model, "core/sales_model/sales_rf_model.joblib")
for col, le in encoders.items():
    joblib.dump(le, f"core/sales_model/encoders/{col}.joblib")
