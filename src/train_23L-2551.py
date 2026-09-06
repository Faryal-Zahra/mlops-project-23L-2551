import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

print("Loading dataset...")
df = pd.read_csv("data/dataset.csv")


df = df.select_dtypes(include=["number"])


df = df.fillna(df.median())

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model_23L-2551.pkl")

print("Model trained and saved successfully in model/ folder!")