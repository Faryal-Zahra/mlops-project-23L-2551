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

# Normalization
X = (X - X.mean()) / X.std() 

# Hyperparameters
n_estimators = 150
learning_rate = 0.05

print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model_23L-2551.pkl")

print("Model trained and saved successfully in model/ folder!")