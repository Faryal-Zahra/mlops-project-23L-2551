# MLOps Project 23L-2551 — House Price Prediction

Student ID: **23L-2551**

A minimal, reproducible MLOps pipeline that separates source code, data, and
model artifacts, and trains a `RandomForestRegressor` to predict house prices.

## Project Structure

```
├── data/              # Raw dataset (ignored by git, not versioned)
├── src/
│   └── train_23L-2551.py   # Training script
├── model/             # Trained model artifact output (ignored by git)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup & Usage

Run these commands from the project root.

```bash
# 1. Clone the repository
git clone https://github.com/Faryal-Zahra/mlops-project-23L-2551.git
cd mlops-project-23L-2551

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your dataset at data/dataset.csv (columns: area_sqft, bedrooms,
#    bathrooms, age_years, distance_to_city_km, price)

# 5. Run the training script
python src/train_model_23L-2551.py
```

The trained model is serialized with `joblib` and saved to
`model/house_price_model_23L-2551.pkl`.

## Notes

- `data/` and `model/` are excluded from version control via `.gitignore` —
  only source code and configuration files are committed.
- See `requirements.txt` for the exact package versions needed to reproduce
  the environment.
