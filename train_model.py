from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'student-mat.csv'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')

data = pd.read_csv(DATA_PATH, sep=';')
features = ['studytime', 'failures', 'absences', 'G1', 'G2']
target = 'G3'
missing = [col for col in features + [target] if col not in data.columns]
if missing:
    raise ValueError(f'Missing required columns: {missing}')

X = data[features].copy()
y = data[target].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

candidates = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1),
}

results = []
best_name = None
best_mae = float('inf')
best_model = None
for name, model in candidates.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))
    results.append({'Model': name, 'MAE': round(mae, 3), 'RMSE': round(rmse, 3), 'R2': round(r2, 3)})
    print(f'{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}')
    if mae < best_mae:
        best_mae, best_name, best_model = mae, name, model

joblib.dump(best_model, MODEL_DIR / 'student_model.pkl')
joblib.dump({'features': features, 'model_name': best_name, 'grade_scale': '0-20', 'dataset': 'UCI Student Performance (Mathematics)'}, MODEL_DIR / 'model_info.pkl')
pd.DataFrame(results).to_csv(MODEL_DIR / 'model_results.csv', index=False)
print(f'\nBest model (lowest test MAE): {best_name}')
print('Saved model, model_info.pkl and model_results.csv to models/')
print('Note: These are held-out results from a single train/test split, not a guarantee of real-world accuracy.')
