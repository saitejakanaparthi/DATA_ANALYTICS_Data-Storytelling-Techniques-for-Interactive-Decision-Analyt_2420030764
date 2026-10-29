import os
import pandas as pd
import numpy as np
from app import load_data
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def test_all():
    datasets = ["superstore", "global", "online", "supermarket"]

    for ds in datasets:
        df = load_data(ds).copy()
        print(f"\n==================== DATASET: {ds.upper()} (Total Rows: {len(df)}) ====================")
        target = "Sales"
        if target not in df.columns:
            print("Sales target not found!")
            continue
            
        df = df.dropna(subset=[target])
        
        # Sample for fast interactive performance
        if len(df) > 1500:
            df = df.sample(n=1500, random_state=42)
            
        possible_features = [
            "Quantity", "Discount", "Category", "Sub_Category",
            "Region", "Segment", "Year", "Month", "Quarter", "PaymentMode"
        ]
        features = [col for col in possible_features if col in df.columns and df[col].dropna().nunique() > 1]
        if not features:
            features = [col for col in possible_features if col in df.columns]
            
        print(f"Features used: {features}")
        
        X = df[features].copy()
        y = pd.to_numeric(df[target], errors="coerce").fillna(0)
        
        numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(X[col])]
        categorical_features = [col for col in features if col not in numeric_features]
        
        transformers = []
        if numeric_features:
            transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features))
        if categorical_features:
            transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features))
            
        preprocessor = ColumnTransformer(transformers=transformers)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        base_estimators = [
            ("lr", LinearRegression()),
            ("dt", DecisionTreeRegressor(random_state=42, max_depth=8)),
            ("rf", RandomForestRegressor(n_estimators=40, max_depth=10, random_state=42, n_jobs=1)),
            ("gb", GradientBoostingRegressor(n_estimators=40, learning_rate=0.08, max_depth=4, random_state=42))
        ]
        
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
            "Random Forest": RandomForestRegressor(n_estimators=40, max_depth=10, random_state=42, n_jobs=1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=40, learning_rate=0.08, max_depth=4, random_state=42),
            "Proposed Hybrid Stacked Regression Ensemble (HSRE)": StackingRegressor(
                estimators=base_estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=3,
                n_jobs=1
            )
        }
        
        results = []
        for name, m in models.items():
            pipe = Pipeline([("preprocessor", preprocessor), ("model", m)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            results.append({
                "model": name,
                "r2": float(r2),
                "mae": float(mae),
                "rmse": float(rmse)
            })
            print(f"{name:55} | R2: {r2:7.4f} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f}")

if __name__ == "__main__":
    test_all()
