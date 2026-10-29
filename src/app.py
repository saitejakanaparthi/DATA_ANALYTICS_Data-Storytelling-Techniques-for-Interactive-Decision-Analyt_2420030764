from flask import Flask, render_template, jsonify, Response, request
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

app = Flask(__name__)

# ============================================================
# DATASET PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")

DATASETS = {
    "superstore": [
        "Superstore Dataset.csv",
        "superstore_dataset.csv",
        "superstore_preprocessed.csv"
    ],
    "global": [
        "globalsuper.csv",
        "global_superstore.csv",
        "global_superstore_preprocessed.csv"
    ],
    "online": [
        "onlinesales.csv",
        "online_sales.csv",
        "online_sales_preprocessed.csv"
    ],
    "supermarket": [
        "supermarket_dataset.csv",
        "supermarket_cleaned.csv"
    ]
}

DATA_PATH = os.path.join(
    DATA_DIR,
    DATASETS["superstore"][0]
)


# ============================================================
# FIND DATASET FILE
# ============================================================

def get_dataset_path(dataset):
    filenames = DATASETS.get(dataset, DATASETS["superstore"])
    for filename in filenames:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            return path
    return DATA_PATH


# ============================================================
# LOAD DATA
# ============================================================

_DATA_CACHE = {}

def load_data(dataset="superstore"):
    if dataset in _DATA_CACHE:
        return _DATA_CACHE[dataset].copy()

    data_path = get_dataset_path(dataset)

    try:
        df = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="latin-1")

    # Clean column names: strip whitespace and replace dots, hyphens, spaces with underscores
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(".", "_", regex=False)
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # ========================================================
    # MAP DIFFERENT DATASET COLUMN NAMES (SAFE ALIAS MAPPING)
    # ========================================================

    aliases = {
        "Sales": [
            "Sales",
            "Amount",
            "Revenue",
            "Total_Sales"
        ],
        "Profit": [
            "Profit",
            "Net_Profit"
        ],
        "Quantity": [
            "Quantity",
            "Qty"
        ],
        "Discount": [
            "Discount",
            "Discount_Rate"
        ],
        "Order_ID": [
            "Order_ID",
            "OrderId",
            "Order_Id"
        ],
        "Customer_ID": [
            "Customer_ID",
            "CustomerId",
            "Customer_Id"
        ],
        "Customer_Name": [
            "Customer_Name",
            "CustomerName"
        ],
        "Product_Name": [
            "Product_Name",
            "ProductName",
            "Product"
        ],
        "Category": [
            "Category",
            "Product_Category"
        ],
        "Sub_Category": [
            "Sub_Category",
            "SubCategory",
            "Sub_category"
        ],
        "Region": [
            "Region",
            "Market",
            "Area"
        ],
        "Segment": [
            "Segment",
            "Customer_Segment"
        ],
        "Order_Date": [
            "Order_Date",
            "OrderDate",
            "Date"
        ],
        "PaymentMode": [
            "PaymentMode",
            "Payment_Mode",
            "Payment_Type"
        ]
    }

    for standard_col, possible_cols in aliases.items():
        if standard_col not in df.columns:
            for possible_col in possible_cols:
                if possible_col in df.columns:
                    df[standard_col] = df[possible_col]
                    break

    # ========================================================
    # ESSENTIAL IDENTIFIERS & METRIC SAFETY
    # ========================================================

    if "Order_ID" not in df.columns:
        df["Order_ID"] = df.index.astype(str)

    if "Customer_ID" not in df.columns:
        if "Customer_Name" in df.columns:
            df["Customer_ID"] = df["Customer_Name"].astype(str)
        else:
            df["Customer_ID"] = df["Order_ID"]

    # ========================================================
    # CLEAN NUMERIC COLUMNS
    # ========================================================

    numeric_cols = ["Sales", "Profit", "Quantity"]
    if "Discount" in df.columns:
        numeric_cols.append("Discount")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.replace("£", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
                .replace({
                    "nan": None,
                    "NaN": None,
                    "None": None,
                    "": None
                })
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    # ========================================================
    # CONVERT DATES (ONLY IF PRESENT)
    # ========================================================

    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
        valid_dates = df["Order_Date"].dropna()
        if not valid_dates.empty:
            df["Year"] = df["Order_Date"].dt.year
            df["Month"] = df["Order_Date"].dt.month
            df["Month_Name"] = df["Order_Date"].dt.month_name()
            df["Quarter"] = df["Order_Date"].dt.quarter

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df["Profit_Margin"] = 0.0
    non_zero_sales = df["Sales"] != 0
    df.loc[non_zero_sales, "Profit_Margin"] = (
        df.loc[non_zero_sales, "Profit"] / df.loc[non_zero_sales, "Sales"]
    ) * 100

    _DATA_CACHE[dataset] = df
    return df.copy()


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    df = load_data()
    print("=" * 60)
    print("DATA LOADED")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Sales:", df["Sales"].sum())
    print("Profit:", df["Profit"].sum())
    if "Category" in df.columns:
        print("Categories:", df["Category"].dropna().unique().tolist())
    print("=" * 60)

    return render_template("dashboard.html")


# ============================================================
# DATA API
# ============================================================

@app.route("/data")
def get_data():
    try:
        dataset = request.args.get("dataset", "superstore")
        df = load_data(dataset)

        print("API DATASET:", dataset, "| FILE:", get_dataset_path(dataset))

        json_data = df.to_json(orient="records", date_format="iso")
        return Response(json_data, mimetype="application/json")

    except Exception as e:
        print("DATA API ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ============================================================
# KPI API
# ============================================================

@app.route("/kpis")
def get_kpis():
    try:
        dataset = request.args.get("dataset", "superstore")
        df = load_data(dataset)

        total_sales = float(df["Sales"].fillna(0).sum())
        total_profit = float(df["Profit"].fillna(0).sum())
        total_orders = int(df["Order_ID"].nunique())
        total_quantity = int(df["Quantity"].fillna(0).sum())
        total_customers = int(df["Customer_ID"].nunique()) if "Customer_ID" in df.columns else int(total_orders)

        profit_margin = float((total_profit / total_sales) * 100) if total_sales != 0 else 0.0

        return jsonify({
            "total_sales": round(total_sales, 2),
            "total_profit": round(total_profit, 2),
            "total_orders": total_orders,
            "total_quantity": total_quantity,
            "total_customers": total_customers,
            "profit_margin": round(profit_margin, 2)
        })

    except Exception as e:
        print("KPI ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ============================================================
# SEGMENT SALES
# ============================================================

@app.route("/segment-sales")
def segment_sales():
    try:
        dataset = request.args.get("dataset", "superstore")
        df = load_data(dataset)

        if "Segment" not in df.columns:
            return jsonify({"segments": [], "sales": []})

        result = (
            df.groupby("Segment")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        return jsonify({
            "segments": [str(s) for s in result.index.tolist()],
            "sales": [float(v) for v in result.values.tolist()]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# PROFIT BY CATEGORY
# ============================================================

@app.route("/category-profit")
def category_profit():
    try:
        dataset = request.args.get("dataset", "superstore")
        df = load_data(dataset)

        if "Category" not in df.columns:
            return jsonify({"categories": [], "profit": []})

        result = (
            df.groupby("Category")["Profit"]
            .sum()
            .sort_values(ascending=False)
        )

        return jsonify({
            "categories": [str(c) for c in result.index.tolist()],
            "profit": [float(v) for v in result.values.tolist()]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# DISCOUNT VS PROFIT
# ============================================================

@app.route("/discount-profit")
def discount_profit():
    try:
        dataset = request.args.get("dataset", "superstore")
        df = load_data(dataset)

        if "Discount" not in df.columns or "Profit" not in df.columns:
            return jsonify({"discount": [], "profit": []})

        return jsonify({
            "discount": [float(v) for v in df["Discount"].fillna(0).tolist()],
            "profit": [float(v) for v in df["Profit"].fillna(0).tolist()]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ============================================================
# ML MODEL COMPARATIVE ANALYSIS (BASE MODELS + HSRE)
# ============================================================

@app.route("/ml-prediction")
def ml_prediction():
    try:
        dataset = request.args.get("dataset", "superstore")
        df = load_data(dataset).copy()

        target = "Sales"
        if target not in df.columns:
            return jsonify({"error": "Sales column not available in selected dataset"}), 400

        df = df.dropna(subset=[target])

        # Sample for fast, responsive dashboard execution while maintaining statistical significance
        if len(df) > 1500:
            df = df.sample(n=1500, random_state=42)

        if len(df) < 20:
            return jsonify({"error": "Insufficient records for ML model training"}), 400

        # IDENTIFY AVAILABLE FEATURES
        possible_features = [
            "Quantity",
            "Discount",
            "Category",
            "Sub_Category",
            "Region",
            "Segment",
            "Year",
            "Month",
            "Quarter",
            "PaymentMode"
        ]

        # Use only valid features that exist and have variation
        features = [
            col for col in possible_features
            if col in df.columns and df[col].dropna().nunique() > 1
        ]

        if not features:
            features = [col for col in possible_features if col in df.columns]

        if not features:
            return jsonify({"error": "No valid predictive features found"}), 400

        X = df[features].copy()
        y = pd.to_numeric(df[target], errors="coerce").fillna(0)

        # NUMERIC / CATEGORICAL FEATURE SPLIT
        numeric_features = [
            col for col in features
            if pd.api.types.is_numeric_dtype(X[col])
        ]

        categorical_features = [
            col for col in features
            if col not in numeric_features
        ]

        # PREPROCESSING PIPELINE
        transformers = []

        if numeric_features:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ])
            transformers.append(("numeric", numeric_pipeline, numeric_features))

        if categorical_features:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("categorical", categorical_pipeline, categorical_features))

        preprocessor = ColumnTransformer(transformers=transformers)

        # TRAIN / TEST SPLIT (80% / 20%, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        # 1. DEFINE BASE REGRESSION ESTIMATORS (FROM LITERATURE SURVEY)
        base_estimators = [
            ("lr", LinearRegression()),
            ("dt", DecisionTreeRegressor(random_state=42, max_depth=8)),
            ("rf", RandomForestRegressor(n_estimators=40, max_depth=10, random_state=42, n_jobs=1)),
            ("gb", GradientBoostingRegressor(n_estimators=40, learning_rate=0.08, max_depth=4, random_state=42))
        ]

        # 2. DEFINE PROPOSED HYBRID STACKED REGRESSION ENSEMBLE (HSRE)
        hsre_ensemble = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=1
        )

        # 3. COMPLETE MODEL ARSENAL (BASE MODELS + PROPOSED HSRE)
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
            "Random Forest": RandomForestRegressor(n_estimators=40, max_depth=10, random_state=42, n_jobs=1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=40, learning_rate=0.08, max_depth=4, random_state=42),
            "Proposed Hybrid Stacked Regression Ensemble (HSRE)": hsre_ensemble
        }

        # 4. TRAIN AND EVALUATION UNDER UNIFORM CONDITIONS
        results = []
        actual_vs_predicted = []

        for model_name, model in models.items():
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            r2 = float(r2_score(y_test, predictions))
            mae = float(mean_absolute_error(y_test, predictions))
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

            results.append({
                "model": model_name,
                "r2": round(r2, 4),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2)
            })

            # Capture Actual vs Predicted for the Proposed HSRE model
            if model_name == "Proposed Hybrid Stacked Regression Ensemble (HSRE)":
                y_test_list = y_test.tolist()
                pred_list = [float(p) for p in predictions]
                
                # Sample 50 points or sorted slice for high quality, fast visualization
                sample_count = min(60, len(y_test_list))
                indices = np.linspace(0, len(y_test_list) - 1, sample_count, dtype=int)
                for idx in indices:
                    actual_vs_predicted.append({
                        "sample_id": int(idx + 1),
                        "actual": round(float(y_test_list[idx]), 2),
                        "predicted": round(float(pred_list[idx]), 2)
                    })

        # BEST MODEL SELECTION (based on R² Score)
        best_model_entry = max(results, key=lambda x: x["r2"])

        print(f"\n==================== ML COMPARISON: {dataset.upper()} ====================")
        for res in results:
            print(f"{res['model']:52} | R²: {res['r2']:7.4f} | MAE: {res['mae']:8.2f} | RMSE: {res['rmse']:8.2f}")
        print("Best Model:", best_model_entry["model"])

        return jsonify({
            "dataset": dataset,
            "models": results,
            "best_model": best_model_entry["model"],
            "best_r2": best_model_entry["r2"],
            "best_mae": best_model_entry["mae"],
            "best_rmse": best_model_entry["rmse"],
            "actual_vs_predicted": actual_vs_predicted,
            "features": features,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test))
        })

    except Exception as e:
        print("ML COMPARISON ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)