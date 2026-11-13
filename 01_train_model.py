import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import re

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "health_activity_data.csv"
MODEL_PATH = ROOT / "models" / "heartcoach_rf.pkl"
ENC_PATH = ROOT / "models" / "label_encoder.pkl"
META_PATH = ROOT / "docs" / "used_features.json"

NUM_CANDIDATES = [
    "Age","Height_cm","Weight_kg","BMI","Daily_Steps","Calories_Intake",
    "Hours_of_Sleep","Heart_Rate","Exercise_Hours_per_Week",
    "Systolic","Diastolic","Alcohol_Consumption_per_Week"
]
CAT_CANDIDATES = ["Gender","Smoker","Diabetic"]
TARGET = "Heart_Disease"

BP_REGEX = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})")

def split_bp(series):
    systolic, diastolic = [], []
    for v in series.astype(str):
        m = BP_REGEX.search(v)
        if m:
            systolic.append(float(m.group(1)))
            diastolic.append(float(m.group(2)))
        else:
            systolic.append(np.nan)
            diastolic.append(np.nan)
    return pd.Series(systolic, name="Systolic"), pd.Series(diastolic, name="Diastolic")

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Blood_Pressure" in out.columns and "Systolic" not in out.columns:
        sys, dia = split_bp(out["Blood_Pressure"])
        out["Systolic"] = sys
        out["Diastolic"] = dia
    if "BMI" not in out.columns and {"Height_cm","Weight_kg"}.issubset(out.columns):
        h_m = out["Height_cm"] / 100.0
        out["BMI"] = out["Weight_kg"] / (h_m * h_m)
    return out

def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = derive_features(df)
    for c in NUM_CANDIDATES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [c for c in NUM_CANDIDATES if c in df.columns] + \
           [c for c in CAT_CANDIDATES if c in df.columns]
    if TARGET in df.columns:
        keep.append(TARGET)
    return df[keep].copy()

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
    df = load_and_prepare(DATA_PATH)

    if TARGET not in df.columns:
        raise RuntimeError(f"Target column '{TARGET}' not found in dataset.")

    df = df.dropna(subset=[TARGET])

    # -----------------------------
    # FIX 1: FORCE LABEL ORDER
    # -----------------------------
    le = LabelEncoder()
    le.classes_ = np.array(["No", "Yes"])  # ensures class 1 = Yes
    y_enc = le.transform(df[TARGET].astype(str).str.strip())

    X = df.drop(columns=[TARGET])
    num_cols = [c for c in X.columns if c in NUM_CANDIDATES]
    cat_cols = [c for c in X.columns if c in CAT_CANDIDATES]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ]
    )

    # -----------------------------
    # FIX 2: MAKE MODEL SENSITIVE TO “YES”
    # -----------------------------
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_split=4,   # helps rare cases
        min_samples_leaf=2,    # prevents ignoring small patterns
        class_weight="balanced",
        random_state=42
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]  # P(Yes)

    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")

    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

    print("\n=== Updated Model Metrics ===")
    print(metrics)

    joblib.dump(pipe, MODEL_PATH)
    joblib.dump(le, ENC_PATH)

    meta = {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "target": TARGET,
        "classes": list(le.classes_),
        "metrics": metrics,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
