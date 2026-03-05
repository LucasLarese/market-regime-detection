from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.config import CFG
from src.data_ingest import load_ohlcv
from src.features import make_features
from src.labels import label_regimes, make_target, REGIME_ORDER

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "regime_model.joblib"

FEATURE_COLS = ["ret_1d","vol_20d","trend_60d","mom_5d","mom_20d","hl_range","co_return","volchg_5d"]

def build_dataset():
    df = load_ohlcv(start=CFG.start)
    feat = make_features(df, vol_window=CFG.vol_window, trend_window=CFG.trend_window)
    regime = label_regimes(feat, vol_quantile=CFG.vol_quantile)
    target = make_target(regime, horizon=CFG.horizon)
    ds = feat.join(target).dropna()
    X = ds[FEATURE_COLS]
    y = ds["target_regime"]
    return X, y

def main():
    X, y = build_dataset()
    split = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=CFG.random_state,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    pipe = Pipeline([("clf", clf)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Holdout accuracy: {acc:.3f}")

    proba = pipe.predict_proba(X_test.iloc[[-1]])[0]
    class_order = list(pipe.named_steps["clf"].classes_)
    proba_series = pd.Series(proba, index=class_order).sort_values(ascending=False)

    print("\nExample: predicted regime probabilities for the last test sample:")
    print(proba_series.round(3).to_string())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()