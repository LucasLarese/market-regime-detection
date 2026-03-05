import argparse
from pathlib import Path
import joblib
import pandas as pd
import json

from src.config import CFG
from src.data_ingest import load_ohlcv
from src.features import make_features
from src.labels import REGIME_ORDER

MODEL_PATH = Path("models") / "regime_model.joblib"
FEATURE_COLS = ["ret_1d","vol_20d","trend_60d","mom_5d","mom_20d","hl_range","co_return","volchg_5d"]

def get_feature_row(date: str | None = None) -> tuple[pd.Timestamp, pd.DataFrame]:
    """
    Build the latest feature row (or nearest previous trading day to `date`).
    Returns (row_date, X_row_df).
    """
    df = load_ohlcv(start=CFG.start)
    feat = make_features(df, vol_window=CFG.vol_window, trend_window=CFG.trend_window)

    if date is None:
        row_date = feat.index[-1]
        row = feat.loc[[row_date], FEATURE_COLS]
        return row_date, row

    ts = pd.to_datetime(date)
    # pick the closest available trading day on/before date
    idx = feat.index[feat.index <= ts]
    if len(idx) == 0:
        raise ValueError(f"No data on or before {date}. Try a later date.")
    row_date = idx[-1]
    row = feat.loc[[row_date], FEATURE_COLS]
    return row_date, row

def main():
    parser = argparse.ArgumentParser(description="Predict market regime probabilities.")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (optional). Uses latest if omitted.")
    parser.add_argument("--json", action="store_true", help="Outpiut as JSON")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it first with: python -m src.train"
        )

    model = joblib.load(MODEL_PATH)

    row_date, X_row = get_feature_row(args.date)

    proba = model.predict_proba(X_row)[0]
    classes = list(model.named_steps["clf"].classes_)

    proba_series = pd.Series(proba, index=classes).sort_values(ascending=False)

    pred = proba_series.index[0]
    conf = float(proba_series.iloc[0])

    result = {
        "date_used": str(row_date.date()),
        "predicted_regime": pred,
        "predicted_probability": round(conf, 6),
        "probabilities": {k: round(float(v), 6) for k, v in proba_series.items()},
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Date used: {row_date.date()}")
        print(f"Predicted regime: {pred} (p={conf:.3f})")
        print("\nProbabilities:")
        print(proba_series.round(3).to_string())

if __name__ == "__main__":
    main()