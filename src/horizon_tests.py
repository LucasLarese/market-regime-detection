from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.config import CFG
from src.data_ingest import load_ohlcv
from src.features import make_features
from src.labels import label_regimes, make_target
from src.model import make_model

RESULTS_DIR = Path("reports")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ["ret_1d","vol_20d","trend_60d","mom_5d","mom_20d","hl_range","co_return","volchg_5d"]

def build_xy(horizon: int):
    df = load_ohlcv(start=CFG.start)
    feat = make_features(df, vol_window=CFG.vol_window, trend_window=CFG.trend_window)
    regime = label_regimes(feat, vol_quantile=CFG.vol_quantile)
    target = make_target(regime, horizon=horizon)

    ds = feat.join(target).dropna()
    X = ds[FEATURE_COLS]
    y = ds["target_regime"]
    return X, y

def evaluate_cv(X, y, model_name: str, n_splits: int = 8):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, baccs, f1s = [], [], []

    for tr, te in tscv.split(X):
        model = make_model(model_name)
        model.fit(X.iloc[tr], y.iloc[tr])

        pred = model.predict(X.iloc[te])
        y_true = y.iloc[te]

        accs.append(accuracy_score(y_true, pred))
        baccs.append(balanced_accuracy_score(y_true, pred))
        f1s.append(f1_score(y_true, pred, average="macro"))

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_accuracy_mean": float(np.mean(baccs)),
        "balanced_accuracy_std": float(np.std(baccs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
    }

def main():
    horizons = [5, 10, 20, 60]
    models = ["rf", "hgb"]

    rows = []
    for h in horizons:
        X, y = build_xy(horizon=h)
        for m in models:
            metrics = evaluate_cv(X, y, model_name=m, n_splits=8)
            rows.append({"horizon": h, "model": m, **metrics})
            print(f"Done: horizon={h}, model={m}")

    results = pd.DataFrame(rows).sort_values(["horizon", "model"]).reset_index(drop=True)

    # Pretty print
    show = results[[
        "horizon","model",
        "accuracy_mean","balanced_accuracy_mean","macro_f1_mean",
        "accuracy_std","balanced_accuracy_std","macro_f1_std"
    ]].copy()

    print("\nCV Results (mean ± std):")
    for _, r in show.iterrows():
        print(
            f"h={int(r.horizon):>2}  {r.model:<3} | "
            f"acc {r.accuracy_mean:.3f}±{r.accuracy_std:.3f}  "
            f"bacc {r.balanced_accuracy_mean:.3f}±{r.balanced_accuracy_std:.3f}  "
            f"f1 {r.macro_f1_mean:.3f}±{r.macro_f1_std:.3f}"
        )

    out_csv = RESULTS_DIR / "horizon_model_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()