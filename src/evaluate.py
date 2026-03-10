from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from src.config import CFG
from src.data_ingest import load_ohlcv
from src.features import make_features
from src.labels import label_regimes, make_target, REGIME_ORDER

REPORT_DIR = Path("reports") / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ["ret_1d","vol_20d","trend_60d","mom_5d","mom_20d","hl_range","co_return","volchg_5d"]

def build_dataset():
    df = load_ohlcv(start=CFG.start)
    feat = make_features(df, vol_window=CFG.vol_window, trend_window=CFG.trend_window)
    regime = label_regimes(feat, vol_quantile=CFG.vol_quantile)
    target = make_target(regime, horizon=CFG.horizon)

    ds = feat.join(target).dropna().copy()
    X = ds[FEATURE_COLS]
    y = ds["target_regime"]
    return ds, X, y

def make_model():
    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=CFG.random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    return Pipeline([("clf", clf)])

def walk_forward_probabilities(ds, X, y, n_splits=8):
    """
    Walk-forward evaluation. For each fold:
    train on past, predict probabilities on next block.
    returns a DataFrame of probabilities aligned to test dates.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    probas = []
    metrics = []

    for i, (tr, te) in enumerate(tscv.split(X), start=1):
        model = make_model()
        model.fit(X.iloc[tr], y.iloc[tr])

        y_true = y.iloc[te]
        y_pred = model.predict(X.iloc[te])

        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")

        metrics.append({"fold": i, "accuracy": acc, "balanced_accuracy": bacc, "macro_f1": f1m})

        fold_proba = model.predict_proba(X.iloc[te])
        classes = list(model.named_steps["clf"].classes_)
        fold_df = pd.DataFrame(fold_proba, index=X.iloc[te].index, columns=classes)
        probas.append(fold_df)

    proba_df = pd.concat(probas).sort_index()
    metrics_df = pd.DataFrame(metrics)
    return proba_df, metrics_df

def plot_probabilities(proba_df: pd.DataFrame, outpath: Path):
    # all regimes should appear as columns
    for r in REGIME_ORDER:
        if r not in proba_df.columns:
            proba_df[r] = 0.0
    proba_df = proba_df[REGIME_ORDER]

    plt.figure(figsize=(12, 5))
    for r in REGIME_ORDER:
        plt.plot(proba_df.index, proba_df[r], label=r, linewidth=1)

    plt.title("Walk-forward predicted regime probabilities")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()

def main():
    ds, X, y = build_dataset()
    proba_df, metrics_df = walk_forward_probabilities(ds, X, y, n_splits=8)

    print("Walk-forward metrics (by fold):")
    print(metrics_df.round(3).to_string(index=False))
    print("\nMean metrics:")
    print(metrics_df[["accuracy","balanced_accuracy","macro_f1"]].mean().round(3).to_string())

    outpath = REPORT_DIR / "regime_probabilities.png"
    plot_probabilities(proba_df, outpath)
    print(f"\nSaved to: {outpath}")

if __name__ == "__main__":
    main()
