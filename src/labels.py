import numpy as np
import pandas as pd

REGIME_ORDER = ["bull_calm", "bull_volatile", "bear_calm", "bear_volatile"]

def label_regimes(features: pd.DataFrame,
                  vol_col: str = "vol_20d",
                  trend_col: str = "trend_60d",
                  vol_quantile: float = 0.70) -> pd.Series:
    """
    Create 4 regimes based on trend x volatility.
    """
    vol_thresh = features[vol_col].quantile(vol_quantile)

    vol_regime = np.where(features[vol_col] >= vol_thresh, "high_vol", "low_vol")
    trend_regime = np.where(features[trend_col] >= 0, "up_trend", "down_trend")

    def combine(tr, vr):
        if tr == "up_trend" and vr == "low_vol":
            return "bull_calm"
        if tr == "up_trend" and vr == "high_vol":
            return "bull_volatile"
        if tr == "down_trend" and vr == "low_vol":
            return "bear_calm"
        return "bear_volatile"

    regimes = [combine(t, v) for t, v in zip(trend_regime, vol_regime)]
    return pd.Series(regimes, index=features.index, name="regime")

def make_target(regime: pd.Series, horizon: int = 20) -> pd.Series:
    """
    Predict regime horizon days ahead.
    """
    return regime.shift(-horizon).rename("target_regime")