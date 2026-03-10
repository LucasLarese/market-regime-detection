import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame,
                  vol_window: int = 20,
                  trend_window: int = 60) -> pd.DataFrame:
    """
    Create features from OHLCV.
    Assumes df is indexed by Date ascending and has columns:
    Open, High, Low, Close, Volume
    """
    data = df.copy()

    # log price & returns
    data["log_close"] = np.log(data["Close"])
    data["ret_1d"] = data["log_close"].diff()

    # rolling volatility
    data["vol_20d"] = data["ret_1d"].rolling(vol_window).std()

    # trend proxy (past return)
    data["trend_60d"] = data["log_close"].diff(trend_window)

    # momentum features
    data["mom_5d"] = data["log_close"].diff(5)
    data["mom_20d"] = data["log_close"].diff(20)

    # range / intraday movement proxies
    data["hl_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["co_return"] = (data["Close"] - data["Open"]) / data["Open"]

    # volume change (log)
    data["volchg_5d"] = np.log(data["Volume"]).diff(5)

    # drop rows with NaNs created by rolling/diff
    return data.dropna().copy()
