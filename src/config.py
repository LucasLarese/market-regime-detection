from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    symbol: str = "spy.us"
    source: str = "stooq"
    start: str = "2005-01-01"

    horizon: int = 5           # predict regime 20 trading days ahead
    vol_window: int = 20       # rolling vol window
    trend_window: int = 60     # trend lookback window
    vol_quantile: float = 0.70 # high vol threshold quantile

    random_state: int = 42

CFG = Config()