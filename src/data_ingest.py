from pathlib import Path
import pandas as pd
from pandas_datareader import data as pdr

from src.config import CFG

DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw_spy.parquet"

def load_ohlcv(symbol: str = CFG.symbol, start: str = CFG.start) -> pd.DataFrame:
    df = pdr.DataReader(symbol, CFG.source, start=start)
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    return df

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_ohlcv()
    df.to_parquet(RAW_PATH)
    print(f"Saved raw data to {RAW_PATH} with shape {df.shape}")

if __name__ == "__main__":
    main()