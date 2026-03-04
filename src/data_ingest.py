'''
load the financial data
'''
from pandas_datareader import data as pdr
import pandas as pd


def load_spy(start="2005-01-01"):
    df = pdr.DataReader("spy.us", "stooq", start=start)
    df = df.sort_index()
    return df


if __name__ == "__main__":
    df = load_spy()
    print(df.tail())
