# Market Regime Detection

End-to-end machine learning workflow to **label, detect, and predict financial market regimes** (trend × volatility) using daily market data.

This project is designed as an ML engineering case study:
- Data ingestion (reproducible)
- Feature engineering (time-series, leakage-aware)
- Regime labeling (transparent rules)
- Model training + evaluation (time-based split)
- Production-ready pipeline scripts

## Dataset
Daily OHLCV data for SPY from Stooq (pulled via `pandas-datareader`).

## Quickstart
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
