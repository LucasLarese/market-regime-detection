# Market Regime Detection

## Overview
End-to-end machine learning workflow to **label, detect, and predict financial market regimes** (trend × volatility) using daily market data.

This project is designed as an ML engineering case study:
- time-series feature engineering
- leakage safe model evaluation
- probabilistic classification
- reproducible ML pipeline design
- quantitative analysis of model usefulness

## Dataset
Daily OHLCV data for SPY from Stooq - pulled via `pandas-datareader`.

Each observation contains:
- Open
- High
- Low
- Close
- Volume

## Quickstart
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project Aim
The goal of the project is to predict the future market regime using current and historical market information

The regime is defined as a combination of:
- trend direction
- market volatility

The output thus is a multiclass classification model which predicts the regime the market will be in after a specified number of trading days.

## Regime Definition
Two core signals were used to define the market regime.

- **Trend**, measured by the 60 day log return.
- **Volatility**, measured by the 20 day rolling standard deviation of daily returns.

Four market regimes were produced by these two dimensions:
- bull_calm: upward trend, low volatility
- bull_volatile: upward trend, high volatility
- bear_calm: downward trend, low volatility
- bear_volatile: downward trend, high volatility

This allows for easily interpreterable classification targets.

## Feature Engineering
The OHLCV data was transformed into a set of time-series features using only past information to avoid any look ahead bias.

Engineered features include:
- daily log return
- rolling 20 day volatility
- 60 day trend
- 5 and 20 day momentum
- high-low price range
- open to close return
- 5 day volume change

These features were specifically used to try and target short-term movement and market uncertainty.

## Notebook Workflow
Two Jupyter notebooks were used for this project.

### 01_exploration.ipynb
This notebook was used for data exploration, feature design, regime labeling, and inital visualization of regime classification.

### 02_modeling.ipynb
Used to compare models, walk-forward evaluation, horizon analysis, and interpreting results of the final model.

## Production Pipeline
After prototyping in the two notebooks, the logic was used into python modules in the src/ directory.

Key components:
- config.py: sets definitions for data structure and definitions for feature engineering
- data_ingest.py: downloads and saves inputted market data
- features.py: generates the time-series features
- labels.py: creates the regime labels and forecasting target depending on horizon length
- model.py: defines the ML model used in the project
- train.py: trains and saves the ML model
- predict.py: outputs the predicted regime proabilities
- evaluate.py: performs walk-forward backtesting and visualization
- horizon_tests.py:

## Modeling Approach
The problem was formulated as a multiclass classification task.

Two models were initially tested in 02_modeling.ipynb:
- Logistic Regression as the baseline
- Random Forest as a non-linear model

Later, a comparison also included HistGradientBoosting to test a more advanced boosting classifier.

The purpose to comparing multiple models was to determine not only which had the best accuracy, but also understand whether the regime structure could be captured by simply linear relationships or required a more flexible non-linear model.

## Evaluation
Because this is a time-series problem, train_test splitting was not used.

Instead the prokect uses walk-forward validation with 'TimeSeriesSplit' which ensures that:
- training is alwasys performed on past data
- testing is always performed on future data

The prevents any bias from looking into the future and better reflects how a financial forecasting system would be evaluated in practice on live markets.

## Metrics and Class Imbalance
The target regimes were found to be imbalanced, with "bull_calm" occuring much more frequently than other regimes.

As a result, accuracy alone is misleading and to better evaluate the model quality across all classes the following metrics were used:

- Accuracy
- Balanced Accuracy
- Macro F1 Score

This helps assess whether the model is only predicting the dominant class - "bull_calm" - or actually learning useful distinctions across market states.

## Forecast Horizon Study
The project also compares the model performance across multiple forecast horizons:

- 5 trading days
- 10 trading days
- 20 trading days
- 60 trading days

This was to judge how performant the model is across various time scales.

The result was that the model performs better on shorter horizon windows, which makes sense as market structure becomes less stable when looking over longer forecast windows.

## Results
The best performing window was a 5 day forecast horizon.

- Accuracy: 0.823
- Balanced Accuracy: 0.634
- Macro F1: 0.625

These results indicate that the model can capture meaningful market structures beyond the majority class, while in a noisy and uncertain financial environment.

## Probabilistic Regime Forecasting (CHECK AGAIN)
Instead of returning only a single class prediction, the model outputs probabilities for each possible regime.

This is important in financial settings because decisions are rarely made on hard labels alone. Probability outputs allow uncertainty to be incorporated into risk management and decision making.

##
