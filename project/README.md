# Project Title: Tomorrow's Stock Trend Predictor
**Stage:** Problem Framing & Scoping (Stage 01)

## Problem Statement
Day traders struggle to identify reliable short-term opportunities in volatile markets. This project predicts next-day price movements (up/down) for liquid stocks using technical indicators, helping retail investors make data-driven decisions without requiring advanced financial expertise. The solution matters because even a 55-60% prediction accuracy could significantly improve trading outcomes compared to random guessing.

## Stakeholder & User
- **Primary Stakeholder:** Retail trading platform (e.g., Robinhood)
- **End User:** Individual day traders
- **Workflow Context:** Predictions generated daily after market close, consumed via email/API before next trading session

## Useful Answer & Decision
- **Type:** Predictive binary classification
- **Metric:** >60% test accuracy (benchmarked against S&P 500's 52% baseline)
- **Artifact:** 
  - Daily prediction report (CSV)
  - Jupyter notebook with backtesting visuals

## Assumptions & Constraints
- Data: Market datas, assumed to be mostly clean, is available from public API
- Tools: Python (yfinance, scikit-learn), calculation can be done on a personal laptop
- Compliance: No insider/alternative data used

## Known Unknowns / Risks
- Impact of breaking news events (mitigation: exclude earnings weeks)
- Model overfitting to noise (mitigation: time-series cross-validation)
- API rate limits (mitigation: cache historical data)

## Lifecycle Mapping
Goal → Stage → Deliverable
- Define prediction problem & scope → Problem Framing & Scoping (Stage 01) → README.md + stakeholder memo
- Collect/preprocess stock data → Data Collection & Preprocessing (Stage 02) → Cleaned dataset (CSV) + data dictionary
- Identify key technical indicators → Exploratory Data Analysis (Stage 03) → EDA notebook + feature importance report
- Train price direction classifier → Modeling (Stage 04) → Pickled model + accuracy metrics
- Validate with historical data → Backtesting (Stage 05) → Walk-forward validation results
- Generate daily predictions → Deployment (Stage 06) → Daily buy/sell signals (CSV)
- Monitor live accuracy → Performance Tracking (Stage 07) → Weekly accuracy dashboard

## Repo Plan
- /data/: OHLCV CSVs + news headlines
- /src/: feature_engineering.py, prediction_model.py
- /notebooks/: EDA, model training, backtesting
- /docs/: User guide for interpreting signals
- **Update Cadence:** Daily predictions at market close + weekly model retraining


## Data Storage
- **Raw Data**: `data/raw/` - Original TSLA data from Yahoo Finance
- **Processed Data**: `data/processed/` - Cleaned data with technical indicators
- **Formats**: CSV files with timestamped filenames
- **Environment**: Paths configured via `.env` file

## Key Features
- **Data Acquisition**: Yahoo Finance API integration
- **Technical Indicators**: SMA, Returns, Volatility calculations
- **Target Variable**: Next-day price direction prediction
- **Modular Code**: Reusable functions across all stages

## Usage
1. Set up environment variables in `.env`
2. Run notebooks in sequence from Stage 03 to Stage 06
3. Data flows through: Acquisition → Storage → Preprocessing → Modeling