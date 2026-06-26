# Hybrid Classical–Quantum Machine Learning for Financial Regime Detection

## Overview

This repository investigates whether **near-term quantum machine learning (QML)** methods can provide useful inductive biases for **financial regime detection** when embedded within a **hybrid classical–quantum architecture**.

Rather than claiming quantum advantage, the objective is to **rigorously evaluate when quantum components help, when they fail, and why**, under realistic constraints such as small qubit counts, noisy simulations, and limited data availability.

The emphasis is on **methodological rigor, benchmarking against strong classical baselines, and economic interpretability** — not quantum novelty.

---

## Project Status

The full pipeline is **implemented and operational**, covering:

- Data ingestion and feature engineering
- Feature importance analysis and selection
- Classical ensemble model training (LightGBM, XGBoost)
- Quantum Kernel SVC training via PennyLane (CPU and GPU)
- Walk-forward backtesting with realistic execution assumptions
- Multi-model comparison and stress testing
- Portfolio construction from ranked model outputs
- A Next.js dashboard for interactive result exploration

---

## Motivation

Financial markets exhibit **non-stationary behavior** driven by latent regimes (e.g. volatility states, risk-on / risk-off environments). Detecting these regimes is central to:

- Risk management
- Portfolio construction
- Drawdown control
- Strategy robustness

Quantum machine learning offers alternative ways to construct **nonlinear feature maps and kernels**, but its real-world relevance — particularly in finance — remains unclear.

This project addresses the question:

> *Can small, noisy quantum circuits act as useful feature transformations when integrated into well-specified classical ML pipelines?*

---

## Problem Definition

We study **market regime classification** using financial time-series data, focusing on **directional and risk-adjusted outcomes** rather than raw price prediction.

Targets are defined over multiple horizons (5d, 10d, 20d) using **future cumulative log returns**, including a **volatility-adjusted multi-class formulation**:

- `0` = poor risk-adjusted outcome (sell / avoid)
- `1` = neutral outcome (hold)
- `2` = strong risk-adjusted outcome (buy)

This framing aligns more closely with real trading and risk decisions than binary up/down labels. All quantum models are evaluated **strictly relative to classical baselines** trained on the same inputs.

---

## Repository Structure

```
.
├── main.py                          # End-to-end pipeline orchestration
├── utilities.py                     # Shared helpers (folder creation, JSON I/O, etc.)
├── requirements.txt
├── configs/
│   ├── base.yaml                    # Experiment-level defaults
│   ├── ML.yaml                      # Classical model config
│   └── QML.yaml                     # Quantum model config
├── Classes/
│   ├── FeatureExtraction/
│   │   ├── DataGenerator.py         # yfinance downloader
│   │   ├── FeatureEngineering.py    # Feature construction (trends, vol, path dependence)
│   │   ├── SignalAnalysis.py        # Per-feature importance & stability analysis
│   │   └── FeatureAnalysis.py       # Cross-horizon feature ranking and pruning
│   ├── ModelArchitectures/
│   │   ├── QuantumModels.py         # Quantum Kernel SVC (PennyLane, state-embedding mode)
│   │   └── TreeEnsemble.py          # LightGBM and XGBoost classifiers + walk-forward CV
│   ├── ModelAnalysis/
│   │   ├── BackTesting.py           # Realistic backtesting engine with execution model
│   │   └── ModelComparison.py       # Multi-model scoring, stress tests, calibration metrics
│   ├── Trading/
│   │   └── PortfolioManagement.py   # Capital allocation across ranked models/stocks
│   └── Plotting.py                  # Shared visualisation utilities
├── data/
│   ├── json/
│   │   ├── tracked_stocks.json      # Universe of stocks, organised by sector
│   │   ├── portfolio_key_features.json  # Per-stock core feature sets
│   │   ├── stock_specific_model_rankings.json  # Best model per stock
│   │   └── portfolio/
│   │       └── holdings_diversity.jsonl  # Timestamped portfolio snapshots
│   └── results/                     # Trained model parquets, backtest artifacts
└── dashboard/                       # Next.js interactive results dashboard
    ├── app/
    │   └── api/                     # REST endpoints (run, catalog, portfolio, trades, history)
    └── components/dashboard/        # React UI components
```

---

## Data

### Sources

- Historical daily OHLCV data via `yfinance`
- Equity indices, crypto assets, and FX pairs
- Universe defined in `data/json/tracked_stocks.json`, organised by economic sector

Data is stored locally and processed into per-asset feature matrices to prevent cross-asset leakage.

---

## Feature Engineering

Feature construction follows a **deliberately conservative, finance-first approach**, avoiding indicator overloading and excessive dimensionality.

### Feature Families

| Family | Features |
|---|---|
| Trend / Momentum | Rolling mean log returns (5d, 10d, 20d); price-to-MA ratios (20d, 60d) |
| Risk / Volatility | Rolling volatility (20d, 60d); EWMA vol; vol-of-vol; downside vol |
| Path Dependence | Drawdown; rolling max drawdown (252d); time since peak |
| Distributional Shape | Skewness of returns (60d) |
| Volume | Log volume z-scores (20d) |

All features use **only information available at time _t_** to prevent look-ahead bias.

---

## Target Construction

Targets are derived from future cumulative log returns:

$$R_{t,h} = \sum_{i=1}^{h} \log \left(\frac{P_{t+i}}{P_{t+i-1}}\right)$$

Two formulations are used:

- **Binary direction** (up / down)
- **Volatility-adjusted multi-class targets** — future returns normalised by current volatility and discretised into quantiles (Sell / Hold / Buy)

The model is trained on the **10-day volatility-adjusted signal** (`signal_voladj_10d`) by default.

---

## Signal Analysis & Feature Selection

Before training, the pipeline performs **rigorous classical signal analysis** to identify a **small, stable, and interpretable feature set**.

### Step 1 — Permutation Importance (Primary Filter)

L1-regularised logistic regression is trained for each horizon (5d, 10d, 20d). Permutation importance is computed on held-out validation data. Features with near-zero or negative importance are discarded.

### Step 2 — Cross-Horizon Stability

Importance results are aggregated across horizons to compute mean importance, variance, and stability ratios. Features significant at only one horizon are treated as tactical rather than structural.

### Step 3 — Redundancy Pruning (Correlation & VIF)

Feature-feature correlations are computed and features are grouped into families. Variance Inflation Factor (VIF) confirms redundancy numerically. Within each family, only the strongest representative feature is retained.

### Step 4 — Coefficient Interpretation

L1 logistic model coefficient signs are inspected for economic coherence across horizons (for interpretability only — not used as a selection criterion).

### Final Core Feature Set

Following the pipeline, the analysis consistently identifies a compact signal set dominated by:

- **Medium-term trend** (primary driver)
- **Relative trend positioning** (MA ratio)
- **Current volatility regime**

Volume-based and extended path-dependence features did **not** demonstrate stable incremental value for the tested assets and horizons, and were excluded.

This small, interpretable feature set is intentionally well-suited for dimensionality-constrained quantum embeddings.

---

## Model Architectures

### Classical Baselines — Tree Ensembles (`TreeEnsemble.py`)

Two gradient-boosted tree classifiers are trained as baselines:

| Model | Library | Key Config |
|---|---|---|
| LightGBM | `lightgbm` | 5000 estimators, LR 0.02, early stopping (100 rounds) |
| XGBoost | `xgboost` | 5000 estimators, LR 0.02, `hist` tree method |

Both use `objective="multiclass"` with 3 classes and are trained via **walk-forward cross-validation** (`walkforward_cv_predict`) to respect the time-series structure of the data.

### Quantum Model — Quantum Kernel SVC (`QuantumModels.py`)

The quantum model uses a **quantum kernel Support Vector Classifier** built with PennyLane:

**Feature Map:** angle embedding with RY rotations and CNOT entanglement

```
for depth repetitions:
    RY(x[i]) for each qubit i
    CNOT(0→1), CNOT(1→2)
```

**Kernel construction (fast path):** Rather than evaluating O(N²) pairwise circuits, the implementation uses a **state-embedding approach**:

1. Compute `|ψ(x)>` once per sample — O(N) circuit calls
2. Compute the Gram matrix via BLAS matrix multiply: `K_ij = |<ψ_i|ψ_j>|²`

This is substantially faster than pairwise evaluation for moderately sized datasets.

**Calibration:** SVC decision scores are calibrated to probabilities using logistic regression, enabling probability-weighted position sizing downstream.

**Hardware targets:**
- `lightning.qubit` (CPU, Windows/Mac)
- `lightning.gpu` (GPU, Linux)

**Hyperparameter search:** C ∈ {0.1, 1.0, 10.0}, circuit depth = 2 by default.

---

## Training Pipeline

The end-to-end pipeline is orchestrated from `main.py`:

```
1. download_data()              — fetch OHLCV data via yfinance
2. feature_engineering()        — compute full feature set for all tickers
3. feature_analysis()           — prune to core feature set per stock
4. model_training()             — train ensemble + quantum models
      ├── ensemble_model_training()      — LGBM + XGBoost walk-forward CV
      └── quantum_model_training()       — Quantum Kernel SVC with C sweep
5. run_backtests_with_comparison()  — backtest all models, rank by performance
6. generate_diversified_portfolio() — allocate capital across top models
```

### Data splits

- **Dev / Test**: 80 / 20 chronological split
- **Walk-forward CV**: 5 folds with a time-series gap equal to the prediction horizon
- **Quantum test calibration**: final 20% of dev set used as a chronological calibration slice before test evaluation

---

## Backtesting & Evaluation (`BackTesting.py`, `ModelComparison.py`)

The `BackTest` class simulates realistic execution:

| Parameter | Default |
|---|---|
| Transaction cost | 5 bps per unit turnover |
| Execution lag | 1 day |
| Entry threshold | 20% probability |
| Position mode | Long-only |
| Hold rule | Signal-until-change |

**Performance metrics computed:**

- Total and annualised log return
- Annualised volatility
- Sharpe ratio
- Maximum drawdown (log space)
- Average annual turnover
- Hit rate
- % time in market

**Probability quality metrics** (`ModelComparison.py`):

- Log loss
- Multiclass Brier score
- Expected Calibration Error (ECE) on top-class confidence

**Stress testing** (`StressGrid`): each model is re-evaluated across a grid of transaction costs (5 / 10 / 20 bps), execution lags, and entry thresholds to assess robustness.

### Model Ranking

`StockModelEvaluator` scores and ranks all models for a given stock using a weighted composite of the metrics above, with trading efficiency metrics (Sharpe per turnover, return per turnover) given the highest weight. Rankings are saved to `data/json/stock_specific_model_rankings.json`.

---

## Portfolio Construction (`PortfolioManagement.py`)

`PortfolioManager` takes the per-stock model rankings and constructs a diversified portfolio:

1. Loads sector/symbol data from `data/json/tracked_stocks.json`
2. Selects the best-performing model for each stock
3. Scores stocks using a configurable **investment philosophy** (loaded from `data/json/portfolio_philosophy.json`)
4. Allocates capital using **SOFTMAX weighting** over allocation scores
5. Assesses sector exposure and saves a timestamped snapshot to `data/json/portfolio/holdings_diversity.jsonl`

Available philosophy presets include `BALANCED_V1` (default), with configurable weighting towards risk controls, calibration quality, or pure return.

---

## Dashboard (`dashboard/`)

A **Next.js** dashboard provides interactive exploration of results. It is served separately from the Python pipeline.

### API Endpoints

| Route | Description |
|---|---|
| `/api/run` | Trigger a pipeline run and stream progress |
| `/api/catalog` | List available trained models and backtest results |
| `/api/portfolio` | Return current portfolio state and sector exposure |
| `/api/trades` | Return trade-level detail for a given model run |
| `/api/history` | Return equity curve and drawdown time series |
| `/api/assets` | List available stock assets |

### Dashboard Components

- **KPI Row** — summary metrics at a glance
- **Equity Curve Card** — cumulative return chart
- **Drawdown Card** — drawdown time series
- **Exposure Card** — position and sector exposure
- **Signal Scatter Card** — predicted probabilities vs outcomes
- **Model Compare Tab** — side-by-side model performance comparison
- **Portfolio Tab** — capital allocation and diversity breakdown
- **Trades Tab** — trade-level log with entry/exit detail
- **Stock History Ticker** — scrolling historical price with signal overlay

### Running the Dashboard

```bash
cd dashboard
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## Installation

```bash
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pennylane` | 0.43.2 | Quantum circuit simulation |
| `lightgbm` | 4.6.0 | Gradient-boosted tree classifier |
| `xgboost` | 3.1.3 | Gradient-boosted tree classifier |
| `scikit-learn` | 1.8.0 | SVC, logistic regression, metrics |
| `yfinance` | 0.2.66 | Market data ingestion |
| `pandas` | 2.3.3 | Data manipulation |
| `numpy` | 2.4.1 | Numerical computation |

For GPU-accelerated quantum kernels on Linux, install `pennylane-lightning[gpu]` separately.

---

## Running the Pipeline

```python
# main.py — configure then run

# 1. Full pipeline (download → features → train → backtest → portfolio)
run_full_pipeline(
    portfolio_symbols,
    download_path='data/csv/historical/training/raw/',
    architectrure_list=['ensemble', 'quantum'],
    system='linux'   # or 'windows' for CPU
)

# 2. Backtest + comparison only (if models are already trained)
run_backtests_with_comparison(
    portfolio_symbols,
    feat_dict_path='data/json/portfolio_key_features.json',
    performance_dict_path='data/json/stock_specific_model_rankings.json'
)

# 3. Portfolio construction only (if rankings are already saved)
generate_diversified_portfolio(
    performance_dict_path='data/json/stock_specific_model_rankings.json',
    portfolio_sectors_path='data/json/tracked_stocks.json',
    philosophy_path='data/json/portfolio_philosophy.json',
    philosophy='BALANCED_V1'
)
```

---

## Design Principles

- **No data leakage**: all features use only information available at time _t_; train/val/test splits are strictly chronological
- **No quantum novelty bias**: quantum models are always compared to strong, properly tuned classical baselines on identical data splits
- **Realistic execution**: backtests include transaction costs, execution lag, and entry thresholds
- **Economic interpretability**: feature selection and model ranking are grounded in financially meaningful metrics, not raw classification accuracy
- **Modular architecture**: each stage (data, features, models, evaluation, portfolio) is independently replaceable

---

## Limitations & Caveats

- Quantum kernel construction is computationally expensive even with the state-embedding optimisation; scaling beyond ~3–5 qubits / a few hundred samples requires further engineering
- The feature map (RY + CNOT ladder) is a simple baseline; more expressive maps may improve quantum performance but also increase barren plateau risk
- Results are simulation-only — no real hardware noise model is applied
- Backtests assume liquid markets and do not model slippage or market impact beyond a flat transaction cost