# Hybrid Classical–Quantum Machine Learning for Financial Regime Detection

## Overview

This repository investigates whether **near-term quantum machine learning (QML)** methods can provide useful inductive biases for **financial regime detection** when embedded within a **hybrid classical–quantum architecture**.

Rather than claiming quantum advantage, the objective is to **rigorously evaluate when quantum components help, when they fail, and why**, under realistic constraints such as small qubit counts, noisy simulations, and limited data availability.

The emphasis is on **methodological rigor, benchmarking against strong classical baselines, and economic interpretability**, not quantum novelty.

---

## Motivation

Financial markets exhibit **non-stationary behavior** driven by latent regimes (e.g. volatility states, risk-on / risk-off environments). Detecting these regimes is central to:

- Risk management  
- Portfolio construction  
- Drawdown control  
- Strategy robustness  

Quantum machine learning offers alternative ways to construct **nonlinear feature maps and kernels**, but its real-world relevance—particularly in finance—remains unclear.

This project addresses the question:

> *Can small, noisy quantum circuits act as useful feature transformations when integrated into well-specified classical ML pipelines?*

---

## Problem Definition

We study **market regime classification** using financial time-series data, focusing on **directional and risk-adjusted outcomes** rather than raw price prediction.

Targets are defined over multiple horizons (5d, 10d, 20d) using **future cumulative log returns**, including a **volatility-adjusted multi-class formulation**:

- `0` = poor risk-adjusted outcome (sell / avoid)  
- `1` = neutral outcome (hold)  
- `2` = strong risk-adjusted outcome (buy)  

This framing aligns more closely with real trading and risk decisions than binary up/down labels.

All quantum models are evaluated **strictly relative to classical baselines** trained on the same inputs.

---

## Data

### Data Sources

- Historical daily OHLCV market data  
- Equity indices, crypto assets, and FX pairs (initial experiments focus on single-asset analysis)

Data is stored locally and processed into per-asset feature matrices to avoid cross-asset leakage during signal analysis.

---

## Feature Engineering

Feature construction follows a **deliberately conservative, finance-first approach**, avoiding indicator overloading and excessive dimensionality.

### Engineered Feature Families

Features are grouped into economically motivated families:

#### Trend / Momentum
- Rolling mean log returns (5d, 10d, 20d)
- Price-to-moving-average ratios (20d, 60d)

#### Risk / Volatility
- Rolling volatility (20d, 60d)
- EWMA volatility
- Volatility-of-volatility
- Downside volatility

#### Path Dependence
- Drawdown
- Rolling maximum drawdown
- Time since last peak

#### Distributional Shape
- Skewness of returns

#### Volume
- Log volume z-scores

All features are constructed using **only information available at time _t_** to prevent look-ahead bias.

---

## Target Construction

Targets are derived from **future cumulative log returns**:

\[
R_{t,h} = \sum_{i=1}^{h} \log \left(\frac{P_{t+i}}{P_{t+i-1}}\right)
\]

Two formulations are used:

- **Binary direction** (up / down)
- **Volatility-adjusted multi-class targets**, where future returns are normalized by current volatility and discretized into quantiles

The final rows of each dataset naturally contain NaNs for the target, reflecting **unknowable future information**. These rows are dropped **only at model training time**.

---

## Signal Analysis & Feature Selection

Before introducing quantum models, the project performs **rigorous classical signal analysis** to identify a **small, stable, and interpretable feature set**.

### Step 1 — Permutation Importance (Primary Filter)

- L1-regularized logistic regression is trained for each horizon
- Permutation importance is computed on validation data
- Features with near-zero or negative importance are discarded

This step answers:

> *Does removing this feature actually hurt predictive performance?*

---

### Step 2 — Cross-Horizon Stability

Permutation importance results are aggregated across 5d, 10d, and 20d targets to assess:

- Mean importance  
- Variance across horizons  
- Stability ratios  

Features that matter **only at a single horizon** are treated as tactical rather than structural signals.

---

### Step 3 — Redundancy Pruning (Correlation & VIF)

Using the full historical feature matrix:

- Feature–feature correlation matrices are computed
- Highly correlated features are grouped into families
- Variance Inflation Factor (VIF) is used as a numerical confirmation

Within each family, **only the strongest representative feature is retained**, based on permutation importance and interpretability.

---

### Step 4 — Coefficient Interpretation

Coefficient signs from sparse logistic models are used **only for interpretation**, not selection, answering:

- Is the effect economically sensible?
- Is the sign stable across horizons?

---

## Final Core Feature Set

Following the full selection and validation pipeline, the analysis identifies a compact core signal set dominated by:

- **Medium-term trend** (primary driver)
- **Relative trend positioning**
- **Current volatility regime**

Volume-based and extended path-dependence features did **not** demonstrate stable incremental value for the tested assets and horizons and were excluded.

This small, interpretable feature set is intentionally well-suited for:

- Dimensionality-constrained quantum embeddings  
- Controlled hybrid experiments  
- Ablation and benchmarking studies  

---

## Model Architecture (Upcoming)

The next phase of the project introduces a **hybrid classical–quantum architecture**:

