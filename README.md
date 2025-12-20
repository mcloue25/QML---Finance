# Hybrid Classical–Quantum Machine Learning for Financial Regime Detection

## Overview

This repository investigates whether **near-term quantum machine learning (QML)** methods can provide useful inductive biases for **financial regime detection** when embedded within a **hybrid classical–quantum architecture**.

Rather than claiming quantum advantage, the objective is to **rigorously evaluate when quantum components help, when they fail, and why**, under realistic constraints such as small qubit counts, noisy simulations, and limited data availability.

The emphasis is on **methodology, benchmarking, and economic interpretability**, not quantum novelty.

---

## Motivation

Financial markets exhibit **non-stationary behavior** driven by latent regimes (e.g., volatility states, risk-on / risk-off environments). Detecting these regimes is central to:

- Risk management  
- Portfolio allocation  
- Drawdown control  
- Strategy robustness  

Quantum machine learning offers new approaches to constructing **nonlinear feature maps and kernels**, but its real-world relevance—particularly in finance—remains unclear.

This project addresses the question:

> *Can a small, noisy quantum circuit serve as a useful feature transformation when integrated into a classical machine learning pipeline?*

---

## Problem Definition

We study **market regime classification** using financial time-series data.

Example tasks include:
- Volatility regime detection (low vs. high variance)
- Market state classification (bull / bear / sideways)
- Latent regime discovery via clustering

All quantum models are evaluated **strictly in comparison with strong classical baselines**.

---

## Data and Features

### Data Sources
- Equity index returns  
- Volatility indices  
- Foreign exchange (FX) pairs  
- Yield curve features  

### Feature Engineering
- Log returns  
- Rolling volatility measures  
- Momentum indicators  
- Cross-asset correlation features  
- Lagged macroeconomic variables  

Feature dimensionality is intentionally constrained to reflect **realistic near-term quantum hardware limits**.

---

## Model Architecture

The core contribution of this work is a **hybrid classical–quantum neural architecture**:

