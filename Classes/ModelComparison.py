from __future__ import annotations
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Iterable

from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve


# Probability quality utilities
def brier_score_multiclass(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3):
    ''' Multiclass Brier score (lower is better).
        Mean squared error between predicted prob vector and one-hot labels.
    '''
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    Y = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))


def reliability_curve_topclass(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10):
    ''' Calibration curve on model confidence:
        x-axis: predicted confidence = max(p)
        y-axis: empirical accuracy within bin
    '''
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    correct = (pred == y_true).astype(int)
    frac_pos, mean_pred = calibration_curve(correct, conf, n_bins=n_bins, strategy="uniform")
    return mean_pred, frac_pos


def expected_calibration_error_topclass(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10):
    ''' Simple ECE on top-class confidence
        lower is better
    '''
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)

    pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)
    correct = (pred == y_true).astype(int)

    # Bin by confidence
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(conf[mask]))
        w = float(np.sum(mask)) / n
        ece += w * abs(bin_acc - bin_conf)
    return float(ece)



# Stress test config
@dataclass(frozen=True)
class StressGrid:
    transaction_costs: Tuple[float, ...] = (0.0005, 0.0010, 0.0020)  # 5/10/20 bps
    execution_lags: Tuple[int, ...] = (1, 2)  # t+1, t+2
    entry_thresholds: Tuple[float, ...] = (0.10, 0.20, 0.30)  # stricter signal filter


# Model evaluation class (works with your BackTest / BacktestConfig)
class StockModelEvaluator:
    ''' Evaluate many models for ONE stock in a consistent, time-series-correct way.
        Assumes you already have:
        - prices: pd.Series indexed by date
        - y_dev, y_test (np.ndarray), and dates_dev, dates_test (np.ndarray or pd.DatetimeIndex)
        - model outputs as DataFrames with date index and probability columns

    This class provides:
      - Probability quality: log loss, Brier, ECE, reliability curve points
      - Trading performance via your BackTest class
      - Turnover efficiency (return per turnover, Sharpe per turnover)
      - Sensitivity / stress tests (costs, lag, thresholds)
    '''
    def __init__(self, stock_name: str, prices: pd.Series, *, BackTestClass, BacktestConfigClass, horizon_days: int = 10, n_classes: int = 3, ann_factor: int = 252):
        self.stock_name = stock_name
        self.prices = prices.copy()
        self.BackTestClass = BackTestClass
        self.BacktestConfigClass = BacktestConfigClass
        self.horizon_days = horizon_days
        self.n_classes = n_classes
        self.ann_factor = ann_factor
        # Storage
        self._models: Dict[str, Dict[str, Any]] = {}


    # Register model predictions
    def register_model(self, model_tag:str, *, preds_dev:pd.DataFrame, preds_test:pd.DataFrame, y_dev:np.ndarray, y_test:np.ndarray, dates_dev:np.ndarray, dates_test:np.ndarray, proba_cols:Tuple[str, str, str] =("p_class_0", "p_class_1", "p_class_2"), pred_col:str ="y_pred"):
        ''' Register one model's dev/test predictions for later evaluation.
            preds_dev/preds_test must be indexed by date (DatetimeIndex preferred) and contain proba_cols.
            y_dev/y_test must align 1:1 with dates_dev/dates_test.
        '''
        preds_dev = preds_dev.copy()
        preds_test = preds_test.copy()

        # Ensure dates are datetime
        dates_dev = pd.to_datetime(dates_dev)
        dates_test = pd.to_datetime(dates_test)

        # Reindex preds to the provided date arrays (prevents silent misalignment)
        preds_dev = preds_dev.reindex(pd.DatetimeIndex(dates_dev))
        preds_test = preds_test.reindex(pd.DatetimeIndex(dates_test))
        
        self._models[model_tag] = {
            "preds_dev": preds_dev,
            "preds_test": preds_test,
            "y_dev": np.asarray(y_dev).astype(int),
            "y_test": np.asarray(y_test).astype(int),
            "dates_dev": pd.DatetimeIndex(dates_dev),
            "dates_test": pd.DatetimeIndex(dates_test),
            "proba_cols": proba_cols,
            "pred_col": pred_col,
        }


    # Core metric computations
    def _extract_proba(self, preds: pd.DataFrame, proba_cols: Tuple[str, str, str]):
        proba = preds.loc[:, list(proba_cols)].to_numpy(dtype=float)
        # Optional: guard against tiny numeric issues
        proba = np.clip(proba, 1e-12, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba



    # def probability_quality_metrics(self, y_true: np.ndarray, proba: np.ndarray):
    #     ''' Probability quality metrics suitable for your policy layer.
    #     '''
    #     metrics = {}
    #     print(y_true)
    #     print("---------")
    #     print(proba)
    #     a-b
    #     metrics["log_loss"] = float(log_loss(y_true, proba, labels=list(range(self.n_classes))))
    #     metrics["brier"] = brier_score_multiclass(y_true, proba, n_classes=self.n_classes)
    #     metrics["ece_topclass"] = expected_calibration_error_topclass(y_true, proba, n_bins=10)
    #     return metrics



    def probability_quality_metrics(self, y_true: np.ndarray, proba: np.ndarray):
        ''' Probability quality metrics suitable for your policy layer.
        '''
        proba = np.asarray(proba, dtype=float)
        y_true = np.asarray(y_true).astype(int)

        valid = ~np.isnan(proba).any(axis=1)
        # if valid.sum() == 0:
        #     raise ValueError("All rows are NaN. Your predictions are not aligned to labels/dates.")

        y_true_v = y_true[valid]
        proba_v = proba[valid]

        # clip + renormalise (safe)
        proba_v = np.clip(proba_v, 1e-12, 1.0)
        proba_v = proba_v / proba_v.sum(axis=1, keepdims=True)

        return {
            "n_total": int(len(y_true)),
            "n_used": int(valid.sum()),
            "n_dropped": int((~valid).sum()),
            "log_loss": float(log_loss(y_true_v, proba_v, labels=list(range(self.n_classes)))),
            "brier": brier_score_multiclass(y_true_v, proba_v, n_classes=self.n_classes),
            "ece_topclass": expected_calibration_error_topclass(y_true_v, proba_v, n_bins=10),
        }




    def reliability_topclass_points(self, y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10):
        ''' Returns x/y points for a confidence calibration curve:
            - x: mean predicted confidence
            - y: empirical accuracy
        '''
        x, y = reliability_curve_topclass(y_true, proba, n_bins=n_bins)
        return {"mean_confidence": list(map(float, x)), "empirical_accuracy": list(map(float, y))}


    # Trading evaluation
    def _run_backtest(
        self,
        preds_df: pd.DataFrame,
        *,
        transaction_cost: float,
        execution_lag: int,
        entry_threshold: float,
        hold_rule: str = "signal_until_change",
        mode: str = "long_only",
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        '''
        Run your BackTest with stress-test parameters.
        Requires your BackTest.run() to use config.transaction_cost and config.execution_lag and config.entry_threshold.
        If you haven't added execution_lag/entry_threshold to BacktestConfig yet, do that first.
        '''
        cfg = self.BacktestConfigClass(
            horizon_days=self.horizon_days,
            transaction_cost=transaction_cost,
            mode=mode,
            hold_rule=hold_rule,
            # these two must exist in your config; add them if not present
            execution_lag=execution_lag,
            entry_threshold=entry_threshold,
        )
        bt = self.BackTestClass(config=cfg)
        bt_df = bt.run(preds_df=preds_df, prices=self.prices)
        metrics = bt.performance_metrics(bt_df, ann_factor=self.ann_factor)
        return bt_df, metrics


    @staticmethod
    def turnover_efficiency(bt_df: pd.DataFrame, metrics: Dict[str, float]) -> Dict[str, float]:
        ''' Add turnover-efficiency diagnostics.
        '''
        total_turnover = float(bt_df["turnover"].sum())
        total_return = float(bt_df["net_ret"].sum())  # total log return
        ret_per_turnover = total_return / total_turnover if total_turnover > 0 else 0.0

        avg_turnover_per_year = float(metrics.get("avg_turnover_per_year", 0.0))
        sharpe = float(metrics.get("sharpe", 0.0))
        sharpe_per_turnover = sharpe / avg_turnover_per_year if avg_turnover_per_year > 0 else 0.0

        return {
            "total_turnover": total_turnover,
            "return_per_turnover": float(ret_per_turnover),
            "sharpe_per_turnover": float(sharpe_per_turnover),
        }
    

    # Public API: evaluate models
    def evaluate_model(
        self,
        model_tag: str,
        *,
        stress: Optional[StressGrid] = None,
        base_transaction_cost: float = 0.0005,
        base_execution_lag: int = 1,
        base_entry_threshold: float = 0.20,
        hold_rule: str = "signal_until_change",
        mode: str = "long_only",
        include_reliability_points: bool = True,
    ):
        '''
        Evaluate a single registered model and optionally run stress tests.
        Returns a dict you can save to JSON / parquet.
        '''
        if model_tag not in self._models:
            raise KeyError(f"Model '{model_tag}' not registered.")

        m = self._models[model_tag]
        proba_cols = m["proba_cols"]

        # Probability quality (DEV and TEST)
        dev_proba = self._extract_proba(m["preds_dev"], proba_cols)
        test_proba = self._extract_proba(m["preds_test"], proba_cols)

        dev_pq = self.probability_quality_metrics(m["y_dev"], dev_proba)
        test_pq = self.probability_quality_metrics(m["y_test"], test_proba)

        reli = None
        if include_reliability_points:
            reli = {
                "dev_topclass": self.reliability_topclass_points(m["y_dev"], dev_proba),
                "test_topclass": self.reliability_topclass_points(m["y_test"], test_proba),
            }

        # Trading performance on TEST predictions (your backtest uses probabilities)
        bt_test, bt_metrics = self._run_backtest(
            m["preds_test"],
            transaction_cost=base_transaction_cost,
            execution_lag=base_execution_lag,
            entry_threshold=base_entry_threshold,
            hold_rule=hold_rule,
            mode=mode,
        )
        eff = self.turnover_efficiency(bt_test, bt_metrics)

        out = {
            "stock": self.stock_name,
            "model_tag": model_tag,
            "prob_quality_dev": dev_pq,
            "prob_quality_test": test_pq,
            "reliability": reli,
            "bt_metrics_test": bt_metrics,
            "turnover_efficiency_test": eff,
        }

        # Stress tests (sensitivity to execution assumptions)
        if stress is not None:
            rows = []
            for tc in stress.transaction_costs:
                for lag in stress.execution_lags:
                    for th in stress.entry_thresholds:
                        bt_df, met = self._run_backtest(
                            m["preds_test"],
                            transaction_cost=tc,
                            execution_lag=lag,
                            entry_threshold=th,
                            hold_rule=hold_rule,
                            mode=mode,
                        )
                        met2 = dict(met)
                        met2.update(self.turnover_efficiency(bt_df, met))
                        met2.update({
                            "transaction_cost": float(tc),
                            "execution_lag": int(lag),
                            "entry_threshold": float(th),
                        })
                        rows.append(met2)

            out["stress_test"] = pd.DataFrame(rows).sort_values(
                ["transaction_cost", "execution_lag", "entry_threshold"]
            ).reset_index(drop=True)

        return out

    def evaluate_all_models(
        self,
        *,
        stress: Optional[StressGrid] = None,
        base_transaction_cost: float = 0.0005,
        base_execution_lag: int = 1,
        base_entry_threshold: float = 0.20,
        hold_rule: str = "signal_until_change",
        mode: str = "long_only",
    ):
        '''
        Evaluate all registered models.

        Returns:
          - summary_df: one-row-per-model table for quick ranking/filtering
          - details: dict keyed by model_tag with full outputs (including stress df)
        '''
        details: Dict[str, Any] = {}
        summary_rows: List[Dict[str, Any]] = []

        for tag in self._models.keys():
            res = self.evaluate_model(
                tag,
                stress=stress,
                base_transaction_cost=base_transaction_cost,
                base_execution_lag=base_execution_lag,
                base_entry_threshold=base_entry_threshold,
                hold_rule=hold_rule,
                mode=mode,
                # Full summaray or only main points
                include_reliability_points=False,
            )
            details[tag] = res

            # Flatten into a compact summary row
            row = {
                "stock": self.stock_name,
                "model_tag": tag,

                # Probability quality
                "dev_log_loss": res["prob_quality_dev"]["log_loss"],
                "dev_brier": res["prob_quality_dev"]["brier"],
                "dev_ece": res["prob_quality_dev"]["ece_topclass"],

                "test_log_loss": res["prob_quality_test"]["log_loss"],
                "test_brier": res["prob_quality_test"]["brier"],
                "test_ece": res["prob_quality_test"]["ece_topclass"],

                # Trading metrics (test)
                "test_sharpe": res["bt_metrics_test"]["sharpe"],
                "test_annual_return": res["bt_metrics_test"]["annual_return"],
                "test_annual_vol": res["bt_metrics_test"]["annual_volatility"],
                "test_max_dd": res["bt_metrics_test"]["max_drawdown_log"],
                "test_avg_turnover_y": res["bt_metrics_test"]["avg_turnover_per_year"],
                "test_pct_in_mkt": res["bt_metrics_test"]["pct_time_in_market"],

                # Turnover efficiency
                "test_return_per_turnover": res["turnover_efficiency_test"]["return_per_turnover"],
                "test_sharpe_per_turnover": res["turnover_efficiency_test"]["sharpe_per_turnover"],
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows).sort_values(
            ["test_sharpe", "test_brier"], ascending=[False, True]
        ).reset_index(drop=True)

        return summary_df, details
    
# NOTE - CONTEXT
# Context / Interpretation Guide for Model Comparison Table
#
# This table compares two models (XGBoost vs LGBM) per asset across:
# (1) Probability quality (beliefs): dev_log_loss, dev_brier, dev_ece, test_log_loss
# (2) Economic outcomes (belief → policy → execution): turnover, drawdown, exposure,
#     return_per_turnover, sharpe_per_turnover.
# This is NOT a prediction leaderboard; it evaluates how probabilistic beliefs
# translate into trading efficiency and risk-adjusted economics.
#
# Probability metrics (lower is better):
# - log_loss / brier: accuracy + confidence of probabilities
# - ECE: calibration quality (knowing when the model is right); critical for
#   probability-weighted trading policies. Similar log loss with different ECE
#   implies similar hit rates but very different confidence quality.
#
# Dev vs test gaps indicate generalisation:
# - Moderate, consistent degradation is expected.
# - Large dev→test blowups signal regime overfitting, independent of returns.
#
# Economic metrics (test set):
# - avg_turnover_y: trading intensity; lower = more robust to costs.
# - return_per_turnover: return efficiency per unit of trading.
# - sharpe_per_turnover: key summary metric (risk-adjusted efficiency per trade).
#   Small but consistent differences here are meaningful.
#
# Stablecoins should look "bad":
# - Little to no directional signal → losses after costs are expected.
# - Better models trade less, lose less, and stay calibrated.
#   Sensible behaviour here increases pipeline credibility.
#
# Valid conclusions:
# - Probability quality (esp. ECE) matters for economic efficiency.
# - Models can have similar Sharpe but very different trading efficiency.
# - No single model dominates; performance is asset- and regime-dependent.
# - Results show reasonable generalisation and no obvious leakage.
#
# Next steps:
# - Aggregate sharpe_per_turnover across assets (mean/median, win-rates).
# - Compare raw vs calibrated probabilities (ECE ↓, efficiency ↑/stable).
# - For new models (e.g. QML), similar Sharpe + lower turnover + better ECE
#   is a meaningful contribution.

