import os
import json
from dataclasses import dataclass
from typing import Dict, Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pretty_print_json(obj: dict) -> None:
    '''Pretty print a dict as JSON.'''
    print(json.dumps(obj, indent=4, default=str))


@dataclass
class BacktestConfig:
    horizon_days: int = 10
    # per unit turnover (e.g., 5 bps)
    transaction_cost: float = 0.0005
    mode: Literal["long_only", "long_short"] = "long_only"
    hold_rule: Literal["signal_until_change", "fixed_horizon"] = "signal_until_change"
    # signal_until_change: position follows latest signal (shifted by 1 day)
    # fixed_horizon: enter position and hold for horizon_days (ignores intermediate signals)


class BackTest:
    ''' Backtesting utility for buy/hold/sell ML signals.

    Expected inputs:
      - preds_df: DataFrame indexed by date with at least:
          * 'y_pred' in {0,1,2} (sell/hold/buy)
        Optional columns:
          * 'p_class_0','p_class_1','p_class_2' for probability-based rules later

      - prices: Series indexed by date (same calendar as preds_df). Close is fine.

    Outputs:
      - a backtest DataFrame containing returns, positions, costs, equity curve
      - a metrics dict
    '''

    def __init__(self, path: Optional[str] = None, config: Optional[BacktestConfig] = None):
        self.path = path
        self.config = config or BacktestConfig()
        if self.path:
            os.makedirs(self.path, exist_ok=True)


    # ---------- Data prep ----------
    @staticmethod
    def ensure_datetime_index(df_or_s: pd.DataFrame | pd.Series):
        ''' Data  Prep
        '''
        if not isinstance(df_or_s.index, pd.DatetimeIndex):
            df_or_s = df_or_s.copy()
            df_or_s.index = pd.to_datetime(df_or_s.index)
        return df_or_s.sort_index()


    @staticmethod
    def compute_log_returns(prices: pd.Series):
        ''' Daily log returns. (Works well for additive cumulation.)
        '''
        return np.log(prices).diff()


    def baseline_buy_and_hold(self, prices: pd.Series, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None):
        ''' Construct a buy-and-hold baseline equity curve.
        Args:
            prices (pd.Series) : Price series indexed by date used to compute daily log returns.
            start (pd.Timestamp) : Optional start date for the backtest window. If provided, performance is evaluated only from this date onward.
            end (pd.Timestamp, optional): Optional end date for the backtest window. If provided, performance is evaluated only up to this date.
        Returns:
            pd.DataFrame:
                DataFrame indexed by date containing:
                - ret: daily log returns of the asset
                - position_lag: constant exposure (1.0 = fully invested)
                - strategy_ret: daily strategy returns
                - cost: transaction costs (zero for buy-and-hold)
                - net_ret: returns after costs
                - cum_ret: cumulative log return
        '''
        # Chronological ordering and DatetimeIndex
        prices = self.ensure_datetime_index(prices)

        # Optionally restrict to a sub-period
        if start or end:
            prices = prices.loc[start:end]
        # Compute daily log returns
        ret = self.compute_log_returns(prices).dropna()
        # Initialise output DataFrame aligned with return dates
        out = pd.DataFrame(index=ret.index)
        # Buy-and-hold: always fully invested
        out["ret"] = ret
        out["position_lag"] = 1.0
        # Strategy return equals asset return (no timing or leverage)
        out["strategy_ret"] = out["position_lag"] * out["ret"]
        # No transaction costs for buy-and-hold
        out["cost"] = 0.0
        # Net returns equal gross returns
        out["net_ret"] = out["strategy_ret"]
        # Cumulative log return (additive over time)
        out["cum_ret"] = out["net_ret"].cumsum()

        return out



    # def map_actions_to_position(self, y_pred: pd.Series) -> pd.Series:
    #     ''' BINARY EXPOSURE
    #         Signal --> position  
    #         Map action classes to target position.
    #         Default: long-only (sell/hold => flat, buy => long).
    #         Optional: long-short (sell => -1, hold => 0, buy => +1).
    #     '''
    #     if self.config.mode == "long_only":
    #         mapping = {0: 0.0, 1: 0.0, 2: 1.0}
    #     else:
    #         mapping = {0: -1.0, 1: 0.0, 2: 1.0}
    #     return y_pred.map(mapping).astype(float)



    def probability_weighted_position(self,df: pd.DataFrame, threshold: float = 0.20, vol_window: int = 20, use_vol_scaling: bool = True):
        ''' Long-only probability-weighted position sizing.
            - signal = p_buy - p_sell
            - pos = clip(signal, 0, 1)
            - threshold: ignore weak signals
            - optional vol scaling: reduce exposure when volatility is high
        '''
        required = {"p_class_2", "p_class_0"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for prob sizing: {sorted(missing)}")
        # Net confidence: buy vs sell
        signal = df["p_class_2"] - df["p_class_0"]
        # Base position (long-only) in [0, 1]
        pos = signal.clip(lower=0.0, upper=1.0)
        # Filter weak trades
        pos = pos.where(signal >= threshold, 0.0)

        if use_vol_scaling:
            # Need returns to compute vol scaling
            if "ret" not in df.columns:
                raise ValueError("Column 'ret' required for vol scaling (compute returns before calling).")

            vol = df["ret"].rolling(vol_window).std()
            target_vol = vol.median()

            # Scale down when vol > target_vol; never lever up above 1.0
            vol_scale = (target_vol / vol).clip(upper=1.0)
            pos = pos * vol_scale

            # Handle early NaNs from rolling vol window
            pos = pos.fillna(0.0)

        return pos



    def apply_hold_rule(self, target_pos: pd.Series) -> pd.Series:
        ''' Convert target positions to executed positions depending on hold rule.
            - signal_until_change: follow latest signal daily
            - fixed_horizon: enter and hold for horizon_days after a non-zero signal
        '''
        if self.config.hold_rule == "signal_until_change":
            return target_pos

        # fixed_horizon: very simple implementation
        h = self.config.horizon_days
        pos = target_pos.copy()
        executed = pd.Series(0.0, index=pos.index)

        holding = 0
        current = 0.0
        for i, (dt, p) in enumerate(pos.items()):
            if holding > 0:
                executed.iloc[i] = current
                holding -= 1
                continue

            # If new signal is non-zero, enter for h days; otherwise stay flat
            if p != 0.0:
                current = p
                holding = h
                executed.iloc[i] = current
                holding -= 1
            else:
                current = 0.0
                executed.iloc[i] = 0.0

        return executed




    # NOTE - Main backtest
    def run(self, preds_df: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        ''' Run backtest using preds_df actions against a price series.

            Notes for horizon=10:
            - Prediction at time t must only use info available at t.
            - We shift positions by 1 day to simulate entering at t+1 (no look-ahead).
        '''
        preds_df = self.ensure_datetime_index(preds_df)
        prices = self.ensure_datetime_index(prices)

        # Align to common dates
        df = preds_df.copy()
        df = df.join(prices.rename("price"), how="inner")

        # Returns
        df["ret"] = self.compute_log_returns(df["price"])
        df = df.dropna(subset=["ret", "y_pred"])

        # Positions
        # df["target_position"] = self.map_actions_to_position(df["y_pred"])
        df["target_position"] = self.probability_weighted_position(df)
        df["target_position"] = self.apply_hold_rule(df["target_position"])

        # Shift to avoid look-ahead: position decided at t applied from t+1
        df["position_lag"] = df["target_position"].shift(1).fillna(0.0)

        # Strategy gross return
        df["strategy_ret"] = df["position_lag"] * df["ret"]

        # Transaction costs on turnover (absolute change in position)
        df["turnover"] = df["position_lag"].diff().abs().fillna(0.0)
        df["cost"] = df["turnover"] * self.config.transaction_cost

        # Net return + equity curve (log space)
        df["net_ret"] = df["strategy_ret"] - df["cost"]
        df["cum_ret"] = df["net_ret"].cumsum()

        return df

    # ---------- Metrics ----------
    @staticmethod
    def performance_metrics(bt_df: pd.DataFrame, ann_factor: int = 252) -> Dict[str, float]:
        ''' Compute performance metrics from the backtest DataFrame.
        '''
        total_return = float(bt_df["cum_ret"].iloc[-1])
        ann_return = float(bt_df["net_ret"].mean() * ann_factor)
        ann_vol = float(bt_df["net_ret"].std(ddof=0) * np.sqrt(ann_factor))
        sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0

        # Max drawdown computed on cumulative log return curve
        dd = bt_df["cum_ret"] - bt_df["cum_ret"].cummax()
        max_dd = float(dd.min())

        avg_turnover = float(bt_df["turnover"].mean() * ann_factor)
        hit_rate = float((bt_df["net_ret"] > 0).mean())

        pct_in_market = float((bt_df["position_lag"].abs() > 0).mean())

        return {
            "total_log_return": total_return,
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe": sharpe,
            "max_drawdown_log": max_dd,
            "avg_turnover_per_year": avg_turnover,
            "hit_rate": hit_rate,
            "pct_time_in_market": pct_in_market,
        }
    

    @staticmethod
    def trade_stats(bt_df: pd.DataFrame):
        ''' Basic trade stats (works best for long-only).
            A "trade" starts when position goes 0->1 (or 0->-1) and ends when back to 0.
        '''
        pos = bt_df["position_lag"].fillna(0.0)
        changes = pos.diff().fillna(0.0)

        entries = (changes != 0) & (pos != 0)
        exits = (changes != 0) & (pos == 0)

        n_entries = int(entries.sum())
        n_exits = int(exits.sum())

        # Avg holding length (in days) for completed trades
        entry_idx = list(bt_df.index[entries])
        exit_idx = list(bt_df.index[exits])

        # Pair trades in order (simple pairing)
        durations = []
        if entry_idx and exit_idx:
            j = 0
            for e in entry_idx:
                while j < len(exit_idx) and exit_idx[j] <= e:
                    j += 1
                if j < len(exit_idx):
                    durations.append((exit_idx[j] - e).days)
                    j += 1

        avg_duration = float(np.mean(durations)) if durations else 0.0

        return {
            "n_entries": n_entries,
            "n_exits": n_exits,
            "avg_trade_duration_days": avg_duration,
        }








    @staticmethod
    def plot_equity_curve(bt_df: pd.DataFrame, title: str = "Strategy Equity Curve") -> None:
        '''Plot cumulative log returns.'''
        plt.figure(figsize=(12, 5))
        plt.plot(bt_df.index, bt_df["cum_ret"], label="Strategy (log cum ret)")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Cumulative log return")
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_positions(bt_df: pd.DataFrame, title: str = "Position Over Time") -> None:
        '''Plot held position over time.'''
        plt.figure(figsize=(12, 3))
        plt.plot(bt_df.index, bt_df["position_lag"], linewidth=1)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Position")
        plt.tight_layout()
        plt.show()


    def save(self, bt_df: pd.DataFrame, metrics: Dict[str, float], name: str) -> None:
        '''Save backtest outputs to disk (parquet + json).'''
        if not self.path:
            raise ValueError("BackTest was initialized without a path; cannot save.")

        bt_path = os.path.join(self.path, f"{name}_bt.parquet")
        metrics_path = os.path.join(self.path, f"{name}_metrics.json")

        bt_df.to_parquet(bt_path)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        print(f"Saved:\n  {bt_path}\n  {metrics_path}")
