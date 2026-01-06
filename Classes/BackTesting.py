import os
import json
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Literal

def pretty_print_json(obj: dict):
    '''Pretty print a dict as JSON.'''
    print(json.dumps(obj, indent=4, default=str))


@dataclass
class BacktestConfig:
    ''' Config for params to run backtest under to simulate market conditions
    '''
    horizon_days: int = 10
    # per unit turnover (e.g., 5 bps)
    transaction_cost: float = 0.0005
    mode: Literal["long_only", "long_short"] = "long_only"
    hold_rule: Literal["signal_until_change", "fixed_horizon"] = "signal_until_change"
    execution_lag: int = 1
    entry_threshold: float = 0.20
    # signal_until_change: position follows latest signal (shifted by 1 day)
    # fixed_horizon: enter position and hold for horizon_days (ignores intermediate signals)


class BackTest:
    ''' Backtesting utility for regime/action predictions (sell/hold/buy) with realistic execution.
    Expected inputs:
      - preds_df: DataFrame indexed by date with at least:
          * y_pred in {0,1,2} (sell/hold/buy)
        Optional (required for probability-weighted sizing):
          * p_class_0, p_class_1, p_class_2

      - prices: Series indexed by date with matching calendar to preds_df

    Outputs:
      - bt_df: DataFrame with returns, positions, turnover, costs, equity curve
      - metrics: dict of risk/return statistics (separate helper)
    '''

    def __init__(self, path: Optional[str] =None, config: Optional[BacktestConfig] =None):
        self.path = path
        self.config = config or BacktestConfig()
        if self.path:
            os.makedirs(self.path, exist_ok=True)


    def create_folder(self, folder_name: str):
        os.makedirs(folder_name, exist_ok=True)
    


    # NOTE - StaticMethods
    @staticmethod
    def ensure_datetime_index(df_or_s: pd.DataFrame | pd.Series):
        ''' Ensures time series index is a DatetimeIndex and sorted chronologically for time-series correctness
        Args:
            df_or_s (DataFrame) : DF contianing feature data & date idnex col
        '''
        if not isinstance(df_or_s.index, pd.DatetimeIndex):
            df_or_s = df_or_s.copy()
            df_or_s.index = pd.to_datetime(df_or_s.index)
        return df_or_s.sort_index()



    @staticmethod
    def compute_log_returns(prices: pd.Series):
        ''' Compute daily log returns
        '''
        return np.log(prices).diff()
    

    # NOTE - Calculating Performance Metrics
    @staticmethod
    def performance_metrics(bt_df:pd.DataFrame, ann_factor:int =252):
        ''' Compute standard performance statistics
        Notes:
          - Using log returns: annual_return approximated by mean(log_ret)*252
          - cum_ret is cumulative log return (equity = exp(cum_ret))
          - drawdown computed in log space (consistent with cum_ret)
        '''
        total_return = float(bt_df["cum_ret"].iloc[-1])
        ann_return = float(bt_df["net_ret"].mean() * ann_factor)
        ann_vol = float(bt_df["net_ret"].std(ddof=0) * np.sqrt(ann_factor))
        sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0

        # Log drawdown from peak cumulative log return
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
    def trade_stats(bt_df:pd.DataFrame, entry_threshold:float =0.10):
        ''' Trade stats for continuous positions by defining a "trade" as being meaningfully invested.
        Entry: position_lag crosses from < threshold to >= threshold
        Exit : position_lag crosses from >= threshold to < threshold
        '''
        pos = bt_df["position_lag"].fillna(0.0).abs()

        in_trade = pos >= entry_threshold
        prev = in_trade.shift(1).fillna(False)

        entries = in_trade & (~prev)
        exits = (~in_trade) & prev
        entry_dates = bt_df.index[entries]
        exit_dates = bt_df.index[exits]

        # Pair entries with the next exit after each entry
        durations = []
        j = 0
        for e in entry_dates:
            while j < len(exit_dates) and exit_dates[j] <= e:
                j += 1
            if j < len(exit_dates):
                durations.append((exit_dates[j] - e).days)
                j += 1

        n_entries = int(entries.sum())
        n_exits = int(exits.sum())
        avg_duration = float(np.mean(durations)) if durations else 0.0

        return {
            "entry_threshold": float(entry_threshold),
            "n_entries": n_entries,
            "n_exits": n_exits,
            "avg_trade_duration_days": avg_duration,
        }


    def baseline_buy_and_hold(self, prices: pd.Series, start: Optional[pd.Timestamp]=None, end: Optional[pd.Timestamp]=None):
        ''' Buy and hold baseline equity curve (always fully invested)
            Shows how trading accounts value changes over time
            Account balance on Y-axis
        Args:
            prices: Price series indexed by date.
            start/end: Optional evaluation window bounds.
        Returns:
            out (DataFrame) : DataFrame indexed by date with:
                - ret: asset daily log return
                - position_lag: fixed exposure (1.0)
                - strategy_ret: position * ret
                - cost: transaction costs (0 for buy/hold)
                - net_ret: strategy_ret - cost
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



    def probability_weighted_position(self,df: pd.DataFrame, threshold:float =0.20, vol_window:int =20, use_vol_scaling:bool =True):
        ''' Long only probability weighted position sizing
        Main idea:
            signal = p_buy - p_sell
            pos = clip(signal, 0, 1)
            + thresholding (ignore weak/conflicted signals)
            + optional volatility scaling (reduce exposure in high vol regimes)
        Args:
            df (DataFrame) : DataFrame containing p_class_2 (buy prob) and p_class_0 (sell prob) and optionally 'ret' if vol scalings enabled
            threshold (Float) : MIN (p_buy - p_sell) required to take exposure
            vol_window (Int) : rolling window for realized volatility estimate
            use_vol_scaling (Bool) : if True, scale exposure down when vol is elevated
        Returns:
            pd.Series of target positions in [0,1]
        '''
        # Net confidence: buy vs sell
        signal = df["p_class_2"] - df["p_class_0"]
        # Base position (long-only) in [0, 1] and filter weak trades
        pos = signal.clip(lower=0.0, upper=1.0)
        pos = pos.where(signal >= threshold, 0.0)

        # Need returns to compute vol scaling
        if use_vol_scaling:
            vol = df["ret"].rolling(vol_window).std()
            target_vol = vol.median()
            # Scale down when vol > target_vol; never lever up above 1.0
            vol_scale = (target_vol / vol).clip(upper=1.0)
            pos = pos * vol_scale
            # Handle early NaNs from rolling vol window
            pos = pos.fillna(0.0)

        return pos



    def apply_hold_rule(self, target_pos: pd.Series):
        ''' Main function for converting target positions into executed target positions based on hold rule
            - signal_until_change:
                Follow the latest target position daily (before lagging).
            - fixed_horizon:
                When a non-zero target position appears, hold it for horizon_days
                regardless of intermediate target changes.
        Args:
            target_pos (pd.col) : column of probability wewighted target positions
        '''
        if self.config.hold_rule == "signal_until_change":
            return target_pos

        # Fixed-horizon: simple state machine
        h = self.config.horizon_days
        pos = target_pos.copy()
        executed = pd.Series(0.0, index=pos.index)

        holding = 0
        current = 0.0
        for i, (dt, p) in enumerate(pos.items()):
            # If currently holding a position, keep it until timer expires
            if holding > 0:
                executed.iloc[i] = current
                holding -= 1
                continue

            # If a new non-zero signal appears - enter and hold for h days
            if p != 0.0:
                current = p
                holding = h
                executed.iloc[i] = current
                holding -= 1
            else:
                # Otherwise remain flat
                current = 0.0
                executed.iloc[i] = 0.0

        return executed



    # NOTE - Main backtest
    def run(self, preds_df: pd.DataFrame, prices: pd.Series):
        ''' Run backtest using model predictions against a price series.
            Key time-series correctness constraint:
            - Position is shifted by 1 day (decision at t executed at t+1)
            - Prevents look-ahead bias and mimics realistic order placement.
        Args:
            preds_df (DataFrame): indexed by date, contains y_pred and/or class probabilities
            prices (Series): indexed by date, price series
        Returns:
            bt_df: DataFrame containing returns, positions, turnover, costs, cum returns
        '''
        # Convert date col to datetime index
        preds_df = self.ensure_datetime_index(preds_df)
        prices = self.ensure_datetime_index(prices)

        # Align both inputs to the same trading calendar
        df = preds_df.copy()
        df = df.join(prices.rename("price"), how="inner")

        # Compute daily log returns
        df["ret"] = self.compute_log_returns(df["price"])
        df = df.dropna(subset=["ret", "y_pred"])

        # Build target position from probabilities (policy layer)
        df["target_position"] = self.probability_weighted_position(df, threshold=self.config.entry_threshold)
        df["target_position"] = self.apply_hold_rule(df["target_position"])
        df["position_lag"] = df["target_position"].shift(self.config.execution_lag).fillna(0.0)

        # Execution lag: today's signal becomes tomorrow's position
        df["position_lag"] = df["target_position"].shift(1).fillna(0.0)

        # Strategy gross returns = exposure * asset return
        df["strategy_ret"] = df["position_lag"] * df["ret"]

        # Turnover = absolute change in exposure (proxy for trading activity)
        df["turnover"] = df["position_lag"].diff().abs().fillna(0.0)

        # Transaction costs proportional to turnover
        df["cost"] = df["turnover"] * self.config.transaction_cost

        # Net returns after costs + cumulative log equity curve
        df["net_ret"] = df["strategy_ret"] - df["cost"]
        df["cum_ret"] = df["net_ret"].cumsum()
        return df

    

    def trade_list(self,bt_df, price_col=None, pos_col=None, entry_threshold: float = 0.10, exit_threshold: float = 0.10):
        ''' Build a trade list from continuous exposure by thresholding exposure
        Args:
            bt_df (DataFrame) : backtest DataFrame output from run()
            price_col (Series) : optional override for price column name
            pos_col (Series) : optional override for position column name
            entry_threshold (Float) : exposure threshold to define entering a trade
            exit_threshold (Float) : currently unused (exit defined by dropping below threshold)
        Returns:
            trades_df (DataFrame) : DF contianing all executed trade info
        '''
        df = bt_df.copy()

        # Use abs exposure so this works even if you later support shorts
        pos = df[pos_col].fillna(0.0).astype(float).abs()
        price = df[price_col].astype(float)

        # Threshold-defined trade state
        in_trade = pos >= entry_threshold
        prev_in_trade = in_trade.shift(1).fillna(False)

        # Crossings define entries/exits
        entries = in_trade & (~prev_in_trade)
        exits = (~in_trade) & prev_in_trade

        entry_dates = df.index[entries]
        exit_dates = df.index[exits]

        # If last trade still open, close at final bar
        if len(entry_dates) > len(exit_dates):
            exit_dates = exit_dates.append(pd.Index([df.index[-1]]))

        trades = []
        j = 0
        exit_dates_list = list(exit_dates)

        # Pair each entry with the next exit after it
        for e in entry_dates:
            while j < len(exit_dates_list) and exit_dates_list[j] <= e:
                j += 1
            if j >= len(exit_dates_list):
                break

            x = exit_dates_list[j]
            j += 1

            entry_px = float(price.loc[e])
            exit_px = float(price.loc[x])
            ret = (exit_px / entry_px) - 1.0

            trades.append({
                "entry_time": e,
                "exit_time": x,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "return": ret,
                # Profit / Loss from each executed trade
                "pnl_pct": ret * 100.0,
                "bars_held": int(df.loc[e:x].shape[0] - 1),
                "avg_exposure": float(pos.loc[e:x].mean()),
                "max_exposure": float(pos.loc[e:x].max()),
            })

        return pd.DataFrame(trades)




    @staticmethod
    def plot_equity_curve(bt_df: pd.DataFrame, title: str = "Strategy Equity Curve", show:bool=False, save_path=None):
        ''' Plot cumulative log returns (equity in log space)
        '''
        plt.figure(figsize=(12, 5))
        plt.plot(bt_df.index, bt_df["cum_ret"], label="Strategy (log cum ret)")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Cumulative log return")
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()


    @staticmethod
    def plot_positions(bt_df: pd.DataFrame, title: str = "Position Over Time", show:bool=False, save_path=None):
        ''' Plot executed position over time (lagged exposure)
        '''
        plt.figure(figsize=(12, 3))
        plt.plot(bt_df.index, bt_df["position_lag"], linewidth=1)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Position")
        plt.tight_layout()
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()



    def save_backtest_artifacts(self, base_dir, run_row, bt_df, trades_df=None):
        ''' Function to save backtest results for visualisation on dashboard
        Args:
            base_dir (String) : Base directory to save all trade and curve info to
            run_row ()
            bt_df (DataFrame) : DF containing backtest results
            trades_df (Bool) : Bool to save trades or not
        Returns:
            runs.parquet: metadata / summary metrics table (append-only)
            curves/{run_id}.parquet: per-day curve data (equity, returns, drawdown, position)
            trades/{run_id}.parquet: optional trade list
        '''
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_row["run_id"]

        #  Save/append run table 
        runs_path = base_dir / "runs.parquet"
        run_df = pd.DataFrame([run_row])

        if runs_path.exists():
            existing = pd.read_parquet(runs_path)
            pd.concat([existing, run_df], ignore_index=True).to_parquet(runs_path, index=False)
        else:
            run_df.to_parquet(runs_path, index=False)

        #  Build curves DF from your bt_df schema 
        df = bt_df.copy()

        # Ensure date column exists (your dates are likely the index)
        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "date"})
            else:
                raise ValueError("bt_df has no 'date' column and index is not a DatetimeIndex.")

         # Daily strategy net return (log)
        df["returns"] = df["net_ret"]
        # convert cumulative log return -> equity curve
        df["equity"] = np.exp(df["cum_ret"])
        # log drawdown
        df["drawdown"] = df["cum_ret"] - df["cum_ret"].cummax()
        # Executed position
        df["position"] = df["position_lag"]

        curve_df = df[["date", "equity", "returns", "drawdown", "position", "run_id"]].copy()
        # convert date to string for parquetjs-lite 
        curve_df["date"] = curve_df["date"].astype(str)
        out_dir = base_dir / str(run_id) / "curves"
        out_dir.mkdir(parents=True, exist_ok=True)


        # NOTE - SAving curves and trdeas to .parquet files
        # Save curve DF
        curve_df.to_parquet(
            out_dir / f"{run_id}.parquet",
            index=False,
            engine="pyarrow",
            compression="snappy",
            version="1.0",
        )

        # Save Trades
        trades_df = trades_df.copy()
        if "date" in trades_df.columns:
            trades_df["date"] = trades_df["date"].astype(str)

        out_dir = base_dir / str(run_id) / "trades"
        out_dir.mkdir(parents=True, exist_ok=True)

        trades_df.to_parquet(
            out_dir / f"{run_id}.parquet",
            index=False,
            engine="pyarrow",
            compression="snappy",
            version="1.0",
        )