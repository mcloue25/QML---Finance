import os 

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression



class FeatureBuilder:
    ''' Class for downloading videos from URLs (YouTube) links
    '''
    def __init__(self, csv_path:str, build='core', output_path:str=None):
        self.csv_path = csv_path
        self.build = build
        self.output_path = output_path
        self.df = pd.read_csv(self.csv_path)
        self.df = self.preprocessing()
        self.create_folder(output_path)


    def create_folder(self, folder_name: str):
        os.makedirs(folder_name, exist_ok=True)


    def preprocessing(self):
        '''  
        '''
        # Remove heading data
        self.df = self.df.iloc[1:].reset_index(drop=True)
        self.df.set_index('date', inplace=True)
        # Convert all cols to numeric
        for col in self.df.columns:
            self.df[col] = self.df[col].astype(float)
        return self.df


    def generate_features(self, build='core', print_cols=False):
        ''' Main funcction for building all signals ill be analysing
        Args:
            build (String) : Will be used for whatever kind of feature generation process I want to follow
            print_cols (Bool) : Prints list of columns after all are generated or not
        '''
        # Calculate 1-day log returns to serve as the base signal for all rolling features
        self.get_rolling_return()

        # Calculate rolling mean returns to capture short- and medium-term trend regimes
        self.get_rolling_mean(5)
        self.get_rolling_mean(10)
        self.get_rolling_mean(20)

        # Calculate rolling cumulative returns to measure total directional movement over time windows
        self.calculate_cuimulative_return(5)
        self.calculate_cuimulative_return(10)
        self.calculate_cuimulative_return(20)

        # Calculate rolling return volatility to identify low- and high-risk market regimes
        self.calculate_volatility(20)
        self.calculate_volatility(60)


        # NOTE - Narrowing stock direction down to one number 
        # NOTE - BINARY DIRECTION
        # self.caclulate_stock_direction(5)
        # self.caclulate_stock_direction(10)
        # self.caclulate_stock_direction(20)

        # NOTE - MultiClass
        self.multi_class_action_vol_adj(num_days=5, vol_col="volatility_20d")
        self.multi_class_action_vol_adj(num_days=10, vol_col="volatility_20d")
        self.multi_class_action_vol_adj(num_days=20, vol_col="volatility_20d")


        # Calculate EWMA volatility to emphasize recent shocks and regime transitions
        self.calculate_EWMA_volatility(20)
        self.calculate_EWMA_volatility(60)

        # Estimate rolling regression slope of log prices to quantify trend strength and persistence
        self.rolling_slope(np.log(self.df["adj_close"]), 20)

        # Compute price-to-moving-average ratio to normalize price trends across different price levels
        self.calculate_moving_AVG_ratio(20)
        self.calculate_moving_AVG_ratio(60)

        # Measure volatility of volatility to detect unstable or transitioning risk regimes
        self.calculate_vol_of_vol(20)

        # Compute drawdown from historical peak to capture path-dependent downside risk
        self.calculate_drawdown()

        # Calculate rolling maximum drawdown to quantify worst-case losses over long horizons
        self.calculate_rolling_max_drawdown()

        # Track time since the most recent peak to distinguish recoveries from prolonged drawdowns
        self.calculate_time_since_last_peak()

        # Calculate volatility with only negative values
        self.calculate_downside_volatility(20)

        # skewness captures asymmetry (crash-like negative tail)
        self.calculate_skew(60)

        # Calculating volume
        self.calculate_volume_z(5)
        self.calculate_volume_z(10)
        self.calculate_volume_z(20)


        # NOTE - REMOVE OHLC FROM FEATURE FOR COLINEARITY REASONS
        if print_cols:
            for i in self.df.columns:
                print(i)
            a-b


    def get_rolling_return(self):
        ''' Calculate the log return for 1 day
        '''
        self.df["log_return_1d"] = np.log(self.df["adj_close"] / self.df["adj_close"].shift(1))



    def get_rolling_mean(self, num_days:int =20):
        '''  Get thge rolling returtn over a number of days
            Positive --> trending regime
            Near zero --> sideways
            Negative --> drawdown regime
        Args:
            num_days (Int) : Number of days 
        '''
        self.df[f"rolling_mean_return_{num_days}d"] = (self.df["log_return_1d"].rolling(window=num_days, min_periods=num_days).mean())  


    def caclulate_stock_direction(self, num_days):
        '''  calculates the stock direction over the future num_days days
        '''
        self.df[f"future_log_return_{num_days}d"] = (self.df["log_return_1d"].shift(-1).rolling(window=num_days).sum())
        self.df[f"direction_{num_days}d"] = (self.df[f"future_log_return_{num_days}d"] > 0).astype(int)



    def multi_class_action_vol_adj(self, num_days:int, vol_col="volatility_20d", eps:float = 1e-12):
        ''' 3-class target using volatility-adjusted future returns:
            0 = bottom third, 1 = middle, 2 = top third
        
        Intepretation:
            +0.5 → strong upside given current risk
            −0.5 → strong downside given current risk
            near 0 → noise / hold

        Resulting Action:
            Bottom 33% → 0 = sell
            Middle 33% → 1 = hold
            Top 33% → 2 = buy
        '''
        self.df[f"future_log_return_{num_days}d"] = (self.df["log_return_1d"].shift(-1).rolling(window=num_days).sum())
        score = self.df[f"future_log_return_{num_days}d"] / (self.df[vol_col].abs() + eps)

        q_low = score.quantile(0.33)
        q_high = score.quantile(0.67)
        self.df[f"signal_voladj_{num_days}d"] = pd.cut(
            score,
            bins=[-np.inf, q_low, q_high, np.inf],
            labels=[0, 1, 2],
            include_lowest=True
        ).astype("Int64")


    
    def calculate_cuimulative_return(self, num_days:int =20):
        ''' Calculates the cumulative return over a period of days
        Args:
           num_days (Int) : Number of days we want to calculate feature over 
        '''
        self.df[f"rolling_cum_return_{num_days}d"] = (self.df["log_return_1d"].rolling(window=num_days, min_periods=num_days).sum())



    def calculate_volatility(self, num_days:int =20):
        ''' Calculates the cumulative return over a period of days
        Args:
           num_days (Int) : Number of days we want to calculate feature over 
        '''
        self.df[f"volatility_{num_days}d"] = (self.df["log_return_1d"].rolling(window=num_days, min_periods=num_days).std())



    def calculate_EWMA_volatility(self, num_days:int =20):
        ''' Calculates the EWMA volatility over a period of num_Days
        Args:
           num_days (Int) : Number of days we want to calculate feature over 
        Returns:
            Adds column to df
        '''
        self.df[f"volatility_ewma_{num_days}d"] = (self.df["log_return_1d"].ewm(span=num_days, adjust=False).std())
        


    def calculate_momentum(self, num_days:int=20):
        ''' Positive --> uptrend
            Near zero --> sideways
            Negative --> drawdown
            1 → bullish regime
            < 1 → bearish regime
        '''
        self.df[f"momentum_{num_days}d"] = (self.df["log_return_1d"].rolling(window=num_days, min_periods=num_days).sum())




    def rolling_slope(self, series, window=20):
        ''' Calculates the linear regression slope of the 
        Args:

        '''
        slopes = np.full(len(series), np.nan)
        x = np.arange(window).reshape(-1, 1)

        for i in range(window, len(series)):
            y = series.iloc[i-window:i].values.reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            slopes[i] = model.coef_[0][0]

        self.df[f"momentum_slope_{window}d"] = slopes


    def calculate_moving_AVG_ratio(self, num_days):
        ''' Calculates the moving average over a period of num_days
        Args:
           num_days (Int) : Number of days we want to calculate feature over 
        '''
        self.df[f"ma_{num_days}d"] = self.df["adj_close"].rolling(window=num_days, min_periods=num_days).mean()
        self.df[f"price_ma_ratio_{num_days}d"] = self.df["adj_close"] / self.df[f"ma_{num_days}d"]
        self.df[f"price_ma_ratio_{num_days}d_centered"] = self.df[f"price_ma_ratio_{num_days}d"] - 1.0

    
    def calculate_vol_of_vol(self, num_days):
        '''   
        '''
        self.df[f"vol_of_vol_{num_days}d"] = (
            self.df[f"volatility_{num_days}d"]
            .rolling(window=num_days, min_periods=num_days)
            .std()
        )

    
    def calculate_drawdown(self):
        '''  
        '''
        running_max = self.df["adj_close"].cummax()
        self.df["drawdown"] = self.df["adj_close"] / running_max - 1.0


    def calculate_rolling_max_drawdown(self):
        '''  
        '''
        self.df["max_drawdown_252d"] = self.df["drawdown"].rolling(window=252, min_periods=252).min()



    def calculate_time_since_last_peak(self):
        ''' 
        '''
        self.df.reset_index(inplace=True)
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df = self.df.dropna(subset=["date"]).sort_values("date")

        # Identify peaks and forward-fill the last peak date
        running_max = self.df["adj_close"].cummax()
        is_peak = self.df["adj_close"].eq(running_max)
        last_peak_date = self.df["date"].where(is_peak).ffill()

        # Compute days since last peak
        self.df["time_since_peak"] = (self.df["date"] - last_peak_date).dt.days
        self.df.set_index('date', inplace=True)

    
    def calculate_downside_volatility(self, num_days:int, min_neg_days:int=5):
        ''' Calculate volatility using only negative return days inside each rolling window
        '''
        neg_returns = self.df["log_return_1d"].where(self.df["log_return_1d"] < 0)
        rolling_std = neg_returns.rolling(window=num_days).std()
        count_neg = neg_returns.rolling(window=num_days).count()
        self.df[f"downside_vol_{num_days}d"] = rolling_std.where(count_neg >= min_neg_days)
        # Handling NaNs outside feature engineering
        # self.df[f"downside_vol_{num_days}d"] = self.df[f"downside_vol_{num_days}d"].fillna(0.0)



    def calculate_skew(self, num_days):
        ''' skewness captures asymmetry (crash-like negative tail)
        '''
        self.df[f"skew_{num_days}d"] = self.df["log_return_1d"].rolling(window=num_days, min_periods=num_days).skew()


    def calculate_dollar_volume(self):
        ''' Accounts for price level
        '''
        self.df["dollar_volume"] = self.df["adj_close"] * self.df["volume"]


    def calculate_volume_z(self, num_days):
        ''' Calculate normalised volume relative to recent history
        '''
        log_vol = np.log1p(self.df["volume"])
        m = log_vol.rolling(num_days, min_periods=num_days).mean()
        s = log_vol.rolling(num_days, min_periods=num_days).std()
        self.df[f"log_volume_z_{num_days}d"] = (log_vol - m) / s