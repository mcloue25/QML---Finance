import os 
import json

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

from ..Plotting import Plotter

# from ...utilities import load_JSON_object, pretty_print_json


# METRIC_SPECS = { 
#     # Trading efficiency (PRIMARY)
#     # Risk-adjusted return per unit of trading – your best single summary metric
#     "test_sharpe_per_turnover": ("max", 3.0),
#     # Raw efficiency of trading activity
#     "test_return_per_turnover": ("max", 2.0),
#     # Classic risk-adjusted performance
#     "test_sharpe": ("max", 1.5),
#     # Absolute performance (kept lower weight to avoid leverage bias)
#     "test_annual_return": ("max", 1.0),

#     # Risk & cost controls
#     # Less negative drawdown is better
#     "test_max_dd": ("max", 1.0),
#     # Lower volatility = more robust
#     "test_annual_vol": ("min", 0.75),
#     # Lower trading intensity = more cost-robust
#     "test_avg_turnover_y": ("min", 0.75),

    
#     # Probability quality ("trust")
#     # Overall probabilistic accuracy
#     "test_log_loss": ("min", 1.25),
#     # Calibration-aware accuracy
#     "test_brier": ("min", 0.75),
#     # Knowing when you're right (critical for probability-weighted policies)
#     "test_ece": ("min", 0.75),
# }



'''   
ToDo:
Optional tweaks depending on your philosophy
If you want a more conservative allocator (less blow-ups)
    Increase risk weights a bit:
    test_max_dd: 1.0 → 1.5
    test_annual_vol: 0.75 → 1.0
    test_avg_turnover_y: 0.75 → 1.0

If you want a probability-driven allocator (sizing-heavy / PM cares about calibration)
    Increase calibration:
    test_ece: 0.75 → 1.25
    test_log_loss: 1.25 → 1.5

If you want pure “alpha hunting”
    Decrease vol and bump return:
    test_annual_return: 1.0 → 1.5
    maybe reduce test_annual_vol to 0.5

'''

@staticmethod
def pretty_print_json(dict_object: object):
    ''' Makes JSON object prettier for being printed 
    '''
    json_object = json.dumps(dict_object)
    parsed_json = json.loads(json_object) 
    print(json.dumps(parsed_json, indent=4))

@staticmethod
def save_JSON_object(json_object: object, json_path: str):
    ''' Saves a JSON Object to a path
    '''
    with open(json_path, 'w') as outfile:
        json.dump(json_object, outfile)

@staticmethod
def load_JSON_object(json_path: str):
    ''' Loads JSON object from a file
    '''
    with open(json_path, "r") as f:
        json_data = json.load(f)
        return json_data
    





class PortfolioManager():
    ''' The function of this class is to 
            1) Take all best performing models + an initial capital & create an optimally diversified portfolio
            2) Being passed a day where trades can be currently being executed evaluate market signals and decide what actions to take for the following day before market close
    
        CONFIDENCE:
            Model Quality / Robustness
            CURRENT Signal Confidence
            Combination of the two

    '''
    def __init__(self):
        pass


    def load_sector_data(self, portfolio_sectors_path):
        ''' Load data on all stock that models were trained for as well as what sector they belong to
        '''
        return load_JSON_object(portfolio_sectors_path)


    # NOTE - ADD THIS LATER ON ONCE IVE GENERATED DATA FOR SPANNING DIFFERRENT MARKETS
    # def set_regime_logic(self, market_regime, asset_class):
    #     '''  
    #     '''
    #     if market_regime == "high_vol":
    #         spec = self.metric_specs["DEFENSIVE_V1"]
    #     elif asset_class == "crypto":
    #         spec = self.metric_specs["CRYPTO_VOL_AWARE_V1"]
    #     else:
    #         spec = self.metric_specs["BALANCED_V1"]


    
    def get_sector_by_symbol(self, sector_data):
        ''' Inverts the architecture of a dict of depth 2
        Args:
            key_features_object (Dict) : Dict where K == Sector && V == Dict where K == Ticker symbol && V == List of core features
        Returns:
            inverted (Dict) : 1D dict where K == ticker symbols && V == Sector they belong to
        '''
        return {symbol: sector for sector, symbols in sector_data.items() for symbol in symbols}
    

    # generate_ranked_model_list
    def analyse_best_performing_models(self, performance_json_path, philosophy_path:str, philosophy:str='BALANCED_V1', show:bool=False, save_path:str=None):
        ''' Function to analyse perfromance metrics 
        '''
        # Load model performance data and trading philosophy for deciding how we want to rank stocks
        perf_data = load_JSON_object(performance_json_path)
        philosophy_data = load_JSON_object(philosophy_path)
        self.metric_specs = philosophy_data[philosophy]

        # Build top performing models DF
        models_df = self.aggregate_top_performing_models(perf_data)

        # Generate ranking using portfolio philosophy
        self.ranked_df = self.generate_rankings(models_df)
        self.ranked_df.set_index('allocation_rank', inplace=True)
        
        # Plotting allocation scores to see if theres any noticable dropoff
        if show:
            print('Ranked Model DataFrame:')
            print(self.ranked_df)
            plot = Plotter()
            plot.line_plot(self.ranked_df, 'allocation_score')
            plot.bar_plot(self.ranked_df, 'allocation_score')
            plot.histogram_plot(self.ranked_df, 'allocation_score')
        
        if save_path:
            self.ranked_df.to_csv(save_path)

        return


    def aggregate_top_performing_models(self, perf_data):
        ''' Aggregate all top performing models for each stock symbol
        '''
        model_df = pd.DataFrame()
        for results_dict in perf_data.values():
            df = pd.DataFrame.from_dict(results_dict["ranked_df"])
            best_row = df.loc[df["model_tag"] == results_dict["best_model"]]
            model_df = pd.concat([model_df, best_row], axis=0)
        model_df = model_df.reset_index(drop=True)
        return model_df
    

    def generate_rankings(self, models_df):
        ''' Function to rank each models performance 

            FOR A START:
                allocation_score = quality_score * signal_confidence 
        '''
        # Remove unneccessary cols
        keep_cols = [i for i in models_df.columns if 'score_' not in i]
        keep_cols.remove('composite_score')
        keep_cols.remove('rank')

        core_cols = models_df[keep_cols]
        ranked_df = core_cols.copy()
        
        score_cols = []
        weights = {}
        for col, (direction, weight) in self.metric_specs.items():
            score_col = f"score_{col}"
            ranked_df[score_col] = self.normalise_series(ranked_df[col], direction=direction)
            score_cols.append(score_col)
            weights[score_col] = weight

        w = pd.Series(weights)
        weighted = ranked_df[score_cols].mul(w, axis=1)
        denom = ranked_df[score_cols].notna().mul(w, axis=1).sum(axis=1)
        ranked_df["quality_score"] = weighted.sum(axis=1) / denom

        # NOTE - Signal confidence - SETTING THIS TO 1 SINCE BOOTSTRAPPED DATA BUT FOR LIVE WILL NEED TO COMPUTE4R SIGNAL QUALITY
        # NOTE - Will need to recompute probabilities
        ranked_df["signal_confidence"] = 1.0

        # Allocation Score
        ranked_df["allocation_score"] = ranked_df["quality_score"] * ranked_df["signal_confidence"]

        # Rank rows
        ranked_df["allocation_rank"] = ranked_df["allocation_score"].rank(ascending=False, method="dense")
        ranked_df = ranked_df.sort_values("allocation_rank").reset_index(drop=True)
        return ranked_df
        


    def normalise_series(self, col: pd.Series, direction: str, winsor_p: float = 0.01):
        ''' Normalise a metric column to [0, 1] so that higher = better.
        Args:
            s (pd.Series) : metric values across models (one stock)
            direction (str) : "max" or "min"
            winsor_p (float) : quantile for clipping outliers
        Returns:
            pd.Series: normalized scores in [0, 1]
        '''
        # Convert to numeric, coerce errors to NaN
        col = pd.to_numeric(col, errors="coerce")
        s = col.copy()  # keep original index/shape

        # Use valid values to compute bounds
        valid = s.dropna()
        if valid.empty:
            return pd.Series(np.nan, index=s.index)

        # winsorization to limit influence of extreme value by clipping
        if winsor_p is not None and winsor_p > 0:
            lo = valid.quantile(winsor_p)
            hi = valid.quantile(1 - winsor_p)
            s = s.clip(lower=lo, upper=hi)
            valid = s.dropna()

        mn, mx = valid.min(), valid.max()

        # If no variation, return neutral scores
        if np.isclose(mx, mn):
            return pd.Series(0.5, index=s.index)

        # Normalisation & Return scaled winner
        scaled = (s - mn) / (mx - mn)
        if direction == "max":
            return scaled
        elif direction == "min":
            return 1.0 - scaled
        else:
            raise ValueError(f"Unknown direction: {direction}")
        



    def genereate_portfolio(self, capital:float, tactic:str ='BASIC', conf_amp:float=5.0, convexity_belief:float =2.0, max_weight:float= 0.10, min_weight:float= 0.005, target_vol:float= 0.20, show:bool=False):
        ''' Function for taking the ranked DF of stock models & generating a portfolio of stocks to invest ion based on their rankings
        Args:
            capital (Float) : Investment capital to disperse amongst stocks
            tactic (String) : Portfolio generation tactic
            conf_amp (Float) : How aggressively I want to favor the top ranked models
            convexity_belief (Float) : How much do I believe the top ranked ideas are disproportionately better
            max_weight (Float) : MAx position size for a single holding
            min_weight (Float) : Min position cutoff
            target_vol (Float) : For scaling down volatile assets
        Returns:
            df (DataFrame) : DF with position sizes based on passed tactic and thresholds
        '''
        df = self.ranked_df.sort_values("allocation_score", ascending=False)
        df = df[df["allocation_score"] > 0].copy()
        scores = df["allocation_score"].values

        match tactic:
            case "BASIC":
                weights = scores / scores.sum()

            case "SOFTMAX":
                # numerical stability
                scores = scores - scores.max()
                weights = np.exp(conf_amp * scores)
                weights = weights / weights.sum()

            case "POWER_LAW":
                weights = scores ** convexity_belief
                weights = weights / weights.sum()

            case _:
                raise ValueError(f"Unknown tactic: {tactic}")


        # Volatility targeting
        vol = df["test_annual_vol"].values
        risk_adj = np.minimum(target_vol / vol, 1.0)
        weights = weights * risk_adj
        weights = weights / weights.sum()

        # Max position cap
        weights = np.minimum(weights, max_weight)
        weights = weights / weights.sum()

        # Min position cutoff
        mask = weights >= min_weight
        weights = weights * mask
        weights = weights / weights.sum()

        # Conversion of positions to euro
        df["weight"] = weights
        df["euro"] = weights * capital

        if show:
            print('Portfolio Distribution:')
            print(df)
            plot = Plotter()
            plot.line_plot(df, 'euro')
            plot.bar_plot(df, 'euro')
            plot.histogram_plot(df, 'euro')
        return df
    

    def assess_portforlio_diversity(self, symbol_sector_dict:dict, position_df:pd.DataFrame, to_json:bool=True, show:bool=False):
        ''' Takes a portfolio state and analyses its diversity between all holdings
        Args:
            sector_data (Dict) : Dict 
            symbol_sector_dict (Dict) : Dict 
            holdings_df (DataFrame) : DF containing all currently held positions
            show (Bool) : Display distribution or not
        '''
        self.holdings_df = position_df.loc[position_df.euro > 0]
        sector_data = [symbol_sector_dict[i] for i in self.holdings_df['stock']]
        self.holdings_df['stock_sector'] = sector_data

        if to_json:
            # self.holdings_dist = {sector : [] for sector in set(self.holdings_df['stock_sector'])}
            # for symbol in self.holdings_df['stock']:
            #     self.holdings_dist[symbol_sector_dict[symbol]].append(symbol)
            # pretty_print_json(self.holdings_dist)

            weights_dict = {}
            holdings_dict = {}
            investment_dict = {}
            weight_exposure_dict = {}
            invest_exposure_dict = {}
            held_sectors = set(self.holdings_df['stock_sector'])
            for sector in held_sectors:
                # Extract distribution of holdings within each sector
                sector_df = self.holdings_df.loc[self.holdings_df['stock_sector'] == sector]
                weights_dict[sector] = list(sector_df['weight'])
                holdings_dict[sector] = list(sector_df['stock'])
                investment_dict[sector] = list(sector_df['euro'])
                # Calculate total weight & investment in each sector
                weight_exposure_dict[sector] = float(sector_df['weight'].sum())
                invest_exposure_dict[sector] = float(sector_df['euro'].sum())
            
            # Store snapshot of portfolio state & save to backend
            snapshot = {
                'ts' : datetime.now(timezone.utc).isoformat(),
                'portfolio_id' : "main",
                'distribution' : holdings_dict,
                'holdings_weight' : weights_dict,
                'holdings_euro' : investment_dict,
                'sector_exposure_euro' : invest_exposure_dict,
                'sector_exposure_weight' : weight_exposure_dict
            }
            self.append_jsonl(output_path="data/json/portfolio/holdings_diversity.jsonl", record=snapshot)

        if show:
            plot = Plotter()
            plot.histogram_plot(self.holdings_df, 'stock_sector')

    
    def plot_model_distribution(self):
        ''' Plot model architecture distribution 
        '''
        plot = Plotter()
        print('self.holdings_df')
        print(self.holdings_df)
        plot.histogram_plot(self.holdings_df, 'model_tag')
        


    
    def append_jsonl(self, output_path: str | Path, record: dict):
        ''' Append a single JSON record to a .jsonl file. Creates the file (and parent dirs) if it does not exist
        Args:
            path: Path to .jsonl file
            record: JSON-serialisable dict
        '''  
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
