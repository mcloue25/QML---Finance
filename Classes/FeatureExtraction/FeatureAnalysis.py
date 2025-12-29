import os
import glob
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from statsmodels.stats.outliers_influence import variance_inflation_factor


RANDOM_STATE=42

FEATURE_FAMILIES = {
    "trend_short": [
        "rolling_mean_return_5d",
        "rolling_mean_return_10d",
    ],

    "trend_medium": [
        "rolling_mean_return_20d",
        "price_ma_ratio_20d_centered",
        "price_ma_ratio_60d_centered",
    ],

    "volatility_level": [
        "volatility_20d",
        "volatility_60d",
    ],

    "volatility_structure": [
        "vol_of_vol_20d",
        "downside_vol_20d",
    ],

    "path_dependence": [
        "drawdown",
        "max_drawdown_252d",
        "time_since_peak",
    ],

    "distribution_shape": [
        "skew_60d",
    ],

    "volume": [
        "log_volume_z_20d",
    ],
}



FEATURE_ROLES = {
    "rolling_mean_return_5d": "trend",
    "rolling_mean_return_20d": "trend",
    "price_ma_ratio_20d_centered": "trend_confirmation",

    "volatility_20d": "risk_level",
    "vol_of_vol_20d": "risk_instability",
    "downside_vol_20d": "downside_risk",

    "drawdown": "regime_state",
    "max_drawdown_252d": "regime_state",
    "time_since_peak": "regime_state",

    "log_volume_z_20d": "participation",
    "skew_60d": "tail_risk",
}



def pretty_print_json(dict_object: object):
    ''' Makes JSON object prettier for being printed 
    '''
    json_object = json.dumps(dict_object)
    parsed_json = json.loads(json_object) 
    print(json.dumps(parsed_json, indent=4))






class FeatureAnalyser:
    '''
    Analyses the outputted results from the 
        - coefficient importance (L1 logistic)
        - permutation importance
        - correlation matrix
        - VIF scores
    Plan:
        1) permutation importance to see if each feature is actually demonstratibly predictive - Does it actually matter
        2) Coefficient Importance - In what direction does it act (Bearish / Bullish)
        3) C0orrelation Matrix - What features are saying the same thing?
        4) Is this feature redundant given the others?

    '''
    def __init__(self, results_dir:str, hist_path:str, stock:str, run_ids:list[str], top_k:int, show_results:bool=False):
        self.results_dir = results_dir
        self.historical_data_path = hist_path
        self.hist_df = pd.read_csv(self.historical_data_path)
        self.hist_df.set_index('date', inplace=True)
        self.stock = stock
        self.run_ids = run_ids
        self.top_k = top_k
        self.show_results = show_results
        # Generates stable core features that have predicive stability between different runs 
        self.generate_core_features(file_pattern="*_perm_importance.csv")

        self.analyse_correlation_matrix(self.core_features)
        


    def load_importance_files(self, results_dir: str, pattern: str = "*_perm_importance.csv"):
        ''' Loads multiple permutation-importance CSVs and returns a combined table:
            rows = features
            cols = run_id (e.g., AAPL_5d, AAPL_10d, etc.)
        '''
        # Recursivly look through all subfolders for perm_importance files
        paths = glob.glob(os.path.join(results_dir, "**", pattern), recursive=True)
        runs = {}
        for p in paths:
            stock = os.path.basename(os.path.dirname(os.path.dirname(p)))
            horizon = os.path.basename(os.path.dirname(p))
            # run_id = f"{stock}_{horizon}"
            s = pd.read_csv(p, index_col=0).iloc[:, 0]
            runs[f"{stock}_{horizon}"] = s

        return pd.DataFrame(runs)
    

    def generate_core_features(self, file_pattern):
        ''' 
        '''
        # Load all files analysing 5d / 10d / 20d importance 
        self.perm_df = self.load_importance_files(self.results_dir, pattern=file_pattern)
        
        # self.stab = self.stability_table(self.perm_df, k=10)
        # Compare commonly occuring features between runs to look for stabile predicitve features
        self.compare_horizons(self.perm_df, self.run_ids, self.top_k, show_results=False)
        self.core_features = self.extract_core_features()



    def compare_horizons(self, perm_df: pd.DataFrame, run_ids: list[str], k: int = 10, show_results=False):
        ''' Simple comparison summary: top-k overlaps between runs.
            > 0.6	Very strong stability
            0.4 – 0.6	Moderate stability
            < 0.3	Weak / unstable
        Args:
            perm_df (DataFrame) : DF containing all permuatation info
            run_ids (List) : List of all runs we want to compare horizons against
            k (Int) : Top k features wwe want to extract
        Returns:
            horizon_df (DataFrame) : DF contaning 
        '''
        top = {rid: set(perm_df[f'{self.stock}_{rid}'].sort_values(ascending=False).head(k).index) for rid in run_ids}
        rows = []
        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                a, b = run_ids[i], run_ids[j]
                inter = len(top[a] & top[b])
                union = len(top[a] | top[b])
                rows.append({"run_a": a, "run_b": b, "topk_intersection": inter, "topk_jaccard": inter / union if union else 0})
        
        self.horizon_df = pd.DataFrame(rows).sort_values("topk_jaccard", ascending=False)
        if show_results:
            print('Horizon DF:')
            print(self.horizon_df)

    

    def extract_core_features(self):
        '''  Using the results from compare_horizons() extract the core feature set
        '''
        core = (self.perm_df.rank(ascending=False).le(10).sum(axis=1))
        core_features = core[core == 3].index.tolist()
        if self.show_results:
            print('Core Features:')
            print(core)
        return core_features
    


    def analyse_correlation_matrix(self, core_features, target_col='BTC-USD_signal_voladj_10d'):
        '''  Using the core features anayse correlation between candidate features
        '''
        corr_df, abs_corr, similar_pairs = self.build_correlation_DF_and_sim_scores(core_features)
        
        perm_scores = self.perm_df[target_col]
        # Get best performing emtric per trend family
        best_per_family = self.select_best_features_by_family(perm_scores, FEATURE_FAMILIES)
        # Compare relative importance to top permutation feature
        relative_importance_dict = self.analyse_permutation_importance(best_per_family, target_col)

        # Correlation cross check
        final_features = [v["feature"] for v in best_per_family.values()]
        kept_features, dropped_features = self.prune_correlated_features(df=self.hist_df, features=final_features, perm_scores=perm_scores, corr_threshold=0.8)

        # Check coefficient makes economic sense
        coef_analysis = self.analyse_coefficient_consistency(final_features, results_dir=self.results_dir)
        print("-"*40)
        print('coef_analysis:')
        print(coef_analysis)

        # Prep X & Y for testing using final feature set
        target = target_col.replace(f'{self.stock}_', '')
        model_df = self.hist_df.dropna(subset=[target])
        X = model_df[final_features].dropna()
        y = model_df.loc[X.index, target]

        # NOTE - Assess permutation stability across different horizons
        model = Pipeline([("scaler", StandardScaler()),("clf", LogisticRegression(penalty="l1", solver="saga", max_iter=500))])
        self.stability_df = self.permutation_stability(model=model, X=X, y=y, n_splits=5, n_repeats=5)
        self.stability_df["stability_class"] = pd.cut(
            self.stability_df["stability_ratio"],
            bins=[-np.inf, 1, 3, np.inf],
            labels=["drop", "conditional", "keep"]
        )


        self.role_df = self.role_consistency_table(perm_df=self.perm_df, stock_name = self.stock, features=final_features, horizons=[5, 10, 20])
        
        if self.show_results:
            pretty_print_json(relative_importance_dict)
            print("-"*40)
            print("KEPT FEATURES:")
            print(kept_features)

            print("\nDROPPED FEATURES:")
            for k, v in dropped_features.items():
                print(k, "-->", v)

            print('-'*40)
            print('Final Result')
            print("-"*40)
            print('STABILITY DF:')
            print(self.stability_df)

            print("-"*40)
            print('ROLE DF:')
            print(self.role_df)
            
        

    def analyse_coefficient_consistency(self, features: list[str], results_dir: str, horizons: list[int] = [5, 10, 20]):
        ''' Analyse coefficient sign and magnitude consistency across horizons.
        '''
        rows = []
        for f in features:
            coefs = {}
            for h in horizons:
                s = pd.read_csv(f"{results_dir}signal_voladj_{h}d/{self.stock}_coef_importance.csv", index_col=0).iloc[:, 0]
                coefs[h] = s.get(f, np.nan)

            coef_series = pd.Series(coefs)
            rows.append({
                "feature": f,
                "coef_5d": coef_series.get(5),
                "coef_10d": coef_series.get(10),
                "coef_20d": coef_series.get(20),
                "sign_consistent": len(set(np.sign(coef_series.dropna()))) == 1,
                "mean_abs_coef": coef_series.abs().mean(),
            })

        return pd.DataFrame(rows)



    def analyse_permutation_importance(self, best_per_family:dict, target_col:str):
        ''' Compare each best trend per family to the top performing metric to analyse relative worth #
            * Strong keep: ≥ 20-30% of top features importance
            * Conditional keep: 5-20%
            * Drop: ~0 or negative   
        '''
        target_perm_df = self.perm_df[target_col]
        max_perm = target_perm_df.abs().max()
        relative_importance_dict = {best_dict['feature'] :  abs(best_dict['importance']) / max_perm for best_dict in best_per_family.values()}
        # pretty_print_json(relative_importance_dict)
        return relative_importance_dict
        


    def prune_correlated_features(self, df: pd.DataFrame, features: list[str], perm_scores: pd.Series, corr_threshold: float = 0.8):
        ''' Remove highly correlated features, keeping the one with higher permutation importance.
        Args:
            df (DataFRame) : historical feature dataframe (time × features)
            features (List) : list of candidate feature names
            perm_scores (Dict) : permutation importance Series (indexed by feature)
            corr_threshold (Float) : absolute correlation threshold
        Returns:
            kept_features: list of retained feature names
            dropped_features: dict explaining which features were dropped and why
        '''
        # Ensure consistent ordering
        features = [f for f in features if f in df.columns and f in perm_scores.index]

        # Compute absolute correlation matrix
        corr = (df[features].dropna().corr().abs())
        dropped = set()
        reasons = {}

        # Upper triangle only (avoid duplicate checks)
        for i in range(len(features)):
            f1 = features[i]
            if f1 in dropped:
                continue
            for j in range(i + 1, len(features)):
                f2 = features[j]
                if f2 in dropped:
                    continue
                if corr.loc[f1, f2] >= corr_threshold:
                    # Compare permutation importance
                    if perm_scores[f1] >= perm_scores[f2]:
                        dropped.add(f2)
                        reasons[f2] = {"dropped_in_favor_of": f1, "abs_corr": corr.loc[f1, f2], "importance_ratio": perm_scores[f2] / perm_scores[f1]}
                    else:
                        dropped.add(f1)
                        reasons[f1] = {"dropped_in_favor_of": f2, "abs_corr": corr.loc[f1, f2], "importance_ratio": perm_scores[f1] / perm_scores[f2]}
                        break  # f1 is gone; move on

        kept = [f for f in features if f not in dropped]
        return kept, reasons




    def select_best_features_by_family(self, perm_scores, feature_families: dict, min_importance: float = 0.0):
        ''' Selects the strongest feature per family based on permutation importance
        Args:
            perm_scores (DataFrame) : DF with permuation scores for each feature
        '''
        selected = {}
        for family, features in feature_families.items():
            # Keep only features that exist in perm_scores
            valid = [f for f in features if f in list(perm_scores.index)]
            if not valid:
                continue

            # Pick feature with max importance
            best = perm_scores.loc[valid].idxmax()
            best_score = perm_scores.loc[best]
            if best_score > min_importance:
                selected[family] = {"feature": best, "importance": best_score}

        return selected
        
        
    
    def build_correlation_DF_and_sim_scores(self, core_features):
        ''' Aanayse correlation between candidate features & genereate similarity scores
        '''
        # Calculate correlation between core features
        corr_df = self.hist_df[core_features].dropna().corr()
        # self.corr_ff.to_csv('data/csv/test/corr_ff.csv')
        if self.show_results:
            print('Corr DF:')
            print(corr_df)
        
        abs_corr = corr_df.abs()
        upper = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))
        similar_pairs = (upper.stack().rename("abs_corr").reset_index().sort_values("abs_corr", ascending=False))
        return corr_df, abs_corr, similar_pairs



    def permutation_stability(self, model, X, y, n_splits=5, n_repeats=5, random_state=RANDOM_STATE):
        ''' Compute permutation importance stability across time splits.
            Returns mean and std importance per feature.
        '''
        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_imps = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            r = permutation_importance(model, X_val, y_val, n_repeats=n_repeats, random_state=random_state)
            all_imps.append(pd.Series(r.importances_mean, index=X.columns))

        imp_df = pd.concat(all_imps, axis=1)

        return pd.DataFrame({
            "mean_importance": imp_df.mean(axis=1),
            "std_importance": imp_df.std(axis=1),
            "stability_ratio": imp_df.mean(axis=1) / (imp_df.std(axis=1) + 1e-8)
        }).sort_values("mean_importance", ascending=False)




    def role_consistency_table(self, perm_df, stock_name, features, horizons=[5, 10, 20]):
        ''' 
        '''
        rows = []

        for f in features:
            row = {"feature": f}
            for h in horizons:
                col = f"signal_voladj_{h}d"
                row[f"imp_{h}d"] = perm_df[f'{stock_name}_{col}'].get(f, np.nan)
            rows.append(row)

        return pd.DataFrame(rows)



    # def top_k_features(self, perm_df: pd.DataFrame, k: int = 10):
    #     ''' Return top-k feature lists per run_id.
    #     Args:
    #         perm_df (DataFrame) : DF containing all permuatation info
    #         k (Int) : Top k features wwe want to extract
    #     Returns:
    #         out (Dict) : Dict of top k features
    #     '''
    #     out = {}
    #     for col in perm_df.columns:
    #         out[col] = perm_df[col].sort_values(ascending=False).head(k).index.tolist()
    #     return out


    # def stability_table(self, perm_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    #     ''' Counts how often each feature appears in the top-k across runs.
    #     Args:
    #         perm_df (DataFrame) : DF containing all permuatation info
    #         k (Int) : Top k features wwe want to extract
    #     Returns:
    #         counts (Series) : Series with counts
    #     '''
    #     tops = []
    #     for col in perm_df.columns:
    #         tops.extend(perm_df[col].sort_values(ascending=False).head(k).index.tolist())

    #     counts = pd.Series(tops).value_counts()
    #     return counts.rename("topk_count").to_frame()
        