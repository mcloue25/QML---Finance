import os
import glob

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




class SignalAnalyser:
    '''
    Analyse which engineered features matter most.

    Outputs:
      - coefficient importance (L1 logistic)
      - permutation importance
      - correlation matrix
      - VIF scores
    '''

    def __init__(self, csv_path:str, feature_cols:list[str], target_col:str, date_col:str | None = None, output_dir:str =None):
        self.csv_path = csv_path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = pd.read_csv(self.csv_path)

        # If date column exists, parse and sort; if not, just sort index after setting it elsewhere
        if date_col and date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors="coerce")
            self.df = self.df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        self.df = self.preprocessing(self.df)



    def preprocessing(self, df: pd.DataFrame):
        ''' Minimal, safe preprocessing for signal analysis:
            - keep only rows where target exists
            - handle infinities
            - DO NOT drop rows just because some engineered feature is NaN
        '''
        df = df.copy()
        # Require target
        df = df.dropna(subset=[self.target_col])
        # Replace inf
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # For sparse-by-design features, you should not drop rows.
        # Instead: fill NaNs in features with 0 (or later with an indicator).
        df[self.feature_cols] = df[self.feature_cols].fillna(0.0)
        return df


    def time_split(self, train_pc:float =0.7, val_pc:float =0.15):
        ''' Chronological train / val / test
        '''
        n = len(self.df)
        train_end = int(n * train_pc)
        val_end = int(n * (train_pc + val_pc))
        train_df = self.df.iloc[:train_end]
        val_df = self.df.iloc[train_end:val_end]
        test_df = self.df.iloc[val_end:]
        # Make Train / Val / Test
        X_train = train_df[self.feature_cols]
        y_train = train_df[self.target_col]
        X_val = val_df[self.feature_cols]
        y_val = val_df[self.target_col]
        X_test = test_df[self.feature_cols]
        y_test = test_df[self.target_col]
        return X_train, y_train, X_val, y_val, X_test, y_test


    def run_analysis(self, train_pc: float = 0.7, val_pc: float = 0.15):
        ''' Run all analyses and return a dict of results.
        Args:
            train_pc
        '''
        X_train, y_train, X_val, y_val, X_test, y_test = self.time_split(train_pc, val_pc)
        # Fit model
        pipe = self.make_l1_logistic_pipeline(max_iter=500)
        pipe = self.fit_model(pipe, X_train, y_train)
        # Importance
        coef_imp = self.coefficient_importance(pipe, list(X_train.columns))
        perm_imp = self.permutation_feature_importance(pipe, X_val, y_val, n_repeats=10)
        # Collinearity diagnostics
        corr = self.correlation_matrix(X_train)
        vif = self.compute_vif(X_train)

        results = {
            "coef_importance": coef_imp,
            "perm_importance": perm_imp,
            "corr": corr,
            "vif": vif,
            "model": pipe
        }
        return results



    def save_results(self, results: dict, prefix: str = "signals"):
        ''' Save tables to CSV for reporting.
        '''
        # How strongly each feature effects the prediction
        results["coef_importance"].to_csv(os.path.join(self.output_dir, f"{prefix}_coef_importance.csv"))
        # How much val performance drops when each feature is randomly shuffled 
        results["perm_importance"].to_csv(os.path.join(self.output_dir, f"{prefix}_perm_importance.csv"))
        # Feature to fetature correlation matrix 
        results["corr"].to_csv(os.path.join(self.output_dir, f"{prefix}_corr.csv"))
        # Measures multicolinearity
        results["vif"].to_csv(os.path.join(self.output_dir, f"{prefix}_vif.csv"))




    def make_l1_logistic_pipeline(self, max_iter: int = 500):
        ''' Create a StandardScaler + L1 LogisticRegression pipeline.
        '''
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=max_iter,
                random_state=RANDOM_STATE
            ))
        ])


    def fit_model(self, pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        ''' Fit pipeline on training data.
        '''
        pipe.fit(X_train, y_train)
        return pipe


    def coefficient_importance(self, pipe: Pipeline, feature_names: list[str]) -> pd.Series:
        ''' Return coefficient-based importance for logistic regression.
            For multi-class, returns mean absolute coefficient across classes.
        '''
        clf = pipe.named_steps["clf"]
        coef = clf.coef_
        # Binary: shape (1, n_features); Multi-class: (n_classes, n_features)
        if coef.ndim == 2 and coef.shape[0] > 1:
            imp = np.mean(np.abs(coef), axis=0)
            signed = np.mean(coef, axis=0)  # optional signed summary
            s = pd.Series(signed, index=feature_names)
            # sort by absolute importance
            return s.reindex(pd.Series(imp, index=feature_names).sort_values(ascending=False).index)
        else:
            s = pd.Series(coef.ravel(), index=feature_names)
            return s.sort_values(key=np.abs, ascending=False)


    def permutation_feature_importance(self, pipe: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, n_repeats: int = 10, scoring:str | None = None):
        ''' Permutation importance (model-agnostic). Returns mean importance sorted descending.#
        '''
        r = permutation_importance(pipe, X_val, y_val, n_repeats=n_repeats, random_state=RANDOM_STATE, scoring=scoring)
        return pd.Series(r.importances_mean, index=X_val.columns).sort_values(ascending=False)


    def correlation_matrix(self, X: pd.DataFrame):
        '''Compute feature correlation matrix.
        '''
        return X.corr()


    def compute_vif(self, X: pd.DataFrame):
        ''' Compute Variance Inflation Factor (VIF) for each feature.
            Note: VIF assumes a roughly linear relationship; standardize first.
        '''
        X_scaled = StandardScaler().fit_transform(X.values)
        vifs = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
        return pd.Series(vifs, index=X.columns).sort_values(ascending=False)






















class ResultsAggregation:
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
        # print(perm_scores)
        print(list(perm_scores.index))
        best_per_family = self.select_best_features_by_family(perm_scores, FEATURE_FAMILIES)
        print(best_per_family)



    def select_best_features_by_family(perm_scores: pd.Series, feature_families: dict, min_importance: float = 0.0):
        ''' Selects the strongest feature per family based on permutation importance
        Args:
            perm_scores (DataFrame) : DF with permuation scores for each feature
        '''
        selected = {}
        for family, features in feature_families.items():
            # print(features)
            # print
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
        