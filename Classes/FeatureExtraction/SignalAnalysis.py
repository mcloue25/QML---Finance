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




def pretty_print_json(dict_object: object):
    ''' Makes JSON object prettier for being printed 
    '''
    json_object = json.dumps(dict_object)
    parsed_json = json.loads(json_object) 
    print(json.dumps(parsed_json, indent=4))




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