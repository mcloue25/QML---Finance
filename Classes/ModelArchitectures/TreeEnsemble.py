import os 
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.base import clone
from collections import defaultdict
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, classification_report

from ..Plotting import Plotter



def create_folder(folder_name: str):
    os.makedirs(folder_name, exist_ok=True)
    

def save_JSON_object(json_object: object, json_path: str):
    ''' Saves a JSON Object to a path
    '''
    with open(json_path, 'w') as outfile:
        json.dump(json_object, outfile)


def load_JSON_object(json_path: str):
    ''' Loads JSON object from a file
    '''
    with open(json_path, "r") as f:
        json_data = json.load(f)
        return json_data



def load_LGBM_Classifier():
    ''' Function for loading an inst of the LGBM Classifier
    Returns:
        lgbm with specific params
    '''
    lgbm = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=5000, # high + early stopping finds the right point
        learning_rate=0.02,
        num_leaves=31, # complexity knob; see tuning docs
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    # Add early stopping via callbacks (modern LightGBM pattern). :contentReference[oaicite:1]{index=1}
    lgbm.set_params(callbacks=[lgb.early_stopping(stopping_rounds=100, first_metric_only=True)])
    return lgbm



def loadXGBoost_Classifier():
    '''  Function to load an instance of the XGBoost Classifier
    '''
    xgbc = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist", # usually fastest on CPU
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss"
    )
    return xgbc




def walkforward_cv_predict(base_model, X, y, n_splits=5, gap=0, labels=None, early_stopping_rounds=None, fit_kwargs=None):
    ''' Walk-forward CV predictions (OOF) with optional 'gap' (embargo) between train and val.
        gap: number of samples to drop from the end of training fold to avoid overlap leakage.
            For horizon h-day labels, a good default is gap=h.
    '''
    if fit_kwargs is None:
        fit_kwargs = {}

    if labels is None:
        labels = sorted(np.unique(y))

    tscv = TimeSeriesSplit(n_splits=n_splits)

    n_classes = len(labels)
    oof_proba = np.full((len(y), n_classes), np.nan)
    oof_pred = np.full(len(y), np.nan)
    history = defaultdict(list)

    model_name = base_model.__class__.__name__
    is_xgb = model_name.startswith("XGB")
    is_lgbm = model_name.startswith("LGBM")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        if gap > 0:
            train_idx = train_idx[:-gap] if len(train_idx) > gap else train_idx[:0]

        if len(train_idx) == 0:
            raise ValueError(f"Fold {fold}: train set empty after applying gap={gap}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = clone(base_model)

        if is_xgb and early_stopping_rounds is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                **fit_kwargs
            )

        elif is_lgbm:
            callbacks = []
            if early_stopping_rounds is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )

        else:
            # fallback for models without early stopping
            model.fit(X_train, y_train, **fit_kwargs)

        proba = model.predict_proba(X_val)
        pred = np.argmax(proba, axis=1)

        oof_proba[val_idx] = proba
        oof_pred[val_idx] = pred

        f1 = f1_score(y_val, pred, average="macro")
        loss = log_loss(y_val, proba, labels=labels)
        best_iter = getattr(model, "best_iteration", None)

        history["fold"].append(fold)
        history["macro_f1"].append(f1)
        history["log_loss"].append(loss)
        history["best_iteration"].append(best_iter)

        print(
            f"Fold {fold}: macroF1={f1:.4f}, "
            f"logloss={loss:.4f}, best_iter={best_iter}"
        )

    mask = ~np.isnan(oof_pred)
    print("\nOOF macroF1:",f1_score(y[mask], oof_pred[mask].astype(int), average="macro"))

    return oof_pred.astype(int), oof_proba, history




def ensemble_train_loop(base_model, X_dev, y_dev, X_test, y_test,gap=20, n_splits=5):
    ''' Main loop for training an ensemble model
    Args:
        base_model (Model) : Base ensemble mdoel for trianing
        X_dev (Tensor) : X training & val data 
        y_dev (List) : Y training & val data
        X_test (Tensor) : X test data
        y_test (List) : Y test data
    Returns:
        results (Dict) : Dict contianing training results info
    '''
    t0 = time.time()
    print(f"DEV samples: {len(y_dev):,} | TEST samples: {len(y_test):,}")
    print(f"CV splits: {n_splits} | gap: {gap}\n")

    print("==> Running walk-forward CV...")
    oof_pred, oof_proba, hist = walkforward_cv_predict(
        base_model,
        X_dev,
        y_dev,
        n_splits=n_splits,
        gap=gap,
        labels=[0,1,2],
        early_stopping_rounds=100
    )

    cv_time = time.time() - t0
    print(f"\nCV complete in {cv_time:.1f}s")

    # Summarize CV history
    mean_f1 = float(np.mean(hist["macro_f1"]))
    std_f1  = float(np.std(hist["macro_f1"]))
    mean_ll = float(np.mean(hist["logloss"]))
    std_ll  = float(np.std(hist["logloss"]))
    print(f"CV macroF1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"CV logloss: {mean_ll:.4f} ± {std_ll:.4f}\n")

    print("==> Fitting final model on all DEV and evaluating on TEST...")
    final_model = clone(base_model)
    final_model.fit(X_dev, y_dev)

    test_proba = final_model.predict_proba(X_test)
    test_pred = test_proba.argmax(axis=1)

    test_f1 = f1_score(y_test, test_pred, average="macro")
    test_loss = log_loss(y_test, test_proba, labels=[0,1,2])

    print(f"TEST macroF1: {test_f1:.4f}")
    print(f"TEST logloss: {test_loss:.4f}\n")

    print("TEST confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, test_pred, labels=[0,1,2]))
    print("\nTEST classification report:")
    print(classification_report(y_test, test_pred, labels=[0,1,2], digits=4))

    total_time = time.time() - t0
    print(f"\nTotal runtime: {total_time:.1f}s")

    results = {
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "hist": hist,
        "test_proba": test_proba,
        "test_pred": test_pred,
        "test_f1": float(test_f1),
        "test_loss": float(test_loss),
        "cv_macro_f1_mean": mean_f1,
        "cv_macro_f1_std": std_f1,
        "cv_logloss_mean": mean_ll,
        "cv_logloss_std": std_ll,
        "runtime_sec": float(total_time),
    }
    return results



def build_oof_df(dates, y_true, oof_pred, oof_proba, horizon, model_name):
    ''' 
    '''
    df = pd.DataFrame(
        index=pd.to_datetime(dates),
        data={
            "y_true": y_true,
            "y_pred": oof_pred
        }
    )

    # add probabilities
    for c in range(oof_proba.shape[1]):
        df[f"p_class_{c}"] = oof_proba[:, c]

    df["horizon"] = horizon
    df["model"] = model_name

    # keep only rows that were actually predicted
    df = df.loc[~df["y_pred"].isna()]
    return df



def analyse_ensemble_results(results_dict:dict, dates_dev, y_dev:list, arch_name:str, output_path:str):
    ''' Main fucntion for 
    '''
    # Create output folder
    create_folder(output_path)

    # Save oof
    oof_df = build_oof_df(
        dates_dev,
        y_dev,
        results_dict['oof_pred'],
        results_dict['oof_proba'],
        horizon=20,
        model_name=arch_name
    )
    oof_df.to_parquet(f"{output_path}{arch_name}.parquet")

    save_JSON_object(results_dict['hist'], 'data/json/lgbm.json')
    # Visualise Learning process
    # plotter = Plotter('')
    # plot