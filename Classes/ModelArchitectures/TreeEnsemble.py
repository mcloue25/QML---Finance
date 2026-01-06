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

from utilities import *
from ..Plotting import Plotter

CLASS_STRS = ['Sell', 'Hold', 'Buy']



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




def walkforward_cv_predict(
    base_model,
    X,
    y,
    splits,
    labels=(0, 1, 2),
    early_stopping_rounds=None,
    fit_kwargs=None
):
    """
    Walk-forward CV using PRECOMPUTED splits to guarantee identical folds across models.
    Returns:
        oof_pred: (T,)
        oof_proba: (T, n_classes)
        history: dict
    """
    if fit_kwargs is None:
        fit_kwargs = {}
    fit_kwargs = dict(fit_kwargs)
    fit_kwargs.pop("callbacks", None)

    labels = list(labels)
    n_classes = len(labels)

    oof_proba = np.full((len(y), n_classes), np.nan)
    oof_pred = np.full(len(y), np.nan)
    history = defaultdict(list)

    model_name = base_model.__class__.__name__
    is_xgb = model_name.startswith("XGB")
    is_lgbm = model_name.startswith("LGBM")

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = clone(base_model)

        # Early stopping handling
        if early_stopping_rounds is not None and (is_xgb or is_lgbm):
            if is_xgb:
                import xgboost as xgb
                early_stop = xgb.callback.EarlyStopping(
                    rounds=early_stopping_rounds,
                    save_best=True
                )
                model.set_params(callbacks=[early_stop])
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_kwargs)

            else:  # LGBM
                import lightgbm as lgb
                callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks, **fit_kwargs)
        else:
            # no early stopping (or non-supported model)
            if is_xgb or is_lgbm:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_kwargs)
            else:
                model.fit(X_train, y_train, **fit_kwargs)

        proba = model.predict_proba(X_val)
        pred = np.argmax(proba, axis=1)

        oof_proba[val_idx] = proba
        oof_pred[val_idx] = pred

        f1 = f1_score(y_val, pred, average="macro")
        loss = log_loss(y_val, proba, labels=labels)

        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = getattr(model, "best_iteration_", None)

        history["fold"].append(fold)
        history["macro_f1"].append(float(f1))
        history["log_loss"].append(float(loss))
        history["best_iteration"].append(best_iter)

        print(f"Fold {fold}: macroF1={f1:.4f} ::: log_loss={loss:.4f}, best_iter={best_iter}")

    mask = ~np.isnan(oof_pred)
    print("\nOOF macroF1:", f1_score(y[mask], oof_pred[mask].astype(int), average="macro"))

    return oof_pred.astype(int), oof_proba, history



def ensemble_train_loop(base_model, X_dev, y_dev, X_test, y_test, splits):
    t0 = time.time()
    print(f"DEV samples: {len(y_dev):,} | TEST samples: {len(y_test):,}")
    print(f"CV folds: {len(splits)} (precomputed)\n")

    print("Running walk-forward CV...")
    oof_pred, oof_proba, hist = walkforward_cv_predict(
        base_model=base_model,
        X=X_dev,
        y=y_dev,
        splits=splits,
        labels=(0, 1, 2),
        early_stopping_rounds=100
    )

    cv_time = time.time() - t0
    print(f"\nCV complete in {cv_time:.1f}s")

    mean_f1 = float(np.mean(hist["macro_f1"]))
    std_f1 = float(np.std(hist["macro_f1"]))
    mean_ll = float(np.mean(hist["log_loss"]))
    std_ll = float(np.std(hist["log_loss"]))

    print(f"CV macroF1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"CV logloss: {mean_ll:.4f} ± {std_ll:.4f}\n")

    print("Fitting on full dev & evaluating on test...")
    final_model = clone(base_model)
    final_model.fit(X_dev, y_dev)

    test_proba = final_model.predict_proba(X_test)
    test_pred = test_proba.argmax(axis=1)
    test_f1 = f1_score(y_test, test_pred, average="macro")
    test_loss = log_loss(y_test, test_proba, labels=[0, 1, 2])

    print(f"TEST macroF1: {test_f1:.4f}")
    print(f"TEST log loss: {test_loss:.4f}\n")

    total_time = time.time() - t0
    print(f"Total runtime: {total_time:.1f}s")

    return {
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



def analyse_ensemble_results(results_dict: dict, y_test: list, dates_dev, y_dev: list, arch_name: str, output_path: str, dates_test=None):
    ''' Main function for analysing a set of ensemble results 
    '''
    create_folder(output_path)
    create_folder(f'{output_path}graphs/')

    # CV diagnostics
    plotter = Plotter('')
    plotter.plot_hist(
        subset_dict(results_dict['hist'], ['log_loss', 'fold']),
        title='Loss Hist',
        xaxis='fold',
        yaxis='log_loss',
        save_path=f'{output_path}graphs/CV_fold_hist.png'
    )

    plotter.plot_confusion_matrices(y_test,
        results_dict['test_pred'],
        labels=[0, 1, 2],
        class_names=CLASS_STRS,
        title_prefix="Test",
        save_path=f'{output_path}graphs/confusion_matrix.png'
    )

    # NOTE - Save DEV / OOF predictions 
    oof_df = build_oof_df(
        dates_dev,
        y_dev,
        results_dict['oof_pred'],
        results_dict['oof_proba'],
        horizon=20,
        model_name=arch_name
    )

    oof_path = f"{output_path}{arch_name}_oof.parquet"
    oof_df.to_parquet(oof_path)

    # NOTE - Save test predictions
    if dates_test is not None:
        test_df = build_oof_df(
            dates_test,
            y_test,
            results_dict['test_pred'],
            results_dict['test_proba'],
            horizon=20,
            model_name=arch_name
        )

        test_path = f"{output_path}{arch_name}_test.parquet"
        test_df.to_parquet(test_path)

        print(f"Saved:\n  {oof_path}\n  {test_path}")
    else:
        print(f"Saved:\n  {oof_path} (test preds not saved: dates_test missing)")

