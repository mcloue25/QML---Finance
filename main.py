import os 

import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Optional, Tuple, Any, Iterable

from utilities import *

# Classes for data fetching, feature generation & feature pruning / signal anlaysis
from Classes.FeatureExtraction.DataGenerator import DataDownloader
from Classes.FeatureExtraction.SignalAnalysis import SignalAnalyser
from Classes.FeatureExtraction.FeatureAnalysis import FeatureAnalyser
from Classes.FeatureExtraction.FeatureEngineering import FeatureBuilder

# Load model architectures for testing
from Classes.ModelArchitectures.TreeEnsemble import *
from Classes.ModelArchitectures.QuantumModels import *

# Classes for analysing & ranking a models performance
from Classes.ModelAnalysis.BackTesting import BacktestConfig, BackTest
from Classes.ModelAnalysis.ModelComparison import StressGrid, StockModelEvaluator

from Classes.Trading.PortfolioManagement import PortfolioManager



def download_data(ticker_list:list, period:str, output_path:str):
    ''' Function for downloading all of the ticker data I want to follow 
    Args:
        ticker_list (List) : List of stock symbols to download
        period (String) : Indicates the amount of time data we want to download
        output_path (String) : Path tot folder to save all CVS's to
    Returns:
        Downloads a CSV of historical data for each symbol in ticker_List and is saved in output_path
    '''
    # period == max by default gets all daya 
    # period = '1y' getrs the last year for testing
    # dl = DataDownloader(ticker_list, period, output_path)
    dl = DataDownloader(ticker_list, period, output_path)
    dl.create_folder(output_path)
    dl.save_data(dl.stock_dict, dl.output_path)



def feature_engineering(folder_path:str, output_path:str=None):
    '''  Function for generating featureset that we will then prune down to only the core features after feature analysis
    Args:
        folder_path (String) : Folder path to folder of historical CSV data
        output_path (String) : Output folder path for updated CSV with additional feature columns added
    '''
    for file in os.listdir(folder_path):
        print(f'Generating Feature Set for {file.split(".")[0]}')
        fb = FeatureBuilder(csv_path=f'{folder_path}{file}', output_path=output_path)
        fb.generate_features(build='core')
        fb.df.to_csv(f'{output_path}{file}')



def generate_importance_statistics(input_path:str, stock:str=None):
    ''' Main function for generating statistics for each features relative importance and reliability
    Args:
        input_path (String) : Folder path for updated CSV's with additional feature columns added
        stock (String) : Name of stock whos features are being analysed
    ''' 
    # Standardised testing features
    feature_cols = [
        # Short & medium trends
        "rolling_mean_return_5d",
        "rolling_mean_return_10d",
        "rolling_mean_return_20d",
        "price_ma_ratio_20d_centered",
        "price_ma_ratio_60d_centered",
        # Risk / Volatility
        "volatility_20d",
        "volatility_60d",
        "vol_of_vol_20d",
        "downside_vol_20d",
        # Path dependence / Regime state
        "drawdown",
        "max_drawdown_252d",
        "time_since_peak",
        # Tail / Distribution shape 
        "skew_60d",
        # Volume confirmation
        "log_volume_z_20d",
    ]
    create_folder('data/csv/feature_analysis/')
    create_folder(f'data/csv/feature_analysis/{stock}/')

    # Analyse feature importance for varying targte columns
    target_cols = ['signal_voladj_5d', 'signal_voladj_10d', 'signal_voladj_20d']
    # target_cols = ['signal_voladj_10d']
    for target in target_cols:
        create_folder(f'data/csv/feature_analysis/{stock}/{target}/')
        analyser = SignalAnalyser(csv_path=f'{input_path}{stock}.csv', feature_cols=feature_cols, target_col=target, date_col=None, output_dir=f'data/csv/feature_analysis/{stock}/{target}/')
        results = analyser.run_analysis()
        analyser.save_results(results, prefix=stock)



def feature_analysis(input_path:str, stock:str=None):
    ''' Function for analysing all generated featuires and assessing their worth for training models
    Args:
        input_path (String) : Path to 
        stock (String) : Name of stock whose features are being analysed
    Returns:
        list of key features for that stock
    '''
    # Takes a subset of features belonging to each trend family and generates importance statistics
    generate_importance_statistics(input_path, stock)

    # Compare feature results
    csv_folder = f'data/csv/feature_analysis/{stock}/'
    run_ids = os.listdir(csv_folder)
    fa = FeatureAnalyser(results_dir=csv_folder, hist_path=f'{input_path}{stock}.csv', stock=stock, run_ids=run_ids, top_k=5, show_results=False)
    key_features = fa.stability_df.loc[fa.stability_df["stability_class"].isin(["keep", "conditional"])]
    return list(key_features.index)



def split_train_val_test(file_path: str, feature_cols: list, target_col: str):
    ''' Function for taking a csv and converting it into an X & y dataset
    Args:
        file_path (String) : Path to CSV file
        feature_cols (List) : Subset of cols to keep from CSV
        target_col (String) : Target column we're trying to rpedict
    Returns:
        X (Tensor) : X data
        y (Tensor) : y data
        dates (Tensor) : date data
    '''
    df = pd.read_csv(file_path)

    # keep date index for ordering/alignment, but not as a feature
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # subset to needed columns and drop rows with missing values
    cols = feature_cols + [target_col]
    df = df[cols].dropna()

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy().astype(int)
    dates = df.index.to_numpy()

    return X, y, dates



def time_train_test_split(X, y, dates, test_size=0.2):
    n = len(y)
    cut = int(n * (1 - test_size))

    X_dev, X_test = X[:cut], X[cut:]
    y_dev, y_test = y[:cut], y[cut:]
    dates_dev, dates_test = dates[:cut], dates[cut:]

    return X_dev, y_dev, dates_dev, X_test, y_test, dates_test





def run_full_pipeline(ticker_object:list, download_path:str, architectrure_list:list):
    ''' Function to run the entire pipeline for all stocks from ends to end 
        Does:
            * Downloads all stock tickers
            * Generates Features
            * Assesses Relevance of generated features & keeps only core featureset
            * Trains Ensemble & Quantum models & saves results to parquet file
    Args:
        ticker_object (Dict) : Dict Where K == Economic sector && V == List of stock symbols to download data for within that sector
        download_path (String) : Folder path to download all stock historical data to
        architectrure_list (List) : List of different architectures to train models under ['ensembles' or 'quantum']
    Returns:
        Generates expanded feature set CSV for all stock symbols
        Trains model architecture variant for each model in architecture list
        Saves portfolio_key_features to JSON for analysis of what features tend to matter more in a given sector & for risk analysis
    '''
    # Download all ticker data
    # for sector, ticker_list in tqdm(ticker_object.items(), desc='Downloading Sector Data...'):
    #     download_data(ticker_list, period="max", output_path=download_path)

    # Geneerate features for all downloaded ticker data
    # feature_engineering(download_path, output_path='data/csv/historical/training/cleaned/')

    # Iterate through each sector/tickeer_list combo & train models + store key features to JSON obeject
    portfolio_key_features = {}
    for sector, ticker_list in ticker_object.items():
        portfolio_key_features[sector] = {}
        for stock_name in ticker_list:
            # Analyse Feature importance for modelling
            key_features = feature_analysis('data/csv/historical/training/cleaned/', stock=stock_name)
            # key_features = ['feat_1', 'feat_2', 'feat_3']
            portfolio_key_features[sector][stock_name] = key_features
            
            # NOTE - Main function for training classical & quantum models
            # NOTE - Having some issue here with either USDC-USD or USDT-USD
            # try:
            #     model_training(csv_path=f'data/csv/historical/training/cleaned/{stock_name}.csv', stock_name=stock_name, feat_list=key_features, architectures=architectrure_list)
            # except:
            #     continue
    save_JSON_object(portfolio_key_features, 'data/json/portfolio_key_features.json')
    




def model_training(csv_path:str, stock_name:str, feat_list=None, horizon:int=10, architectures=['ensemble', 'quantum']):
    ''' Main function for training different models on the same data
    Args:
        csv_path (String) : PAth to historical data CSV 
        stock_name (String) : Name of stock being modelled
        feat_list (List) : List of key features after feature improtance pipeline has been applied
        horizon (Int) : Preferred horizon date
        architectures (List) : List of potential architectures to train from
    '''
    # Load dataset
    X, y, dates = split_train_val_test(
        file_path=csv_path,
        feature_cols=feat_list,
        target_col="signal_voladj_10d"
    )

    # Dev/Test split keeping in chronological
    X_dev, y_dev, dates_dev, X_test, y_test, dates_test = time_train_test_split(X, y, dates, test_size=0.2)

    # Create splits to be used by classicial DL and QML modelling
    splits = make_walkforward_splits(X_dev, n_splits=5, gap=horizon)
    
    # NOTE - Train ensembles
    if 'ensemble' in architectures:
        ensemble_model_training(
            stock_name=stock_name, 
            X_dev=X_dev, y_dev=y_dev, dates_dev=dates_dev, 
            X_test=X_test, y_test=y_test, dates_test=dates_test, 
            splits=splits
        )

    # NOTE - Train quantum model using same splits
    if 'quantum' in architectures: 
        quantum_model_training(
            stock_name=stock_name,
            X_dev=X_dev, y_dev=y_dev, dates_dev=dates_dev,
            X_test=X_test, y_test=y_test, dates_test=dates_test,
            splits=splits
        )




def ensemble_model_training(stock_name, X_dev, y_dev, dates_dev, X_test, y_test, dates_test, splits):
    '''  Mian fucntion for training ensemble models to compare against QML counterparts
    Args:
        stock_name (String) : String stock name
        X_dev (np.ndarray) : X training features
        y_dev (List) : List of ground truthy train & val results
        dates_dev (Series) : Dates column for training features
        X_test (np.ndarray) : X test features
        y_test (List) : List of ground truth test results
        dates_test (Series) : Dates column for test features
        splits () : Folds of time-series correct data
    Returns:
        Trains anmd
    '''
    create_folder(f"data/results/trained_models/ensembles/{stock_name}/")

    # LGBM
    lgbm = load_LGBM_Classifier()
    lgbm_results = ensemble_train_loop(lgbm, X_dev, y_dev, X_test, y_test, splits)
    # Analyse LGBM results
    analyse_ensemble_results(
        lgbm_results, 
        y_test, 
        dates_dev, 
        y_dev,
        "LGBM_V1",
        output_path=f"data/results/trained_models/ensembles/{stock_name}/LGBM_V1/",
        dates_test=dates_test,
    )

    # XGBoost
    xgb = loadXGBoost_Classifier()
    xgb_results = ensemble_train_loop(xgb, X_dev, y_dev, X_test, y_test, splits)
    # Analyse XGBoost results
    analyse_ensemble_results(
        xgb_results, 
        y_test, 
        dates_dev, 
        y_dev,
        "XGBoost_V1",
        output_path=f"data/results/trained_models/ensembles/{stock_name}/XGBoost_V1/",
        dates_test=dates_test,
    )



def quantum_model_training(stock_name, X_dev, y_dev, dates_dev, X_test, y_test, dates_test, splits, depth=2, Cs=(0.1, 1.0, 10.0), model_tag="QKernelSVC_V1"):
    ''' Function for training different kinds of quantum models
    Args:
        stock_name (String) : String stock name
        X_dev (np.ndarray) :
        y_dev (np.ndarray) :
        dates_dev (np.ndarray) :
        X_test (np.ndarray) :
        y_test (np.ndarray) :
        splits () : 
        depth () : Controls how expressive the quantum feature map is 
        Cs (Tuple) : Controls how aggressivly the SVM uses the kernel (bias / variance tradeoff)
        model_tag (String) : Name of run ID
    '''
    create_folder(f"data/results/trained_models/qml/{stock_name}/{model_tag}/")

    cache = KernelCache()
    all_results = []

    for C in Cs:
        results = quantum_kernel_train_loop(
            X_dev, y_dev, X_test, y_test, splits,
            depth=depth,
            C=C,
            model_tag=f"QKernelSVC_depth{depth}_C{C}",
            cache=cache,
            stock_name=stock_name,  # used for safe cache keys
        )
        all_results.append(results)

        # Option A: analyze each run immediately
        analyse_ensemble_results(
            results, y_test, dates_dev, y_dev, results["model_tag"],
            output_path=f"data/results/trained_models/qml/{stock_name}/{model_tag}/{results['model_tag']}/"
        )
    cache.clear() 

    # Option B: also pick and report best (example: lowest CV logloss)
    best = min(all_results, key=lambda r: r["cv_logloss_mean"])
    print(f"Best QML run: {best['model_tag']} | CV logloss mean={best['cv_logloss_mean']:.4f}")

    return all_results, best






def get_sector_by_symbol(key_features_object):
    ''' Inverts the architecture of a dict of depth 2
    Args:
        key_features_object (Dict) : Dict where K == Sector && V == Dict where K == Ticker symbol && V == List of core features
    Returns:
        inverted (Dict) : 1D dict where K == ticker symbols && V == Sector they belong to
    '''
    inverted = {}
    for sector, symbol_feats_dict in key_features_object.items():
        inverted = {**inverted, **{symbol : sector for symbol, _ in symbol_feats_dict.items()}}
    return inverted



def run_backtests_with_comparison(ticker_object:list, feat_dict_path:str, performance_dict_path:str):
    ''' Function to replace current run_bakctests() where it will run backtests for all different model architectures & then compare their results and assess their validity
    Args:
        ticker_list (List) : List of ticker symbols
        feat_dict_path (String) : String to JSON obejct where K==ticker symbol && V== list of core features generated for that symbol
        performance_dict_path (String) : File path to location where we want to save model performance ranking data to
    Returns:
        Runs backtests for all available trained models for a given stock
        Then ranks each models performance in relation to all other models trained for that stock
        Saves all top performing models data to stock_specific_model_rankings.json
    '''
    # Need to invert key features dict so we know what sector each symbol belongs to
    key_features_object = load_JSON_object(feat_dict_path)
    inverted_dict = get_sector_by_symbol(key_features_object)
    
    performance_json = {}
    for sector, ticker_list in ticker_object.items():
        for stock_name in ticker_list:

            # NOTE - Run ensemble backtests
            run_stock_backtests(stock_name)

            # Load historical 
            stock_hist_path = f"data/csv/historical/training/cleaned/{stock_name}.csv"
            prices = pd.read_csv(stock_hist_path, parse_dates=["date"]).set_index("date")["close"]

            # Load y_dev/y_test & dates_dev/dates_test from dataset split function
            X, y, dates = split_train_val_test(
                file_path=stock_hist_path,
                feature_cols=key_features_object[inverted_dict[stock_name]][stock_name],
                target_col="signal_voladj_10d"
            )
            # Split data time serries correct way
            X_dev, y_dev, dates_dev, X_test, y_test, dates_test = time_train_test_split(X, y, dates, test_size=0.2)

            model_pred_paths = {
                "XGBoost_V1": {
                    "dev": f"data/results/trained_models/ensembles/{stock_name}/XGBoost_V1/XGBoost_V1.parquet",
                    "test": f"data/results/trained_models/ensembles/{stock_name}/XGBoost_V1/XGBoost_V1_test.parquet",
                },
                "LGBM_V1": {
                    "dev": f"data/results/trained_models/ensembles/{stock_name}/LGBM_V1/LGBM_V1.parquet",
                    "test": f"data/results/trained_models/ensembles/{stock_name}/LGBM_V1/LGBM_V1_test.parquet",
                },
                # "QKernelSVC_depth2": {
                #     "dev": f"data/results/trained_models/qml/{stock_name}/QKernelSVC_V1/QKernelSVC_depth2_C1.0.parquet",
                #     "test": f"data/results/trained_models/qml/{stock_name}/QKernelSVC_V1/QKernelSVC_depth2_C1.0_test.parquet",
                # }
            }

            # NOTE - Evaluate different model results 
            out_dir = f"data/results/model_comparisons/{stock_name}/"
            model_ranking_dict = evaluate_models_for_stock(
                stock_name=stock_name,
                prices=prices,
                dev_dates=dates_dev,
                test_dates=dates_test,
                y_dev=y_dev,
                y_test=y_test,
                model_pred_paths=model_pred_paths,
                out_dir=out_dir,
                horizon_days=10,
            )
            # Store best performing model results for each stock so portfolioManager can use ranking later
            performance_json[stock_name] = model_ranking_dict
    save_JSON_object(performance_json, performance_dict_path)



def run_stock_backtests(stock_name:list):
    ''' Function to run backtests for a given stock and then analyse model performance against counterparts
    Args:
        stock_name (String) : String stock name
    Returns:
        RUns multiple backtests using different model architectures
    '''
    for ensemble_type in ['XGBoost_V1', 'LGBM_V1']:
        backtest_model_performance(parquet_path=f'data/results/trained_models/ensembles/{stock_name}/{ensemble_type}/{ensemble_type}.parquet', 
                                    model_type=ensemble_type,
                                    stock_hist_path=f'data/csv/historical/training/cleaned/{stock_name}.csv',
                                    stock_name=stock_name,
                                    backtest_output_path='data/results/backtests/')
            


def backtest_model_performance(parquet_path, model_type:str, stock_hist_path:str, stock_name:str, backtest_output_path:str, horizon_days:int=10):
    ''' Run OOF backtest on trained model to evaluate performance 
    Args:
        parquet_path (String) : Path to saved model path
        stock_hist_path (String) : Path to ticker historical data
    '''
    oof_df = pd.read_parquet(parquet_path)
    prices = pd.read_csv(stock_hist_path, parse_dates=["date"]).set_index("date")["close"]
    cfg = BacktestConfig(
        horizon_days=10,
        transaction_cost=0.0005,
        mode="long_only",
        hold_rule="signal_until_change"   # "fixed_horizon"
    )

    # Initialise Backtest inst
    create_folder(f'{backtest_output_path}{stock_name}/{model_type}/')
    bt = BackTest(path=f'{backtest_output_path}{stock_name}/{model_type}/', config=cfg)

    # NOTE - Run backtest & generate metrics and trades list
    bt_df = bt.run(oof_df, prices)
    metrics = bt.performance_metrics(bt_df)
    trade_stats  = bt.trade_stats(bt_df)

    meta = generate_run_ID(stock_name, model_type, horizon_days)
    run_id = str(uuid.uuid4())
    run_row = {**meta, **metrics, **trade_stats}

    # NOTE - NEED TO ADD WARM UP ROWS FOR AUDITING & TRACKING DEFENSIVENESS
    # run_row.update({
    #     "oof_warmup_rows": oof_meta["n_dropped_nan"],
    #     "oof_warmup_end_date": oof_meta["warmup_end_date"],
    # })
    

    trades_made_df = bt.trade_list(bt_df, pos_col="position_lag", price_col="price")
    trades_made_df = trades_made_df.assign(run_id=run_id, symbol=stock_name)
    bt_df = bt_df.assign(run_id=run_id, symbol=stock_name)
    
    # Save backtest results
    bt.save_backtest_artifacts(
        base_dir=f'{backtest_output_path}{stock_name}/{model_type}/',
        run_row=run_row,
        bt_df=bt_df,
        trades_df=trades_made_df
    )

    # NOTE - Results
    pretty_print_json(metrics)
    pretty_print_json(trade_stats)
    create_folder(f'{backtest_output_path}{stock_name}/{model_type}/graphs/')
    bt.plot_equity_curve(bt_df, title="OOF Strategy Equity Curve (10d model)", save_path=f'{backtest_output_path}{stock_name}/{model_type}/graphs/equity_curve.png')
    bt.plot_positions(bt_df, title="OOF Position (10d model)", save_path=f'{backtest_output_path}{stock_name}/{model_type}/graphs/positions.png')




def evaluate_models_for_stock(stock_name: str, prices: pd.Series, *, dev_dates: np.ndarray, test_dates: np.ndarray, y_dev: np.ndarray, y_test: np.ndarray, model_pred_paths: Dict[str, Dict[str, str]], out_dir: str, horizon_days: int = 10):
    ''' model_pred_paths:
        {
            "XGBoost_V1": {"dev": "...parquet", "test": "...parquet"},
            "LGBM_V1":    {"dev": "...parquet", "test": "...parquet"},
            "QKernel...": {"dev": "...parquet", "test": "...parquet"},
        }
        Each parquet should contain columns: p_class_0,p_class_1,p_class_2 (and optionally y_pred) and be indexed by date or have a date column.
    Args:
        stock_name (String) :
        prices
    '''
    # Initialise evaluator
    evaluator = StockModelEvaluator(stock_name=stock_name, prices=prices, BackTestClass=BackTest, BacktestConfigClass=BacktestConfig, horizon_days=horizon_days)

    # Register each model (dev + test)
    for model_tag, paths in model_pred_paths.items():
        preds_dev = pd.read_parquet(paths["dev"])
        preds_test = pd.read_parquet(paths["test"])

        # NOTE - Need to add "date" col back inot data
        # Dealing with dat col
        # if "date" in preds_dev.columns:
        #     preds_dev["date"] = pd.to_datetime(preds_dev["date"])
        #     preds_dev = preds_dev.set_index("date").sort_index()
        # if "date" in preds_test.columns:
        #     preds_test["date"] = pd.to_datetime(preds_test["date"])
        #     preds_test = preds_test.set_index("date").sort_index()

        # Register each model before I comapre them
        evaluator.register_model(model_tag, preds_dev=preds_dev, preds_test=preds_test, y_dev=y_dev, y_test=y_test, dates_dev=dev_dates, dates_test=test_dates)

    # Run evaluation + robustness grid
    summary_df, details = evaluator.evaluate_all_models(stress=StressGrid(), base_transaction_cost=0.0005, base_execution_lag=1, base_entry_threshold=0.20)

    # Save outputs
    create_folder(out_dir)
    summary_path = os.path.join(out_dir, f"{stock_name}_model_comparison.parquet")
    summary_df.to_parquet(summary_path, index=False)

    # Save stress tests (one parquet per model)
    for model_tag, res in details.items():
        if "stress_test" in res and res["stress_test"] is not None:
            stress_df = res["stress_test"]
            stress_df.to_parquet(os.path.join(out_dir, f"{stock_name}_{model_tag}_stress.parquet"), index=False)

    # Rank model performance & return dict with best ranked model
    model_ranking_dict = evaluator.compare_model_results(summary_df)
    return model_ranking_dict



def generate_diversified_portfolio(performance_dict_path:str, portfolio_sectors_path:str, philosophy_path:str, philosophy:str, initial_investment:int =100_000):
    ''' Load the PortfolioManager Class and distribute initial capital across investments
    Args:
        performance_dict_path (String) :
        initial_investment (Int) : 
    '''
    manager = PortfolioManager()
    # Load all sector and symbol data 
    sector_data = manager.load_sector_data(portfolio_sectors_path)
    symbol_sector_dict = manager.get_sector_by_symbol(sector_data)

    # Taking the best performing model for each stock now rank their performances and choose the top N to diversify capital between
    manager.analyse_best_performing_models(performance_dict_path, philosophy_path, philosophy)

    # Disperse capital amongst ranked stocks based on distribution mindset
    position_df = manager.genereate_portfolio(initial_investment, tactic='SOFTMAX', show=False)

    # Show distribution of holdings between sectors
    manager.assess_portforlio_diversity(symbol_sector_dict, position_df, show=True)
    # subset_df = position_df[['stock', 'model_tag', 'weight', 'allocation_score', 'euro']]
    # subset_df.to_csv('data/csv/test/position_df.csv')



def main():
    '''  Main function for building models:
        Plan:
            - Download ticker data
            - Generate list of features 
            - Analyse each features importance / noise contribution
            - Take subset of key features and train model 
            - MODEL TRAINING:
                * Ensembles
                * Time based models
                * QML models
            - Perform backtest
    '''
    # NOTE - Load portfolio data 
    portfolio_symbols = load_JSON_object('data/json/tracked_stocks.json')
    architecture_list = ['ensemble']  # quantum

    # NOTE - For running the entire pipelien in plan above 
    # run_full_pipeline(
    #     portfolio_symbols, 
    #     download_path='data/csv/historical/training/raw/',
    #     architectrure_list=architecture_list
    # )

    # NOTE - CURRENTLY BEING WORKED ON TO REPLACE run_backtests() - will run all backtests & compare model results locally
    # run_backtests_with_comparison(portfolio_symbols, feat_dict_path='data/json/portfolio_key_features.json', performance_dict_path='data/json/stock_specific_model_rankings.json')


    # NOTE - Using best performing model data use PortfolioManager to distribute initial capital into investments
    generate_diversified_portfolio(
        performance_dict_path='data/json/stock_specific_model_rankings.json',
        portfolio_sectors_path = 'data/json/tracked_stocks.json',
        philosophy_path = 'data/json/portfolio_philosophy.json',
        philosophy = 'BALANCED_V1'
    )


if __name__ == "__main__":
    main()
