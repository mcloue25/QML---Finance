import os 

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from utilities import *
from Classes.BackTesting import BacktestConfig, BackTest
from Classes.FeatureExtraction.DataGenerator import DataDownloader
from Classes.FeatureExtraction.SignalAnalysis import SignalAnalyser
from Classes.FeatureExtraction.FeatureAnalysis import FeatureAnalyser
from Classes.FeatureExtraction.FeatureEngineering import FeatureBuilder

# Load model architectures for testing
from Classes.ModelArchitectures.TreeEnsemble import *
from Classes.ModelArchitectures.QuantumModels import *



def download_data(ticker_list:list, period:str, output_path:str):
    ''' Function for downloading all of the ticker data I want to follow 
    Args:
        output_path (String) : Path tot folder to save all CVS's to
    '''
    # period == max by default gets all daya 
    # period = '1y' getrs the last year for testing
    # dl = DataDownloader(ticker_list, period, output_path)
    dl = DataDownloader(ticker_list, period, output_path)
    dl.create_folder(output_path)
    dl.save_data(dl.stock_dict, dl.output_path)



def feature_engineering(folder_path:str, output_path:str=None):
    '''  
    '''
    for file in os.listdir(folder_path):
        print(f'Generating Feature Set for {file.split(".")[0]}')
        fb = FeatureBuilder(csv_path=f'{folder_path}{file}', output_path=output_path)
        fb.generate_features(build='core')
        fb.df.to_csv(f'{output_path}{file}')



def generate_importance_statistics(input_path:str, stock:str=None):
    ''' Main function for generating statistics for each features relative importance and reliability
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
    fa = FeatureAnalyser(results_dir=csv_folder, hist_path=f'{input_path}{stock}.csv', stock=stock, run_ids=run_ids, top_k=5, show_results=True)
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





def run_full_pipeline():
    ''' Function to run the entire pipeline for all stocks from ends to end 
        Does:
            * Downloads all stock tickers
            * Generates Features
            * Assesses Relevance of generated features & keeps only core featureset
            * Trains Ensemble & Quantum models & saves results to parquet file
            * Backtests model pereofrmance against time-series-correct data

    '''
    ticker_list = [
        "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD",
        "XRP-USD", "USDC-USD", "ADA-USD", "DOGE-USD", "TRX-USD"
    ]
    # ticker_list = ["BTC-USD"]

    download_path = 'data/csv/historical/training/raw/'

    # Download all ticker data
    # download_data(ticker_list, period="max", output_path=download_path)

    # Geneerate features for all downloaded ticker data
    # feature_engineering(download_path, output_path='data/csv/historical/training/cleaned/')
    
    features_dict = {}
    for stock_name in ticker_list:
        # Analyse Feature importance for modelling
        key_features = feature_analysis('data/csv/historical/training/cleaned/', stock=stock_name)
        features_dict[stock_name] = key_features
        
        # NOTE - Main functionf or training classical & quantum models
        model_training(csv_path=f'data/csv/historical/training/cleaned/{stock_name}.csv', stock_name=stock_name, feat_list=key_features)


    ''' Get model before archtype
        Might have to move this around so its:
            * stock_name/arch_type (ensemble vs qml) / arch_type / .parquet file
        
    '''
    for arch_type in os.listdir('data/results/trained_models/'):
        for model_type in os.listdir(f'data/results/trained_models/{arch_type}/'):
            

            Perform backtest              f'data/results/trained_models/{arch_type}/{model_type}/'
            backtest_model_performance(parquet_path=f'data/results/trained_models/ensembles/{stock_name}/XGBoost_V1/XGBoost_V1.parquet', 
                                    # This is causing the ISSUE Need to iterate over possible model Architectures
                                    model_type='xgb',
                                    stock_hist_path='data/csv/historical/training/cleaned/BTC-USD.csv',
                                    stock_name=stock_name,
                                    backtest_output_path="data/results/backtests/"


            # # Perform backtest
            # backtest_model_performance(parquet_path=f'data/results/trained_models/ensembles/{stock_name}/XGBoost_V1/XGBoost_V1.parquet', 
            #                         # This is causing the ISSUE Need to iterate over possible model Architectures
            #                         model_type='xgb',
            #                         stock_hist_path='data/csv/historical/training/cleaned/BTC-USD.csv',
            #                         stock_name=stock_name,
            #                         backtest_output_path="data/results/backtests/"


        # backtest_model_performance(parquet_path=f'data/results/trained_models/qml/{stock_name}/QKernelSVC_V1/QKernelSVC_depth2_C10.0/QKernelSVC_depth2_C10.0.parquet', 
        #                         # This is causing the ISSUE Need to iterate over possible model Architectures
        #                            model_type='QKernelSVC_depth2_C10.0',
        #                            stock_hist_path='data/csv/historical/training/cleaned/BTC-USD.csv',
        #                            stock_name=stock_name,
        #                            backtest_output_path="data/results/backtests/")
    
    # save_JSON_object(features_dict, 'data/json/stock_key_features.json')
    




def model_training(csv_path:str, stock_name:str, feat_list=None, horizon:int=10):
    ''' Main function for training different models on the same data
    Args:
        csv_path (String) : PAth to historical data CSV 
        stock_name (String) : Name of stock being modelled
        feat_list (List) : List of key features after feature improtance pipeline has been applied
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
    ensemble_model_training(
        stock_name=stock_name, 
        X_dev=X_dev, y_dev=y_dev, dates_dev=dates_dev, 
        X_test=X_test, y_test=y_test, dates_test=dates_test, 
        splits=splits
    )

    # NOTE - Train quantum model using same splits
    # quantum_model_training(
    #     stock_name=stock_name,
    #     X_dev=X_dev, y_dev=y_dev, dates_dev=dates_dev,
    #     X_test=X_test, y_test=y_test, dates_test=dates_test,
    #     splits=splits
    # )




def ensemble_model_training(stock_name, X_dev, y_dev, dates_dev, X_test, y_test, dates_test, splits):
    '''  Mian fucntion for training ensemble models to compare against QML counterparts
    '''
    create_folder(f"data/results/trained_models/ensembles/{stock_name}/")

    # LGBM
    lgbm = load_LGBM_Classifier()
    lgbm_results = ensemble_train_loop(lgbm, X_dev, y_dev, X_test, y_test, splits)
    analyse_ensemble_results(
        lgbm_results, y_test, dates_dev, y_dev,
        "LGBM_V1",
        output_path=f"data/results/trained_models/ensembles/{stock_name}/LGBM_V1/"
    )

    # XGBoost
    xgb = loadXGBoost_Classifier()
    xgb_results = ensemble_train_loop(xgb, X_dev, y_dev, X_test, y_test, splits)
    analyse_ensemble_results(
        xgb_results, y_test, dates_dev, y_dev,
        "XGBoost_V1",
        output_path=f"data/results/trained_models/ensembles/{stock_name}/XGBoost_V1/"
    )



def quantum_model_training(stock_name, X_dev, y_dev, dates_dev, X_test, y_test, dates_test, splits, depth=2, Cs=(0.1, 1.0, 10.0), model_tag="QKernelSVC_V1"):
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









def backtest_model_performance(parquet_path, model_type:str, stock_hist_path:str, stock_name:str, backtest_output_path:str, horizon_days:int=10):
    ''' Run OOF backtest on trained model to evaluate performance 
    Args:
        parquet_path (String) : Path to saved model path
        stock_hist_path (String) : Path to ticker historical data
    '''
    oof_df = pd.read_parquet(parquet_path)
    prices = pd.read_csv(stock_hist_path, parse_dates=["date"]).set_index("date")["close"]
    print(oof_df)
    cfg = BacktestConfig(
        horizon_days=10,
        transaction_cost=0.0005,
        mode="long_only",
        hold_rule="signal_until_change"   # "fixed_horizon"
    )

    # Initialise Backtest inst
    create_folder(f'{backtest_output_path}{stock_name}/{model_type}/')
    bt = BackTest(path=f'{backtest_output_path}{stock_name}/{model_type}/', config=cfg)

    # Run backtest & generate metrics and trades
    bt_df = bt.run(oof_df, prices)
    metrics = bt.performance_metrics(bt_df)
    trade_stats  = bt.trade_stats(bt_df)

    meta = generate_run_ID(stock_name, model_type, horizon_days)
    run_id = str(uuid.uuid4())
    run_row = {**meta, **metrics, **trade_stats}
    trades_made_df = bt.trade_list(bt_df, pos_col="position_lag", price_col="price")
    trades_made_df = trades_made_df.assign(run_id=run_id, symbol=stock_name)
    bt_df = bt_df.assign(run_id=run_id, symbol=stock_name)
    
    bt.save_backtest_artifacts(
        base_dir=f'{backtest_output_path}{stock_name}/{model_type}/',
        run_row=run_row,
        bt_df=bt_df,
        trades_df=trades_made_df
    )

    # NOTE - Results (Clean this up)
    pretty_print_json(metrics)
    pretty_print_json(trade_stats )
    create_folder(f'{backtest_output_path}{stock_name}/{model_type}/graphs/')
    bt.plot_equity_curve(bt_df, title="OOF Strategy Equity Curve (10d model)", save_path=f'{backtest_output_path}{stock_name}/{model_type}/graphs/equity_curve.png')
    bt.plot_positions(bt_df, title="OOF Position (10d model)", save_path=f'{backtest_output_path}{stock_name}/{model_type}/graphs/positions.png')
    # This not working
    # bt.save(bt_df, {**metrics, **trade_stats}, name=f'{model_type}_h{horizon_days}_long_only')



def run_backtests():

    ticker_list = [
        "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD",
        "XRP-USD", "USDC-USD", "ADA-USD", "DOGE-USD", "TRX-USD"
    ]
    for stock_name in ticker_list:
        backtest_model_performance(parquet_path=f'data/results/trained_models/ensembles/{stock_name}/XGBoost_V1/XGBoost_V1.parquet', 
                                    model_type='xgb',
                                    stock_hist_path='data/csv/historical/training/cleaned/BTC-USD.csv',
                                    stock_name=stock_name,
                                    backtest_output_path="data/results/backtests/")





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

    # NOTE - For running the entire pipelien in plan above 
    run_full_pipeline()


    # run_backtests()

    # NOTE - TESTING
    # stock_name='BTC-USD'
    # backtest_model_performance(parquet_path=f'data/results/trained_models/ensembles/{stock_name}/XGBoost_V1/XGBoost_V1.parquet', 
    #                             model_type='xgb',
    #                             stock_hist_path='data/csv/historical/training/cleaned/BTC-USD.csv',
    #                             stock_name=stock_name,
    #                             backtest_output_path="data/results/backtests/")


    # features_dict = load_JSON_object('data/json/stock_key_features.json')
    # stock_name='BTC-USD'
    # quantum_model_training(csv_path=f'data/csv/historical/training/cleaned/{stock_name}.csv', feat_list=features_dict[stock_name])


if __name__ == "__main__":
    main()
