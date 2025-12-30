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
        fb = FeatureBuilder(f'{folder_path}{file}')
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
    ''' 
    '''
    # NOTE - 
    # generate_importance_statistics(input_path, stock)

    # Compare feature results
    csv_folder = f'data/csv/feature_analysis/{stock}/'
    run_ids = os.listdir(csv_folder)
    fa = FeatureAnalyser(results_dir=csv_folder, hist_path=f'{input_path}{stock}.csv', stock=stock, run_ids=run_ids, top_k=5, show_results=True)



    


def DL_model_training(csv_path):
    ''' Main function for calling trianing pipelines for different kinds of models

    '''
    # Load and split dataset 
    X, y, dates = split_train_val_test(
        file_path=csv_path, 
        feature_cols=['rolling_mean_return_10d', 'price_ma_ratio_20d_centered', 'volatility_20d'], 
        target_col='signal_voladj_10d'
    )
    create_folder('data/results/')

    # X, y already aligned for a given horizon (e.g. y_20)
    X_dev, y_dev, dates_dev, X_test, y_test, dates_test = time_train_test_split(X, y, dates, test_size=0.2)

    # Train LGBM Classifier
    lgbm = load_LGBM_Classifier()
    lgbm_results = ensemble_train_loop(lgbm, X_dev, y_dev, X_test, y_test)
    analyse_ensemble_results(lgbm_results, y_test, dates_dev, y_dev, 'LGBM_V1', output_path='data/results/LGBM_V1/')
    
    # Train XGBoost Classifier
    xgb = loadXGBoost_Classifier()
    xgb_results = ensemble_train_loop(xgb, X_dev, y_dev, X_test, y_test)
    analyse_ensemble_results(xgb_results, y_test, dates_dev, y_dev, 'XGBoost_V1', output_path='data/results/XGBoost_V1/')



def backtest_model_performance(parquet_path, stock_hist_path:str):
    ''' Run OOF backtest on trained model to evaluate performance  
    '''
    oof_df = pd.read_parquet(parquet_path)
    prices = pd.read_csv(stock_hist_path, parse_dates=["date"]).set_index("date")["close"]

    cfg = BacktestConfig(
        horizon_days=10,
        transaction_cost=0.0005,
        mode="long_only",
        hold_rule="signal_until_change"   # or "fixed_horizon"
    )

    bt = BackTest(path="results/backtests", config=cfg)

    bt_df = bt.run(oof_df, prices)
    metrics = bt.performance_metrics(bt_df)
    trades = bt.trade_stats(bt_df)

    pretty_print_json(metrics)
    pretty_print_json(trades)

    bt.plot_equity_curve(bt_df, title="OOF Strategy Equity Curve (10d model)")
    bt.plot_positions(bt_df, title="OOF Position (10d model)")

    bt.save(bt_df, {**metrics, **trades}, name="xgb_h10_long_only")




def main():
    '''  
    '''
    ticker_list = [
        "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD",
        "XRP-USD", "USDC-USD", "ADA-USD", "DOGE-USD", "TRX-USD"
    ]

    download_path = 'data/csv/testing/'

    # Download all ticker data
    # download_data(ticker_list, period="max", output_path=download_path)

    # Engineer features for all tickers
    # feature_engineering(download_path, output_path='data/csv/historical/cleaned/')

    # Analyse Feature importance for modelling
    # feature_analysis('data/csv/historical/cleaned/', stock='BTC-USD')

    csv_path = 'data/csv/historical/cleaned/BTC-USD.csv'
    # DL_model_training(csv_path)


    backtest_model_performance('data/results/XGBoost_V1/XGBoost_V1.parquet', 'data/csv/historical/cleaned/BTC-USD.csv')




if __name__ == "__main__":
    main()
