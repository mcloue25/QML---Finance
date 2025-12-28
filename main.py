import os 

import pandas as pd

from Classes.DataGenerator import DataDownloader
from Classes.FeatureEngineering import FeatureBuilder
from Classes.SignalAnalysis import SignalAnalyser, ResultsAggregation



def create_folder(folder_name: str):
    os.makedirs(folder_name, exist_ok=True)



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
    ra = ResultsAggregation(results_dir=csv_folder, hist_path=f'{input_path}{stock}.csv', stock=stock, run_ids=run_ids, top_k=5, show_results=False)
    # ra.load_importance_files(csv_folder, pattern="*_perm_importance.csv")

    


def main():

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
    feature_analysis('data/csv/historical/cleaned/', stock='BTC-USD')





if __name__ == "__main__":
    main()
