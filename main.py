import os 

from Classes.DataGenerator import DataDownloader
from Classes.FeatureEngineering import FeatureBuilder




def download_data(ticker_list:list, period:str, output_path:str):
    ''' Function for downloading all of the ticker data I want to follow 
    Args:
        output_path (String) : Path tot folder to save all CVS's to
    '''
    # period == max by default gets all daya 
    # period = '1y' getrs the last year for testing
    # dl = DataDownloader(ticker_list, period, output_path)
    dl = DataDownloader(ticker_list, "1y", output_path)
    dl.create_folder(output_path)
    dl.save_data(dl.stock_dict, dl.output_path)




def feature_engineering(folder_path:str):
    '''  
    '''
    for file in os.listdir(folder_path):
        fb = FeatureBuilder(f'{folder_path}{file}')
        fb.build_features(build='core')




def main():

    ticker_list = [
        "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD",
        "XRP-USD", "USDC-USD", "ADA-USD", "DOGE-USD", "TRX-USD"
    ]

    output_path = 'data/csv/testing/'

    # Download all ticker data
    # download_data(ticker_list, period="1y", output_path=output_path)

    # Engineer features for all tickers
    feature_engineering(output_path)

if __name__ == "__main__":
    main()
