import os 
import yfinance as yf
from tqdm import tqdm

data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
print(data.head())



class DataDownloader:
    ''' Class for downloading videos from URLs (YouTube) links
    '''
    def __init__(self, tick_list:list, output_path:str):
        self.tick_list = tick_list
        self.output_path = output_path
        # Fetch all ticker data from yFinance
        self.stock_dict = self.fetch_ticker_data(tick_list)


    def create_folder(self, folder_name: str):
        os.makedirs(folder_name, exist_ok=True)

    
    def fetch_ticker_data(self, tick_list):
        ''' Fucntion to fetch historical data for a given tick symbol
        Args:
        Returns:
        '''
        stock_dict = {ticker : yf.download(ticker, period="max") for ticker in tick_list}
        return stock_dict
    

    def save_data(self, stock_dict:dict, output_path:str):
        ''' Save ticker data for all stocks to output path
        '''
        self.create_folder(output_path)
        for ticker_name, df in stock_dict.items():
            print(f'{output_path}{ticker_name}.csv')
            df.to_csv(f'{output_path}{ticker_name}.csv', index=False)  


        
    
# def test_main():
#     ticker_list = [
#         'BTC',
#         # 'ETH',
#         # 'USDT',
#         # 'XRP',
#         # 'BNB',
#         # 'USDC',
#         # 'SOL'
#     ]

#     output_path = 'data/'
#     downloader = DataDownloader(ticker_list, output_path)
#     downloader.create_folder(output_path)

#     downloader.download_videos()


# if __name__ == "__main__":
#     test_main()
