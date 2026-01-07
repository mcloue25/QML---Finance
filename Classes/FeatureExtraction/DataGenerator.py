import os 
import yfinance as yf
from tqdm import tqdm



class DataDownloader:
    ''' Class for downloading historical stock data
    '''
    def __init__(self, tick_list:list, period:str, output_path:str):
        self.tick_list = tick_list
        self.output_path = output_path
        # Fetch all ticker data from yFinance
        self.stock_dict = self.fetch_ticker_data(tick_list, period)


    def create_folder(self, folder_name: str):
        os.makedirs(folder_name, exist_ok=True)
    

    def fetch_ticker_data(self, tick_list, time_frame):
        ''' Fucntion to fetch historical data for a given tick symbol
        Args:
            ticker_list (List) : List of stock symbols whos data we want to download
            time_frame (String) : period of data we want to download
        Returns:
            stock_dict (Dict) : Dict where k == Stock symbol && V == stock historical data
        '''
        stock_dict = {}
        pbar = tqdm(tick_list)
        for ticker in pbar:
            pbar.set_description(f"Downloading {ticker} historical data...")
            df = yf.download(
                ticker,
                period=time_frame,
                auto_adjust=False,   # keep Adj Close
                actions=True,        # add Dividends / Stock Splits when available
                group_by="column",
                progress=False,
                threads=True
            )

            # Standardise col names
            df = df.rename(columns={
                "Adj Close": "adj_close",
                "Close": "close",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Volume": "volume",
                "Dividends": "dividends",
                "Stock Splits": "stock_splits"
            })
            # Clean index and ordering
            df = df[~df.index.duplicated(keep="first")]
            df = df.sort_index()
            # Cleaning date col
            df = df.reset_index().rename(columns={"Date": "date"})
            stock_dict[ticker] = df

        return stock_dict
    

    def save_data(self, stock_dict:dict, output_path:str):
        ''' Save ticker data for all stocks to output path
        Args:
            stock_dict (Dict) : Dict where k == Stock symbol && V == stock historical data
            output_path (String) : String file path to save each stock historical data CSV to
        '''
        self.create_folder(output_path)
        for ticker_name, df in stock_dict.items():
            # print(f'{output_path}{ticker_name}.csv')
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
