import os 


from Classes.DataGenerator import DataDownloader




def download_data(output_path:str):
    ''' 
    '''
    ticker_list = [
        "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD",
        "XRP-USD", "USDC-USD", "ADA-USD", "DOGE-USD", "TRX-USD"
    ]
    dl = DataDownloader(ticker_list, output_path)
    dl.create_folder(output_path)
    dl.save_data(dl.stock_dict, dl.output_path)






def main():

    output_path = 'data/csv/'
    download_data(output_path=output_path)

if __name__ == "__main__":
    main()
