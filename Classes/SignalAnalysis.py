import os 

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression



class SignalAnalyser:
    ''' Function to analyse what economic signals should be used 
    '''
    def __init__(self, csv_path:str, build='core'):
        self.csv_path = csv_path
        self.build = build
        self.df = pd.read_csv(self.csv_path)
        self.df = self.preprocessing()
        self.create_folder('data/csv/historical/cleaned/')


    def create_folder(self, folder_name: str):
        os.makedirs(folder_name, exist_ok=True)


   