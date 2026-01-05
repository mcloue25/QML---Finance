# Commonly used functions
import os 
import json 
import uuid

from datetime import datetime, timezone
from sklearn.model_selection import TimeSeriesSplit



def create_folder(folder_name: str):
    os.makedirs(folder_name, exist_ok=True)



def pretty_print_json(dict_object: object):
    ''' Makes JSON object prettier for being printed 
    '''
    json_object = json.dumps(dict_object)
    parsed_json = json.loads(json_object) 
    print(json.dumps(parsed_json, indent=4))



def save_JSON_object(json_object: object, json_path: str):
    ''' Saves a JSON Object to a path
    '''
    with open(json_path, 'w') as outfile:
        json.dump(json_object, outfile)


def load_JSON_object(json_path: str):
    ''' Loads JSON object from a file
    '''
    with open(json_path, "r") as f:
        json_data = json.load(f)
        return json_data
    

def subset_dict(data:dict, cols:list):
    ''' Used to subset results dicts for displaying a subset of data 
    '''
    return {name: data[name] for name in cols}



def generate_run_ID(symbol:str, model_type:str, horizon_days:int=10):
    '''  Generates a UUID for eahc backtest run for comparison against other runs
    Args:
        symbol (String) : Traded stock name
        model_type (String) : Model type being backtested
        horizon_days (Int) : Horizon for backtest (Default is 10)
    Returns:
        meta (Dict) : Dict containing UUID data for backtest 
    '''
    run_id = str(uuid.uuid4())
    meta = {
        "run_id": run_id,
        "symbol": symbol,
        "model_type": model_type,
        "horizon_days": horizon_days,
        "long_only": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # add your split dates if you have them
    }
    return meta 



def make_walkforward_splits(X, n_splits=5, gap=0):
    '''  Function for creating dataset splits so that the ensemble and qml models are traiend on identical splits
    Args:
        X (Tensor) : All data
        n_splits (Int) : Number of folds in the dataset
        gap (Int) : Space between to avoid leakage
    '''
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_idx, val_idx in tscv.split(X):
        if gap and gap > 0:
            train_idx = train_idx[:-gap] if len(train_idx) > gap else train_idx[:0]
        if len(train_idx) == 0:
            raise ValueError("Train set empty after applying gap")
        splits.append((train_idx, val_idx))
    return splits