# Commonly used functions
import os 
import json 




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