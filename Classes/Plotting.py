import os 
import json 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from itertools import cycle

sns.set(style='darkgrid')
colour_list = ["blue", "magenta", "red", "green", "gray", "lime", "maroon", "navy", "olive", "purple", "silver", "fuchsia", "teal", "yellow", "aqua", "black"]



def pretty_print_json(dict_object: object):
    ''' Makes JSON object prettier for being printed 
    '''
    json_object = json.dumps(dict_object)
    parsed_json = json.loads(json_object) 
    print(json.dumps(parsed_json, indent=4))



class Plotter:
    ''' Plotting Class for Graphs & Visualisations
    '''
    def __init__(self, path):
        self.path = path


    def plot_training_results(self, data:dict, title:str, output_folder:str, xaxis='X-Axis Filler', yaxis='Y-Axis Filler'):
        '''  Plotting function for analysing learning history
        Args:
            data (Dict) : Dict where K == Train / Val && V == Learning history for that signal
            title (String) : Title for plot
            output_folder (String) : Path to output location
            xaxis (String) : Title for X-Axis
            yaxis (String) : Title for Y-Axis
        '''
        colors = cycle(colour_list)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.suptitle(title, fontsize=16)

        for key, val in data.items():
            ax.plot(val, label= key, color=next(colors))

        ax.set_xlabel(xaxis, fontsize=15)
        ax.set_ylabel(yaxis, fontsize=18)
        ax.legend(loc="best")
        ax.margins(0.1)
        fig.tight_layout()
        plt.show()
        
        