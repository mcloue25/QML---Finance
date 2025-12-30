import os 
import json 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from itertools import cycle
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid", context="talk")
COLOUR_LIST = ["blue", "magenta", "red", "green", "gray", "lime", "maroon", "navy", "olive", "purple", "silver", "fuchsia", "teal", "yellow", "aqua", "black"]


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


    def plot_train_val(self, data:dict, title:str, output_folder:str, xaxis='X-Axis Filler', yaxis='Y-Axis Filler'):
        '''  Plotting function for analysing learning history
        Args:
            data (Dict) : Dict where K == Train / Val && V == Learning history for that signal
            title (String) : Title for plot
            output_folder (String) : Path to output location
            xaxis (String) : Title for X-Axis
            yaxis (String) : Title for Y-Axis
        '''
        colors = cycle(COLOUR_LIST)
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
        plt.close()
        

    def plot_hist(self, data:dict, title:str, output_folder:str=None, xaxis='X-Axis Filler', yaxis='Y-Axis Filler', show=False, save_path=None):
        ''' Plot bar plot of results from training ensemble model
        Args:
            data (Dict) : Dict containing data for the X-Axis & Y-Axis
            output_folder (String) : Path to output file
            xaxis (String) : Name of X-Axis column
            yaxis (String) : Name of Y-Axis column
            show (Bool) : Display image or not 
            save_path (String) : Save image to file or not
        '''
        df = pd.DataFrame(data)
        df["fold"] = df["fold"].astype(str)
        plt.figure(figsize=(7, 4))
        sns.barplot(data=df, x="fold", y="log_loss")

        plt.title(title, fontsize=14, pad=10)
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)

        ymin = df["log_loss"].min() * 0.98
        ymax = df["log_loss"].max() * 1.02
        plt.ylim(ymin, ymax)

        for i, v in enumerate(df["log_loss"]):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()



    def plot_confusion_matrices(self, y_true, y_pred, labels=None, class_names=None, title_prefix="", figsize=(12, 5), show=False, save_path=None):
        '''Plot raw and row-normalized confusion matrices side by side.
            Rows = true labels
            Cols = predicted labels
        '''
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))

        if class_names is None:
            class_names = [str(l) for l in labels]

        # Raw confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Row-normalised confusion matrix
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.set_theme(style="white", context="talk")
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Raw Confusion Matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=class_names, yticklabels=class_names, ax=axes[0], cbar=False,)
        axes[0].set_title(f"{title_prefix} Confusion Matrix (Counts)")
        axes[0].set_xlabel("Predicted label")
        axes[0].set_ylabel("True label")

        # Normlised Confusion Matrix
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="viridis", xticklabels=class_names, yticklabels=class_names, ax=axes[1], cbar=False)
        axes[1].set_title(f"{title_prefix} Confusion Matrix (Normalized)")
        axes[1].set_xlabel("Predicted label")
        axes[1].set_ylabel("True label")

        plt.tight_layout()
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.close()

        