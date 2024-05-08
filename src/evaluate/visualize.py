import sys
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.logger import log

import numpy as np


class Visualize():
    def __init__(self) -> None:
        pass

    def plot_prf(self, df, precision, recall, f1, plot_name):
        try:
            log.info(f'Plotting the precision, recall and f1 graph')
            max_length = df[precision].apply(len).max()
            padded_lists = df[precision].apply(lambda x: x + [0] * (max_length - len(x)))
            average_precision_list = np.mean(padded_lists.tolist(), axis=0)

            max_length = df[recall].apply(len).max()
            padded_lists = df[recall].apply(lambda x: x + [0] * (max_length - len(x)))
            average_recall_list = np.mean(padded_lists.tolist(), axis=0)

            max_length = df[f1].apply(len).max()
            padded_lists = df[f1].apply(lambda x: x + [0] * (max_length - len(x)))
            average_f1_list = np.mean(padded_lists.tolist(), axis=0)

            plt.figure()

            k_values = list(range(1, len(average_precision_list) + 1))

            # Plotting precision, recall, and F1-score
            plt.plot(k_values, average_precision_list, label='Precision', marker='o')
            plt.plot(k_values, average_recall_list, label='Recall', marker='o')
            plt.plot(k_values, average_f1_list, label='F1-score', marker='o')

            # Adding labels and title
            plt.xlabel('k values')
            plt.ylabel('Score')
            plt.title('Precision, Recall, and F1-score vs. k')
            plt.xticks(range(1, len(k_values) + 1),rotation = 90)
            plt.legend()
            
            file_path = f"reports/prf_graphs/{plot_name}_prf.png"
            plt.savefig(file_path)

            log.info(f'Plot for the precision, recall and f1 graph saved in {file_path}')

        except Exception as e:
            raise CustomException(e,sys)
        
    def compare_metrics(self,df,model_type,metric_type):
        try:
            log.info(f'Plotting {metric_type} Comparison Graph')

            plt.figure()
            plt.figure(figsize=(10, 10))

            columns = [col for col in df.columns if col.startswith(metric_type)]
            max_val = 0
            for i,column in enumerate(columns):
                max_length = df[column].apply(len).max()
                max_val = max(max_val,max_length)
                padded_lists = df[column].apply(lambda x: x + [0] * (max_length - len(x)))
                average_list = np.mean(padded_lists.tolist(), axis=0)
                plt.plot(list(range(1, len(average_list) + 1)), average_list,label=f'{metric_type} {model_type[i]}', marker='o')
   
            plt.xlabel('k values')
            plt.ylabel('Score')
            plt.title('Precision, Recall, and F1-score vs. k')
            plt.xticks(list(range(1, max_val + 1)),rotation = 90)
            plt.legend()
            
            file_path = f"reports/{metric_type}/compare_{metric_type}.png"
            plt.savefig(file_path)

            log.info(f'Completed Plotting {metric_type} Comparison Graph')

        except Exception as e:
            raise CustomException(e,sys)
