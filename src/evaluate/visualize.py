import sys
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.logger import log


class Visualize():
    def __init__(self) -> None:
        pass

    def plot_prf(self, y_pred, precision_values, recall_values ,f1_score_values, plot_name):
        try:
            log.info(f'Plotting the precision, recall and f1 graph')
            plt.figure()

            k_values = list(range(1, len(y_pred) + 2))
            # Plotting precision, recall, and F1-score
            plt.plot(k_values, precision_values, label='Precision', marker='o')
            plt.plot(k_values, recall_values, label='Recall', marker='o')
            plt.plot(k_values, f1_score_values, label='F1-score', marker='o')

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
        
    def compare_metrics(self,sympoms_list,metric_list,model_type,metric_type):
        try:
            log.info(f'Plotting {metric_type} Comparison Graph')

            plt.figure()

            for i in range(3):
                plt.plot(list(range(1, len(sympoms_list[i]) + 2)), metric_list[i], label=f'{metric_type} {model_type[i]}', marker='o')

            k_val = list(range(1, len(max(sympoms_list, key=len)) + 1))
            plt.xlabel('k values')
            plt.ylabel('Score')
            plt.title('Precision, Recall, and F1-score vs. k')
            plt.xticks(k_val,rotation = 90)
            plt.legend()
            
            file_path = f"reports/{metric_type}/compare_{metric_type}.png"
            plt.savefig(file_path)

            log.info(f'Completed Plotting {metric_type} Comparison Graph')

        except Exception as e:
            raise CustomException(e,sys)
