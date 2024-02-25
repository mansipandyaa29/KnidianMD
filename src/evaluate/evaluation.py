import sys

from src.exception import CustomException
from src.logger import log

class Evaluation():
    def __init__(self) -> None:
        pass
    def evaluate(self,y_truth,y_pred):
        '''
        Takes in the y_truth and y_pred array and returns the metrics
        Returns: F1_Score, Recall, Precision
        '''
        try:
            log.info(f'Calculating the precision, recall and f1 score for k from 1 to {len(y_pred)}')
            k_values = list(range(1, len(y_pred) + 2))
            precision_values,recall_values,f1_score_values = [],[],[]
            for k in k_values:
                precision = self.calculate_precision(y_truth, y_pred[:k])
                precision_values.append(precision)
                recall = self.calculate_recall(y_truth, y_pred[:k])
                recall_values.append(recall)
                f1_score = self.calculate_f1_score(precision, recall)
                f1_score_values.append(f1_score)
            log.info(f'Precision Values: {precision_values}')
            log.info(f'Recall Values: {recall_values}')
            log.info(f'F1 Values: {f1_score_values}')
            log.info(f'Finished calculating the precision, recall and f1 score for k from 1 to {len(y_pred)}')
            return precision_values, recall_values ,f1_score_values
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def calculate_precision(self,y_truth, y_pred):
        set_truth = set(y_truth)
        set_pred = set(y_pred)
        true_positives = len(set_truth.intersection(set_pred))
        false_positives = len(set_pred.difference(set_truth))
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        return precision

    def calculate_recall(self,y_truth, y_pred):
        set_truth = set(y_truth)
        set_pred = set(y_pred)
        true_positives = len(set_truth.intersection(set_pred))
        false_negatives = len(set_truth.difference(set_pred))
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        return recall

    def calculate_f1_score(self,precision,recall):
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score




    
