import sys
from evaluation import Evaluation
from visualize import Visualize
from logger import log
from models import main as models_main
import pandas as pd


def main():
    df = pd.read_csv('data/test_cases.csv')
    y_truth = df['symptoms'].apply(lambda x: x.split(', ')).tolist()[0]

    log.info('Entering the Evaluation Stage')
    symptoms_trad,symptoms_ss = models_main()

    log.info('Symptoms loaded from the Modeling Stage')

    evaluate = Evaluation()
    visualize = Visualize()

    precision_trad, recall_trad, f1_score_trad = evaluate.evaluate(y_truth,symptoms_trad)
    visualize.plot_prf(symptoms_trad, precision_trad, recall_trad, f1_score_trad, plot_name = "traditional")

    precision_ss, recall_ss, f1_score_ss = evaluate.evaluate(y_truth,symptoms_ss)
    visualize.plot_prf(symptoms_ss, precision_ss, recall_ss, f1_score_ss, plot_name = "ss")

    #Compare Precision
    visualize.compare_metrics(symptoms_trad,symptoms_ss,precision_trad,precision_ss,metric_type="Precision")
    #Compare Recall
    visualize.compare_metrics(symptoms_trad,symptoms_ss,recall_trad,recall_ss,metric_type="Recall")
    #Compare F1 Score
    visualize.compare_metrics(symptoms_trad,symptoms_ss,f1_score_trad,f1_score_ss,metric_type="F1")
    

if __name__ == "__main__":
    main()