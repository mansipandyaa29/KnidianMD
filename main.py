import sys
import nltk
from nltk.corpus import stopwords
import pandas as pd

from src.logger import log
from src.utils import load_object

from src.data_preprocessing import DataPreprocessing
from src.models import Model
from src.evaluate import Evaluation, Visualize

import logging
logging.getLogger('transformers').setLevel(logging.WARNING)

log.info('Running logging from main.py file')

log.info('Entering the Data Preprocessing Stage')
data = DataPreprocessing()
symptoms, vocab, itindx = data.preprocess_symptoms()
log.info(f'Initialization of symptoms ({len(symptoms)}), vocab ({len(vocab)}) and itindx ({len(itindx)}) complete')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.discard('no')
sentences,original_text = data.preprocess_medical_history(stop_words)
log.info(f"Medical History to analyse: {original_text}")
log.info(f"Lines to analyze: {sentences}")



log.info('Entering the Model Definition stage')
model = Model()

symptoms_trad = load_object('checkpoints/traditional.pkl')
log.info(f'The extracted Symptoms using the Traditional method are: {symptoms_trad}')

symptoms_trad_ss = load_object('checkpoints/trad_ss.pkl')
log.info(f'The extracted Symptoms using the Traditional and Sentence Similarity method are: {symptoms_trad_ss}')

log.info('Entering the Evaluation Stage')

df = pd.read_csv('data/test_cases.csv')
y_truth = df['symptoms'].apply(lambda x: x.split(', ')).tolist()[0] #7 values

evaluate = Evaluation()
visualize = Visualize()

precision_trad, recall_trad, f1_score_trad = evaluate.evaluate(y_truth,symptoms_trad)
visualize.plot_prf(symptoms_trad, precision_trad, recall_trad, f1_score_trad, plot_name = "traditional")

precision_trad_ss, recall_trad_ss, f1_score_trad_ss = evaluate.evaluate(y_truth,symptoms_trad_ss)
visualize.plot_prf(symptoms_trad_ss, precision_trad_ss, recall_trad_ss, f1_score_trad_ss, plot_name = "trad_ss")

# precision_ss, recall_ss, f1_score_ss = evaluate.evaluate(y_truth,symptoms_ss)
# visualize.plot_prf(symptoms_ss, precision_ss, recall_ss, f1_score_ss, plot_name = "ss")

#Compare Precision
visualize.compare_metrics(symptoms_trad,symptoms_trad_ss,precision_trad,precision_trad_ss,metric_type="Precision")
#Compare Recall
visualize.compare_metrics(symptoms_trad,symptoms_trad_ss,recall_trad,recall_trad_ss,metric_type="Recall")
#Compare F1 Score
visualize.compare_metrics(symptoms_trad,symptoms_trad_ss,f1_score_trad,f1_score_trad_ss,metric_type="F1")





