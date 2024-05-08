import sys
import pandas as pd
import ast

from src.logger import log
from src.utils import load_object

from src.data_preprocessing import DataPreprocessing
from src.models import Model
from src.evaluate import Evaluation, Visualize

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.discard('no')

import logging
logging.getLogger('transformers').setLevel(logging.WARNING)



# log.info('Running logging from main.py file')

# log.info('Entering the Data Preprocessing Stage')
# data = DataPreprocessing()

# log.info("Preprocessing Symptoms Database")
# symptoms, vocab, itindx = data.preprocess_symptoms()
# lang_data = data.preprocess_langchain_data()
# log.info(f'Initialization of symptoms ({len(symptoms)}), vocab ({len(vocab)}) and itindx ({len(itindx)}) complete')

# log.info("Preprocessing Test Data")
# df = pd.read_csv('/Users/mansipandya/Desktop/KnidianMD/data/test_cases.csv')
# df['symptoms'] = df['symptoms'].apply(ast.literal_eval)
# df['Processed_Medical_History'] = df['medical_history'].apply(data.preprocess_medical_history, stop_words=stop_words)

# df.to_pickle('/Users/mansipandya/Desktop/KnidianMD/checkpoints/df_pmh.pkl')

# df = pd.read_pickle('/Users/mansipandya/Desktop/KnidianMD/checkpoints/df_pmh.pkl')

# log.info('Entering the Model Definition stage')
# model = Model()
# hugging_face_model = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
# df['traditional'] = df['Processed_Medical_History'].apply(lambda x: model.traditional(x[0], symptoms, vocab, itindx))
# df['traditional_ss'] = df['Processed_Medical_History'].apply(lambda x: model.trad_ss(x[0], symptoms, vocab, itindx,hugging_face_model))
# df['medcat'] = df['medical_history'].apply(model.medcat)
# df['langchain'] = df['Processed_Medical_History'].apply(lambda x: model.langchain(x[0]))
# df['langchainv2'] = df['Processed_Medical_History'].apply(lambda x: model.langchain_v2(x[0]))

# df.to_pickle('/Users/mansipandya/Desktop/KnidianMD/checkpoints/df_model.pkl')
# df = pd.read_pickle('/Users/mansipandya/Desktop/KnidianMD/checkpoints/df_model.pkl')

# log.info('Entering the Evaluation Stage')
# evaluate = Evaluation()
# df['precision_trad'], df['recall_trad'], df['f1_trad'] = zip(*df.apply(lambda row: evaluate.evaluate(row['symptoms'], row['traditional']), axis=1))
# df['precision_trad_ss'], df['recall_trad_ss'], df['f1_trad_ss'] = zip(*df.apply(lambda row: evaluate.evaluate(row['symptoms'], row['traditional_ss']), axis=1))
# df['precision_medcat'], df['recall_medcat'], df['f1_medcat'] = zip(*df.apply(lambda row: evaluate.evaluate(row['symptoms'], row['medcat']), axis=1))
# df['precision_lang'], df['recall_lang'], df['f1_lang'] = zip(*df.apply(lambda row: evaluate.evaluate(row['symptoms'], row['langchain']), axis=1))
# df['precision_langv2'], df['recall_langv2'], df['f1_langv2'] =  zip(*df.apply(lambda row: evaluate.evaluate(row['symptoms'], row['langchainv2']), axis=1))

# df.to_pickle('/Users/mansipandya/Desktop/KnidianMD/checkpoints/df_prf_pickle.pkl')
df = pd.read_pickle('/Users/mansipandya/Desktop/KnidianMD/checkpoints/df_prf_pickle.pkl')


# log.info('Entering the Visualization Stage')
visualize = Visualize()
# visualize.plot_prf(df,'precision_trad','recall_trad','f1_trad', plot_name = "traditional")
# visualize.plot_prf(df,'precision_trad_ss','recall_trad_ss','f1_trad_ss', plot_name = "traditional_ss")
# visualize.plot_prf(df,'precision_medcat','recall_medcat','f1_medcat', plot_name = "medcat")
# visualize.plot_prf(df,'precision_lang','recall_lang','f1_lang', plot_name = "langchain")
# visualize.plot_prf(df,'precision_langv2','recall_langv2','f1_langv2', plot_name = "langchainv2")

model_type = ['trad','trad + ss','medcat','langchain','langv2']

visualize.compare_metrics(df,model_type,metric_type='precision')
visualize.compare_metrics(df,model_type,metric_type='recall')
visualize.compare_metrics(df,model_type,metric_type='f1')




