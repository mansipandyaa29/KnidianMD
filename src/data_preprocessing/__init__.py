import sys

from preprocess_data import DataPreprocessing
from logger import log
import nltk
from nltk.corpus import stopwords

def main():
    lang_data = DataPreprocessing()
    data = lang_data.preprocess_langchain_data()
    print(data[:5])
    # log.info('Entering the Data Preprocessing Stage')
    # data = DataPreprocessing()

    # symptoms, vocab, itindx = data.preprocess_symptoms()
    # log.info(f'Initialization of symptoms ({len(symptoms)}), vocab ({len(vocab)}) and itindx ({len(itindx)}) complete')

    # nltk.download('stopwords')
    # stop_words = set(stopwords.words('english'))
    # stop_words.discard('no')
    # sentences,original_text = data.preprocess_medical_history(stop_words)
    # log.info(f"Medical History to analyse: {original_text}")
    # log.info(f"Lines to analyze: {sentences}")


    # return symptoms, vocab, itindx, sentences, original_text

if __name__ == "__main__":
    main()