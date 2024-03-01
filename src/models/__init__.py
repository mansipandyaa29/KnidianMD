import sys

from logger import log
from data_preprocessing import main as data_main
from model_architectures import Model

def main():
    log.info('Entering the Model Definition stage')
    symptoms, vocab, itindx, sentences, original_text = data_main()
    log.info("Data Loaded for Training")
    model = Model()
    id,symptoms_trad = model.traditional(symptoms, vocab, itindx, sentences)
    log.info(f'The extracted Symptoms using the Traditional method are: {symptoms_trad}')
    symptoms_trad_ss = model.trad_ss(symptoms, vocab, itindx, sentences)
    log.info(f'The extracted Symptoms using the Traditional and Sentence Similarity method are: {symptoms_trad_ss}')
    symptoms_ss = model.ss(symptoms, sentences)
    log.info(f'The extracted Symptoms using just the Sentence Similarity method are: {symptoms_ss}')
    

if __name__ == "__main__":
    main()
    