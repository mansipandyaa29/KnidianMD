import os
import sys
from src.logger import log
from src.exception import CustomException
from data_utils import bsearch,binsert

import pandas as pd
import re
from langchain_community.document_loaders.csv_loader import CSVLoader


class DataPreprocessing:
    def __init__(self) -> None:
        pass

    def preprocess_langchain_data(self):
        loader = CSVLoader(
        file_path="/Users/mansipandya/Desktop/KnidianMD/data/symptoms_db.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["id", "symptom"],
        },
        )
        data = loader.load()
        return data

    def preprocess_symptoms(self):
        '''
        Reads the data from symptoms_db.csv and splits it into symptoms, vocab and itindx
        Returns: Path to Symptoms, Vocab and Itindx
        '''
        log.info("Entered the preprocess symptoms method")
        try:
            log.info('Reading the symptoms database from symptoms_db.csv as a dataframe')
            df = pd.read_csv('data/symptoms_db.csv')
            symptoms=list()
            vocab=list()
            itindx=list()
            skip_list=['','of','the','a','and','on','with','in','or','to','for','by','but','years','old','personal','lasting','symptoms','reports']

            log.info('Splitting the symptom database into symptoms, vocab, itindx initiated')
            for index, row in df.iterrows():
                # Add term to list of symptoms
                symptoms.append([int(row['id']), row['symptom'].lower().lstrip()])
                
                # Remove special characters from term
                tmp_list = row['symptom'].lower().lstrip().replace(",", " ").replace(";", " ").replace(":", " ").replace(".", " ").replace("-", " ").replace("\"", " ").replace("?", " ").replace("/", " ").replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ").replace("&", " ").lstrip().split(" ")
                
                # Iterate words in symptoms
                for i in tmp_list:
                    i = i.strip()
                    if i in skip_list or i == '':
                        continue
                    pos = bsearch(i, vocab)
                    if i not in vocab:
                        vocab = binsert(i, vocab, pos)
                        itindx = binsert([len(symptoms) - 1], itindx, pos)
                    else:
                        itindx[pos].append(len(symptoms) - 1)

            log.info('Splitting the symptom database into symptoms, vocab, itindx completed')
            return symptoms, vocab, itindx

        except Exception as e:
            raise CustomException(e,sys)
        
    def preprocess_medical_history(self,stop_words):
        '''
        Reads the data from test_cases.txt and splits it into sentences
        Returns: Sentences and Original text
        '''
        log.info("Entered the preprocess symptoms method")
        try:
            log.info('Reading the medical history from test_cases.csv as a dataframe')
            df = pd.read_csv('data/test_cases.csv')
            log.info('Preprocessing the medical history initiated')
            original_text = df.iloc[0].medical_history
            original = df.iloc[0].medical_history.lower() 
            pattern = r'\b\d+(\.\d+)?\s*\w+/\w+\b'
            original = re.sub(pattern, '.', original)
            original = original.replace('-', ' ')
            original = ''.join(char for char in original if char.isalpha() or char.isspace() or char == '.')
            words = original.split()
            filtered_words = [word for word in words if word not in stop_words]
            cleaned_text = ' '.join(filtered_words)
            sentences = cleaned_text.split('.')
            cleaned_sentences = [sentence.strip() for sentence in sentences]
            log.info('Preprocessing the medical history completed')
            return cleaned_sentences,original_text

        except Exception as e:
            raise CustomException(e,sys)
        


        

        


    