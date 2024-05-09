import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from medcat.cat import CAT

   
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,text_data):
        try:
            model = CAT.load_model_pack("data/medcat/medcat_trained/medcat_model_pack_db2b2af3234d151a.zip")
            symptoms = []
            for i in range(len(model.get_entities(text_data)['entities'])):
                if i not in model.get_entities(text_data)['entities']:
                    continue
                each = model.get_entities(text_data)['entities'][i]
                symptoms.append(each['pretty_name'])
            return symptoms
        except Exception as e:
            raise CustomException(e,sys)