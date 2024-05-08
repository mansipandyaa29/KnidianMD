from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from openai import OpenAI

# Binary search of word in list lword
def bsearch(word,lword):
    min=0
    max=len(lword)
    cur=int(max/2)
    while max>0:
        cur=min+int((max-min)/2)
        if word<lword[cur]:
            max=cur
        else:
            min=cur
        if cur<len(lword)-1:
            if word>=lword[cur] and word<lword[cur+1]:
                break
        else:
            if word>=lword[cur]:
                break
        if max==0:
            cur=-1
    return cur

def get_indices(word,skip_list,vocab):
    word=word.strip()
    if word in skip_list or word=='':
        pass
    if word not in vocab:
        max_score=[0,None]
        for j in vocab:
            tmp_score=SequenceMatcher(None,j,word).ratio()
            if tmp_score>max_score[0]:
                max_score[0]=tmp_score
                max_score[1]=j
        if max_score[0]>.82:
            word=max_score[1]
        else:
            pass
    pos=bsearch(word,vocab)
    return pos

def get_symptoms(indices,symptoms,symptoms_list):
    flat_indices_list = set([item for sublist in indices for item in sublist])
    for each_indice in flat_indices_list:
        symptoms_list.append(symptoms[each_indice])
    return symptoms_list

def top_symptoms(sorted_data,val):
    list_of_symptoms = []
    seen_symptom_id = []
    list_of_symptoms.append(sorted_data[0])
    seen_symptom_id.append(sorted_data[0][1])
    for i in range(1,len(sorted_data)):  
        
        if len(list_of_symptoms) == val:
            return list_of_symptoms
        
        if sorted_data[i][1] in seen_symptom_id:
            continue
        else:
            list_of_symptoms.append(sorted_data[i])

        seen_symptom_id.append(sorted_data[i][1])

def Embed_Sentence(medical_history,symptoms_list,hugging_face_model):
    model  = SentenceTransformer(hugging_face_model)
    embed_mh = model.encode(medical_history).reshape(1, -1)
    embedded_symptoms = np.array([model.encode(symptom[1]) for symptom in symptoms_list])
    similarity_vals = cosine_similarity(embed_mh, embedded_symptoms)
    similarities_scores = [[score, *symptom] for score, symptom in zip(similarity_vals[0], symptoms_list)]
    return similarities_scores


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message.content
        