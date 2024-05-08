import sys

from .model_utils import get_indices, get_symptoms, top_symptoms, Embed_Sentence, get_completion_from_messages
from ..exception import CustomException
from ..logger import log
from ..utils import save_object

from difflib import SequenceMatcher
import pickle

import getpass
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

from medcat.cat import CAT


class Model:
    def __init__(self) -> None:
        pass

    def traditional(self, sentences, symptoms, vocab, itindx):
        '''
        Defines the traditional model which is just word matching
        returns symptoms
        '''
        try:
            log.info("Entering the Traditional Model Architecture")
            skip_list=['','of','the','a','and','on','with','in','or','to','for','by','but','years','old','personal','lasting','symptoms','reports']
            id_line=[]
            sn_line=[]
            for sentence in sentences:
                id_line.append([])
                sn_line.append([])
                sentence=sentence.replace(","," ").replace(";"," ").replace(":"," ").replace("-"," ").replace("\""," ").replace("?"," ").replace("/"," ").replace("("," ").replace(")"," ").replace("["," ").replace("]"," ").replace("&"," ")
                words=sentence.lower().split(' ')
                for word in words:
                    
                    pos = get_indices(word,skip_list,vocab)
                    # i: index of terms[] where word is present
                    jk=[0,[]]
                    for i in itindx[pos]:
                        # if terms[i][1].lower().strip() is one word instead of a sentence
                        if symptoms[i][1].lower().strip()==word:
                            jk[0]=1
                            jk[1]=[i]
                            flag=1
                            continue

                        # If symptoms[i][1].lower().strip() is a sentence
                        tmp = symptoms[i][1].split(' ')
                        w_terms=[x for x in tmp if x not in skip_list and x!=word]
                        tmp=sentence.lower().split(' ')
                        w_chlin=[x for x in tmp if x not in skip_list and x!=word]
                        # Matching thresholds for the rest of words in terms to be compared to current words in line (up to 4)
                        lim=[.82,.67,.55]
                        flag=1
                        for h in range(0,min(3,len(w_terms))):
                            max_score=[0,None,None]
                            for j in w_terms[0:5]:
                                for k in w_chlin:
                                    tmp_score = SequenceMatcher(None,j,k).ratio()
                                    if tmp_score>max_score[0]:
                                        max_score[0]=tmp_score
                                        max_score[1]=j
                                        max_score[2]=k

                            if max_score[0]<lim[h]:
                                flag=0
                                break
                            else:
                                w_terms=[x for x in w_terms if x!=max_score[1]]
                                w_chlin=[x for x in w_chlin if x!=max_score[2]]

                        if flag==1:
                            jk[1].append(i)

                    for j in jk[1]:
                        if symptoms[j][0] not in id_line[-1]:
                            id_line[-1].append(symptoms[j][0])
                            sn_line[-1].append(symptoms[j][1])

                #symptoms for a single sentence
                # print(f"\n*** Symptoms found: {sn_line[-1]}")

                # Remove generic symptoms in line and keep specific ones:
                tmp_ind=[]
                for j in sn_line[-1]:
                    w_j=[x for x in j.split(' ') if x not in skip_list]
                    comp=[x for x in sn_line[-1] if x!=j]
                    rem_j=0
                    for k in comp:
                        w_k=[x for x in k.split(' ') if x not in skip_list]
                        if 'history' in w_k:
                            continue
                        if len(w_k)>5:
                            continue
                        nwj=0
                        for l in w_j:
                            if l in w_k:
                                nwj+=1
                        if nwj==len(w_j):
                            rem_j=1
                            break
                    if rem_j==1:
                        # print(f">>> Removing less specific '{j}'")
                        tmp_ind=[sn_line[-1].index(j)]+tmp_ind
                for j in tmp_ind:
                    del id_line[-1][j]
                    del sn_line[-1][j]

            id_final=[]
            sn_final=[]
            for i in range(0,len(id_line)):
                for j in range(0,len(id_line[i])):
                    if id_line[i][j] not in id_final:
                        id_final.append(id_line[i][j])
                        sn_final.append(sn_line[i][j])

            #create a dictionary of the two
            symptom_dict = {id_final[i]: sn_final[i] for i in range(len(sn_final))}
            return symptom_dict

        except Exception as e:
            raise CustomException(e,sys)
        
    def trad_ss(self, sentences, symptoms, vocab, itindx, hugging_face_model):
        '''
        Defines the sentence similarity model which is a combination of word matching and sentence similarity
        returns symptoms
        '''
        try:
            log.info("Entering the Sentence Similarity Model Architecture")
            final_symptoms_list = []
            skip_list=['','of','the','a','and','on','with','in','or','to','for','by','but','years','old','personal','lasting','symptoms','reports']
            for sentence in sentences:
                indices = []
                symptoms_list = []
                if sentence == '':
                    continue
                words=sentence.split()
                for word in words:
                    pos = get_indices(word,skip_list,vocab)
                    indices.append(itindx[pos])
                symptoms_list = get_symptoms(indices,symptoms,symptoms_list)
                if len(symptoms_list)<3:
                    continue
                similarity_scores= Embed_Sentence(sentence,symptoms_list,hugging_face_model)
                sorted_data = sorted(similarity_scores, key=lambda x: x[0], reverse=True)
                #get top three symptoms for each sentence
                list_of_symptoms = top_symptoms(sorted_data,3)
                #final_symptoms_list is append for every sentence with top 3 symptoms
                for each_value in list_of_symptoms:
                    final_symptoms_list.append(each_value)
            #once the final_symptoms_list is appended with 3 symptoms from all the sentences we sort it again
            sorted_symptoms = sorted(final_symptoms_list, key=lambda x: x[0], reverse=True)
            #get the top 25 symptoms names from sorted_symptoms
            result_symptoms_name = [tup[2] for tup in sorted_symptoms[:25]]
            #get the top 25 symptom numbers from sorted_symtoms
            result_symptoms_id = [tup[1] for tup in sorted_symptoms[:25]]
            #create a dictionary of the two
            symptom_dict = {result_symptoms_id[i]: result_symptoms_name[i] for i in range(len(result_symptoms_name))}

            return symptom_dict

        except Exception as e:
            raise CustomException(e,sys)
        
    def medcat(self,document):
        symptom_dict = dict()
        cat = CAT.load_model_pack("/Users/mansipandya/Desktop/KnidianMD/data/medcat/medcat_trained/medcat_model_pack_db2b2af3234d151a.zip")
        for i in range(len(cat.get_entities(document)['entities'])):
            if i not in cat.get_entities(document)['entities']:
                continue
            each = cat.get_entities(document)['entities'][i]
            symptom_dict[int(each['cui'])] = each['pretty_name']
        return symptom_dict
        
    def ss(self, symptoms, sentences):
        '''
        Defines the the basic sentence similarity model
        returns symptoms
        '''
        try:
            log.info("Entering the Sentence Similarity Model Architecture")
            final_symptoms_list = []
            for sentence in sentences:
                if sentence == '':
                    continue
                similarity_scores= Embed_Sentence(sentence,symptoms)
                sorted_data = sorted(similarity_scores, key=lambda x: x[0], reverse=True)
                list_of_symptoms = top_symptoms(sorted_data,3)
                for each_value in list_of_symptoms:
                    final_symptoms_list.append(each_value)
            sorted_symptoms = sorted(final_symptoms_list, key=lambda x: x[0], reverse=True)
            result_symptoms = [tup[2] for tup in sorted_symptoms[:25]]

            save_object('checkpoints/ss.pkl',result_symptoms)

            return result_symptoms

        except Exception as e:
            raise CustomException(e,sys)

    def langchain(self, sentences):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key
        embedding = OpenAIEmbeddings()
        persist_directory = '/Users/mansipandya/Desktop/KnidianMD/docs/chroma/'
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        symptom_list = []
        symptom_id_list = []
        k_number=5
        for sentence in sentences:
            if sentence == '':
                continue
            question = f"{sentence}. What symptoms in the database are present in this sentence?"
            docs = vectordb.max_marginal_relevance_search(question,k=k_number, fetch_k=10)
            for i in range(k_number):
                text = docs[i].page_content
                lines = text.split('\n')
                for line in lines:
                    if line.startswith('id:') and line!= 'id: id':
                        symptom_id = line.split(': ', 1)[1]
                        symptom_id_list.append(int(symptom_id))
                    if line.startswith('symptom:') and line != 'symptom: symptom':
                        symptom = line.split(': ', 1)[1]
                        symptom_list.append(symptom)

        #create a dictionary of the two
        symptom_dict = {symptom_id_list[i]: symptom_list[i] for i in range(len(symptom_list))}
        save_object('checkpoints/langchain.pkl',symptom_dict)
        
        return symptom_dict
    
    def langchain_v2(self, sentences):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key
        embedding = OpenAIEmbeddings()
        persist_directory = '/Users/mansipandya/Desktop/KnidianMD/docs/chroma/'
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        symptom_list = []
        symptom_id_list = []
        k_number=5
        for sentence in sentences:
            if sentence == '':
                continue
            medical_history = sentence
            delimiter = "####"
            system_message = f"""
            You will be provided with a question. \
            The question will be delimited with {delimiter} characters. \
            Output a python list, where each entry in the list is a symptom present in the medical history text below \

            {medical_history}

            If a symptoms is mentioned, it must be associated with it must be present in the medical history above. \
            If no symptoms are found, output an empty list. \

            Only output the list of objects, with nothing else.
            """
            user_message_1 = f"""What symptoms are present in the medical history?"""

            messages =  [  
            {'role':'system', 
            'content': system_message},    
            {'role':'user', 
            'content': f"{delimiter}{user_message_1}{delimiter}"},  
            ] 

            response = get_completion_from_messages(messages)

            response = response.strip('[]')
            response = response.split(', ')
            response = [symptom.strip("'") for symptom in response]

            for symptom in response:
                question = f"{symptom}. What symptoms in the database are present in the previous sentence?"
                docs = vectordb.max_marginal_relevance_search(question,k=k_number, fetch_k=10)
                for i in range(k_number):
                    text = docs[i].page_content
                    lines = text.split('\n')
                    for line in lines:
                        if line.startswith('id:') and line!= 'id: id':
                            symptom_id = line.split(': ', 1)[1]
                            symptom_id_list.append(int(symptom_id))
                        if line.startswith('symptom:') and line != 'symptom: symptom':
                            symptom = line.split(': ', 1)[1]
                            symptom_list.append(symptom)

        #create a dictionary of the two
        symptom_dict = {symptom_id_list[i]: symptom_list[i] for i in range(len(symptom_list))}
                    
        return symptom_dict
            

        




