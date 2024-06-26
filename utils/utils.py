
from PyPDF2 import PdfReader
import ollama
import chromadb
import spacy
import math
from collections import Counter
import gzip
import pickle
import os

from typing import Any, List



nlp = spacy.load('es_core_news_sm')

MAX_CHUNK_SIZE = 512
STRIDE = 50

# Function to extract text from each page of the PDF
def extract_text_from_pdf(pdf_path: str) -> List:
    assert type(pdf_path) == str
    pdf_text = []
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)
        for page_number in range(num_pages):
            page = reader.pages[page_number]
            text = page.extract_text()
            pdf_text.append(text)
    return pdf_text

def clean_data_pdf(pdf_text:list)-> List:
    assert type(pdf_text) == list
    txt = token_txt(pdf_text)
    txt = txt.replace("CODIGO ORGANICO DE LA PRODUCCION, COMERCIO E INVERSIONES, COPCI - Página ", "")
    txt = txt.replace("LEXIS FINDER - www.lexis.com.ec", "")
    txt = txt.split('Art. ')
    return txt

def token_txt(token:list)->str:
    return ' '.join(token)

def get_chunks(txt:str, size:int =  MAX_CHUNK_SIZE, stride:int= STRIDE )->list:
    doc = nlp(txt)
    tokens = [token.text for token in doc]
    rounds = int(len(tokens) / size)
    data = [] 
    if rounds > 0:
        round_size_stride = math.ceil((len(tokens) -  size) / (size - stride) + 1)
    
        start = 0
        end = size
        
        for i in range(round_size_stride):
            token_aux = tokens[start: end]
            data.append(token_txt(token_aux))

            start = end - stride
            end = len(tokens) if i == (round_size_stride - 2)  else  start + size 

    data.append(token_txt(tokens))
    return data


def save_structure(data: any, path:str, file_name:str="chunks")->None:
    try:
        if not os.path.exists(path):
            os.makedirs(path)

        path_file = os.path.join(path, f"{file_name}.pkl.gz")
        # Serialize the data structure to bytes
        serialized_data = pickle.dumps(data)

        # Compress and save the serialized data to a file
        with gzip.open(path_file, 'wb') as file:
            file.write(serialized_data)

        print(f"Data structure has been saved to {path_file}")
    except:
        raise ("Error saving data")


def load_structure(path:str)->Any:

    try:
        # Load the compressed data from the file
        with gzip.open(path, 'rb') as file:
            compressed_data = file.read()

        # Deserialize the compressed data to reconstruct the original data structure
        loaded_data_structure = pickle.loads(compressed_data)
        print("loaded data structure")
        
        return loaded_data_structure
        

    except:
        raise ("Error loading data in file")
    

def get_similarity_score(vestor_a:List[float], vector_b:List[float])-> float:
        import numpy as np
        sim_score = np.dot(vestor_a, vector_b) / (
                np.linalg.norm(vestor_a) * np.linalg.norm(vector_b) + 1e-10
            )
        return sim_score

