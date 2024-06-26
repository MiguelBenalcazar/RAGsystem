from typing import Any
from utils.utils import load_structure
from chromadbClass.chromadb import Chromadb_Class
from BM25.BM25 import BM25, get_k_best_scores
from text.textProcess import NLP
from rerank_search.Process_search import Searching_results
import ollama
import gc


class RAG:
    def __init__(self) -> None:
        self.db = Chromadb_Class()
        self.documents, self.embeddings = self.db.get_documents()

        token = load_structure("./chunk_tokens/tokens.pkl.gz")
        self.BM25_search = BM25(token)

        self.nlp = NLP()
        self.filter = Searching_results()
    
    def __call__(
            self, 
            prompt:str, 
            number_best_bm25:int=10, 
            number_best_similarity:int=10,
            model:str='llama3'
            ) -> Any:
        
        token_prompt = self.nlp.process_text(prompt)
        scores = self.BM25_search.get_scores(token_prompt)
        data_extract, embedding_extract = get_k_best_scores(
            data = self.documents,
            embeddings=self.embeddings, 
            scores = scores, 
            best_k=number_best_bm25
            )
        simil_query = self.db.search_data_collection(
            prompt = prompt, 
            n_results = number_best_similarity)

        similarity_query = simil_query["documents"][0]
        similarity_embedding_query = simil_query["embeddings"][0]

        data_extract.extend(similarity_query)
        embedding_extract.extend(similarity_embedding_query)

        results_search = self.filter(
            documents = data_extract, 
            embeddings=embedding_extract, 
            query= prompt)
        
        # print(results_search['search_results'])
        
        output = ollama.generate(
            model="llama3",
            prompt=f" Given the data {results_search['search_results']} anwswer {prompt}. Do not use any prior knowledge nor your knowledge. \
                  # If you don't get any information related to the question in knowledge, say that you don't know.\
                  # Please be consistent and provide the shortest answer you can. Don't add any note nor any message that you get result base on the information proveided."
            )
        
        output1 = ollama.generate(
            model="phi3",
            prompt=f" Given the data {results_search['search_results']} anwswer {prompt}. Do not use any prior knowledge nor your knowledge. \
                   If you don't get any information related to the question in knowledge, say that you don't know.\
                   Please be consistent and provide the shortest answer you can. Don't add any note nor any message that you get result base on the information proveided."
            )
        
        # output = ollama.generate(
        #     model="llama3",
        #     prompt=f"Answer the query: {prompt} only using the information given in knowledge: {results_search['search_results']}.  \
        #           # Do not use any prior knowledge nor your knowledge. \
        #           # The given information in knowledge is sorted from most important to less important. \
        #           # If you don't get any information related to the question in knowledge, say that you don't know.\
        #           # Please be consistent and provide the shortest answer you can."
        #     )
        
        output2 = ollama.generate(
            model="llama3",
            prompt=f"Answer the query: {prompt} only using the information given in knowledge: {data_extract}.  \
                  # Do not use any prior knowledge nor your knowledge. \
                  # If you don't get any information related to the question in knowledge, say that you don't know.\
                  # Please be consistent and provide the shortest answer you can."
            )
        
        # output1 = ollama.generate(
        #     model="phi3",
        #     prompt=f"Answer the query: {prompt} only using the information given in knowledge: {results_search['search_results']}.  \
        #           # Do not use any prior knowledge nor your knowledge. \
        #           # The given information in knowledge is sorted from most important to less important. \
        #           # If you don't get any information related to the question in knowledge, say that you don't know.\
        #           # Please be consistent and provide the shortest answer you can."
        #     )
        
        output3 = ollama.generate(
            model="phi3",
            prompt=f"Answer the query: {prompt} only using the information given in knowledge: {data_extract}.  \
                  # Do not use any prior knowledge nor your knowledge. \
                  # If you don't get any information related to the question in knowledge, say that you don't know.\
                  # Please be consistent and provide the shortest answer you can."
            )
        
        # output = ollama.generate(
        #     model="llama3",
        #     prompt=f"Given the context information and not prior knowledge, \
        #           answer the query only use the information given. knowledge: {results_search['search_results']} the information provided includes similarity score and cluster, query: {prompt} be consist and give the shortest answer"
        #     )
        
        # output2 = ollama.generate(
        #     model="llama3",
        #     prompt=f"Given the context information and not prior knowledge. knowledge: {data_extract} the information provided includes similarity score and cluster, query: {prompt} be consist and give the shortest answer"
        #     )
        
        # output1 = ollama.generate(
        #     model="phi3",
        #     prompt=f"Given the context information and not prior knowledge, answer the query. knowledge: {results_search['search_results']} the information provided includes similarity score and cluster, query: {prompt} be consist and give the shortest answer"
        #     )
        
        
        # output3 = ollama.generate(
        #     model="phi3",
        #     prompt=f"Given the context information and not prior knowledge. knowledge: {data_extract} the information provided includes similarity score and cluster, query: {prompt} be consist and give the shortest answer"
        #     )
        
        gc.collect()
        
        return {
            'llama3_filter' : output['response'],
            'phi3_filter': output1['response'],
            'llama3' : output2['response'],
            'phi3': output3['response'],
            'result_search_filtered': results_search['search_results'],
            'Retrieved': data_extract


        }

        
