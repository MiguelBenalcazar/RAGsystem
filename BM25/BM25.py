from collections import Counter
import math
from typing import Any, List
import heapq


class BM25():
    def __init__(self, corpus, k1:float=1.5, b:float=0.75 )->None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc) for doc in corpus) / len(corpus)
        self.doc_freqs = []
        self.doc_len = []
        self.idf = {}
        self.initialize()

    def initialize(self)-> None:
        df = {}
        for doc in self.corpus:
            self.doc_len.append(len(doc))
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for term in frequencies:
                if term not in df:
                    df[term] = 0
                df[term] += 1

        for term, freq in df.items():
            self.idf[term] = math.log(1 + (len(self.corpus) - freq + 0.5) / (freq + 0.5))   

    def _score(self, query, index)-> float:
        score = 0.0
        doc = self.doc_freqs[index]
        doc_length = self.doc_len[index]
        for term in query:
            if term in doc:
                freq = doc[term]
                idf = self.idf.get(term, 0)
                score += idf * freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl))
        return score

    def get_scores(self, query)->List[float]:
        scores = []
        for index in range(len(self.corpus)):
            scores.append(self._score(query, index))
        return scores 
    
def get_k_best_scores(data:List[str], embeddings:List[float], scores:List[float], best_k:int=10)-> (List[Any], List[Any]):
        
    k_scores = heapq.nlargest(best_k, enumerate(scores), key=lambda x: x[1])
    k_scores_index = [index for index, _ in k_scores]
    data_extract = [data[i] for i in k_scores_index ]
    embedding_extract = [embeddings[i] for i in k_scores_index ]
    return data_extract, embedding_extract