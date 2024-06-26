from sentence_transformers import SentenceTransformer
import ollama

class Embeddings():
    def __init__(self, type:str = "all-MiniLM-L6-v2" ):
        self.type = type
        if type == "all-MiniLM-L6-v2":
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    
    
    def __call__(self, txt_list:list[str]) -> list[float]:
        if self.type == "all-MiniLM-L6-v2":
            return self.encoder.encode(txt_list)
        else:
            encoder = [ollama.embeddings(model=self.type, prompt=txt) for txt in txt_list]
            return encoder