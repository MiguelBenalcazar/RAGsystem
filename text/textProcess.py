import spacy
from typing import Any, List
from utils.utils import save_structure

class NLP:
    def __init__(self, language:str='en') -> None:
        languages = {'en': 'en_core_web_sm', 'es': 'es_core_news_sm'}
        self.language = language
        self.nlp = spacy.load(languages[language])

    
    def process_text(self, document:str)-> str:
    
        document = document.lower()
        doc = self.nlp(document)
    
        # Filter out stop words
        # filtered_words = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_bracket and not token.is_ascii]
        filtered_words = [token.text for token in doc if 
                          not token.is_stop and 
                          not token.is_punct and
                          not token.is_bracket and
                          not token.is_space ]
        return filtered_words

    def process_list(self, documents:List[str], save:bool=True)->List[Any]:
        tokens = [self.process_text(i) for i in documents]
        if save:
            save_structure(tokens, path="chunk_tokens", file_name="tokens")

        return tokens