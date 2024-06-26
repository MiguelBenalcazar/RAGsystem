import spacy

def token_length(text: str, language: str="en") -> int:
    nlp = None
    if language == 'en':
        nlp = spacy.load('en_core_web_sm')
    elif language == 'es':
        nlp = spacy.load('es_core_news_sm')

    doc = nlp(text)
    return len(doc)