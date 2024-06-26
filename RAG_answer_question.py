import pandas as pd
from RAG_Basic.RAG import RAG
from tqdm.auto import tqdm
from utils.utils import save_structure

def main():
    rag = RAG()
    df = pd.read_csv('./data/question_answer_pairs.txt' , delimiter='\t')

    answer = []
    for i in tqdm(range(0, len(df.Question.values))):
        question = df.Question.values[i]
        if type(question) == str:
            respones = rag(question)
            answer.append(respones)
        else:
            answer.append("")
 
        if i%100==0:
            print(f"Saving process {i}")
            save_structure(answer, "./questions_answered", f"question_answer_pairs_{i}")
        

    save_structure(answer, "./questions_answered", "question_answer_pairs_total")
    

if __name__ == "__main__":
    main()