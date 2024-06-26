import utils.load_files as  files
from semantic_chunkers.chunkers.statistical import StatisticalChunker
from encoder.embedding import Embeddings
from utils.utils import save_structure


def main():
    path = "../test2/extracted_files/text_data"
    
    model = "all-MiniLM-L6-v2"
    text = files.read_folder_file(path)
    encoder = Embeddings(type = model)
  
   
    
    chunker = StatisticalChunker(encoder=encoder)
    chunks = chunker(docs=text)
    chunker.print(chunks[0])
    save_structure(chunks, "./chunks_extracted")
    

if __name__ == "__main__":
    main()