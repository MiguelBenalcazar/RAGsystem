import pymupdf 
import io
from PIL import Image
import tabula
import os
import codecs

def read_folder_file(path_extracted_data:str)->list[str]:
    files = os.listdir(path_extracted_data)
    data_load_files = []
    for file in files:
        file_path = os.path.join(path_extracted_data, file)
        content = read_file(file_path)
        data_load_files.append(content)
    return data_load_files

def read_file(file_path: str)-> str:
    with open(file_path, 'r', encoding='latin-1') as f:
        content = f.read()
    return content

def extract_text_and_images_from_pdf(pdf_path:str, output_image_folder:str)-> (str, int):
    document = pymupdf.open(pdf_path)
    text = ""
    image_count = 0

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        
        # Extract text
        text += page.get_text()

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            image.save(f"{output_image_folder}/page_{page_num + 1}_img_{img_index + 1}.{image_ext}")
            image_count += 1

    return text, image_count

def extract_tables_from_pdf(pdf_path:str, output_csv_path:str):
    tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
    tabula.convert_into(pdf_path, output_csv_path, output_format="csv", pages="all")
    return tables



