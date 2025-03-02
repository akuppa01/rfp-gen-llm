import os
import PyPDF2

def extract_text_from_pdfs(pdf_dir):
    texts = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            with open(os.path.join(pdf_dir, filename), 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                texts.append(text)
    return texts
