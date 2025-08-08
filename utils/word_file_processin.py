import docx
import os

# Function to extract text from .docx files
def extract_text_from_docx(docx_file):
    """
    Extracts text from a .docx file.
    """
    doc = docx.Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text