import os
from huggingface_hub import InferenceClient
import pdfplumber
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Initialize Hugging Face InferenceClient
client = InferenceClient(
    provider="novita",
    api_key=os.environ["HF_TOKEN"],
)

# Initialize ChromaDB client
client_chroma = chromadb.Client()
model = SentenceTransformer('all-MiniLM-L6-v2')

# PDF --> Text extraction function
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


st.title("Candidate Recommendation Sysytem")

uploaded_files = st.file_uploader("Upload Candidate Resumes (PDFs)", accept_multiple_files=True)

job_description = st.text_area("Enter Job Description:")

submit_button = st.button("Compare Resumes with Job Description")

if uploaded_files and job_description and submit_button:

    progress = st.progress(0)
    total_files = len(uploaded_files)

    score_dict = {}
    i = 1

    # Process each uploaded resume file and extract fields for resume comparison
    for file in uploaded_files:
        if file.type == "application/pdf":
            resume_text = extract_text_from_pdf(file)
        elif file.type == "text/plain":
            resume_text = file.read().decode("utf-8")
        else:
            st.error("Unsupported file type. Please upload PDF or text files.")
            break
        
        st.text(f"Parsing resume {i}/{total_files}...")
        progress.progress(int((i / total_files) * 100))

        resume_embedding = model.encode(resume_text)
        jd_embedding = model.encode(job_description)

        score= cosine_similarity([resume_embedding], [jd_embedding])[0][0]
        score_dict[file.name] = score
        i += 1
    
    st.text("Processing completed!")
    progress.empty()
    
    # Display the matching scores for each candidate
    sort_dict = dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
    print (sort_dict)
    st.subheader("Top Candidates Based on Matching Scores:")
    for candidate_name, score in sort_dict.items():
        st.write(f"{candidate_name}: {score:.4f}")
    