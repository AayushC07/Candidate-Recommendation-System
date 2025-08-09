from utils.pdf_processing import extract_text_from_pdf
from utils.word_file_processin import extract_text_from_docx
from utils.field_extraction import extract_jd_fields
from utils.field_extraction import extract_resume_fields
from utils.embedding_storing import jd_section_embeddings, resume_section_embeddings, client_chroma, model
from utils.reasoning import reasoning_function
from utils.similarity_scoring import matching_score
import streamlit as st


# Streamlit setup for the UI
st.title("Candidate Recommendation System")

uploaded_files = st.file_uploader("Upload Candidate Resumes (PDFs)", accept_multiple_files=True)

job_description = st.text_area("Enter Job Description:")

submit_button = st.button("Compare Resumes with Job Description")

if uploaded_files and job_description and submit_button:

    progress = st.progress(0)
    total_files = len(uploaded_files)

    jd_fields = extract_jd_fields(job_description)  # Extract fields from the job description
    st.text("Extracting Job Description fields...")

    if not jd_fields:
        st.error("Could not extract fields from the job description. Please check the format.")
    else:
        jd_section_embeddings(jd_fields)    # Generate embeddings for the job description fields
        
    score_dict = {}
    i = 1

    # Process each uploaded resume file and extract fields for resume comparison
    for file in uploaded_files:
        if file.type == "application/pdf":
            resume_text = extract_text_from_pdf(file)
        elif file.type == "text/plain":
            resume_text = file.read().decode("utf-8")
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(file)
        else:
            st.error("Unsupported file type. Please upload PDF or text files.")
            break

        resume_fields = extract_resume_fields(resume_text)  # Extract fields from the resume text
        
        if not resume_fields:
            st.error(f"Could not extract fields from {file.name}. Please check the resume format.")
            continue
        
        st.text(f"Parsing resume {i}/{total_files}...")
        
        resume_section_embeddings(resume_fields,i)  # Generate embeddings for the resume fields
        progress.progress(int((i / total_files) * 100))

        collection_resume = client_chroma.get_collection(name="candidate_resume_embeddings")
        collection_jd = client_chroma.get_collection(name="candidate_jd_embeddings")

        reason = reasoning_function(resume_fields, jd_fields)  # Generate reasoning for the match

        score= matching_score(collection_resume, collection_jd, i)  # Calculate the matching score between the resume fields and job description fields
        score_dict[resume_fields['Candidate Name']] = {"score": score, "reason": reason}

        st.text(f"Resume {i} processed with score: {score:.4f}")
        i += 1
    
    st.text("Processing completed!")
    progress.empty()

    # Display the matching scores for each candidate
    sort_dict = dict(sorted(score_dict.items(), key=lambda item: item[1]["score"], reverse=True))
    st.subheader("Top Candidates Based on Matching Scores:")
    top_5_candidates = list(sort_dict.items())[:5]
    for candidate_name, data in top_5_candidates:
        score = data["score"]
        reason = data["reason"]
        st.write(f"{candidate_name}: {score:.4f}")
        st.write(f"Reason for eligibility: {reason}")