import streamlit as st
import os
from utils.pdf_processing import extract_text_from_pdf
from utils.word_file_processin import extract_text_from_docx
from utils.field_extraction import extract_jd_fields, extract_resume_fields
from utils.embedding_storing import jd_section_embeddings, resume_section_embeddings
from utils.similarity_scoring import matching_score
from utils.reasoning import reasoning_function

# Setup for FAISS index files
for f in ["faiss_jd.index", "faiss_jd_metadata.pkl", "faiss_resume.index", "faiss_resume_metadata.pkl"]:
    if os.path.exists(f):
        os.remove(f)

# Streamlit app setup
st.title("Candidate Recommendation System")

uploaded_files = st.file_uploader("Upload Candidate Resumes (PDFs, DOCX, TXT)", accept_multiple_files=True)

job_description = st.text_area("Enter Job Description:")

submit_button = st.button("Compare Resumes with Job Description")

if submit_button:
    if not uploaded_files:
        st.error("Please upload at least one resume file.")
    elif not job_description.strip():
        st.error("Please enter a job description.")
    else:
        jd_fields = extract_jd_fields(job_description)
        if not jd_fields:
            st.error("Could not extract fields from job description. Please check the format.")
        else:
            st.text("Generating job description embeddings...")
            jd_section_embeddings(jd_fields)

            st.text("Processing resumes and generating embeddings...")
            score_dict = {}
            total_files = len(uploaded_files)
            i = 1

            for file in uploaded_files:
                if file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(file)
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resume_text = extract_text_from_docx(file)
                elif file.type == "text/plain":
                    resume_text = file.read().decode("utf-8")
                else:
                    st.warning(f"Unsupported file type for {file.name}, skipping.")
                    continue

                resume_fields = extract_resume_fields(resume_text)
                if not resume_fields:
                    st.warning(f"Could not extract fields from {file.name}, please check format.")
                    continue

                st.text(f"Parsing resume {i}/{total_files}...")

                resume_section_embeddings(resume_fields, i)

                reason = reasoning_function(resume_fields, jd_fields)
                score = matching_score(i)
                score_dict[resume_fields['Candidate Name']] = {"score": score, "reason": reason}

                st.text(f"Resume {i} processed with score: {score:.4f}")
                i += 1

            st.success("Processing completed!")

            sorted_candidates = sorted(score_dict.items(), key=lambda x: x[1]["score"], reverse=True)
            st.subheader("Top Candidates Based on Matching Scores:")
            for candidate, data in sorted_candidates[:5]:
                st.write(f"{candidate}: {data['score']:.4f}")
                st.write(f"Reason for eligibility: {data['reason']}")
