import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

# Files to store FAISS indexes and metadata for JD and resumes separately
JD_INDEX_FILE = "faiss_jd.index"
JD_METADATA_FILE = "faiss_jd_metadata.pkl"

RESUME_INDEX_FILE = "faiss_resume.index"
RESUME_METADATA_FILE = "faiss_resume_metadata.pkl"

def load_faiss(index_file, metadata_file, embedding_dim):
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        index = faiss.read_index(index_file)
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    else:
        index = faiss.IndexFlatL2(embedding_dim)
        return index, []

def save_faiss(index, metadata, index_file, metadata_file):
    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

def jd_section_embeddings(jd_fields):
    embedding_dim = model.get_sentence_embedding_dimension()
    index, metadata = load_faiss(JD_INDEX_FILE, JD_METADATA_FILE, embedding_dim)

    texts = [
        jd_fields['Job Title'].replace('\n', '').strip(),
        jd_fields['Responsibilities'].replace('\n', '').strip(),
        jd_fields['Required Skills'].replace('\n', '').strip(),
        jd_fields['Educational Qualifications'].replace('\n', '').strip(),
        jd_fields['Experience Level'].replace('\n', '').strip(),
        jd_fields['Leadership Experience'].replace('\n', '').strip(),
        jd_fields['Certifications'].replace('\n', '').strip(),
        jd_fields['Extra Curriculum Activities'].replace('\n', '').strip(),
    ]

    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")

    index.add(embeddings)
    metadata.extend([
        "job_title", "responsibilities", "required_skills", "educational_qualifications",
        "experience_level", "leadership_experience", "certifications", "extra_curriculum"
    ])

    save_faiss(index, metadata, JD_INDEX_FILE, JD_METADATA_FILE)
    print("Job Description embeddings saved.")

def resume_section_embeddings(resume_fields, i):
    embedding_dim = model.get_sentence_embedding_dimension()
    index, metadata = load_faiss(RESUME_INDEX_FILE, RESUME_METADATA_FILE, embedding_dim)

    texts = [
        resume_fields['Experience'].replace('\n', '').strip(),
        (resume_fields['Projects'] + " " + resume_fields['Experience']).replace('\n', '').strip(),
        (resume_fields['Experience'] + " " + resume_fields['Skills']).replace('\n', '').strip(),
        resume_fields['Education'].replace('\n', '').strip(),
        resume_fields['Experience'].replace('\n', '').strip(),
        resume_fields['Leadership'].replace('\n', '').strip(),
        resume_fields['Certifications'].replace('\n', '').strip(),
        resume_fields['Extra Curriculum Activities'].replace('\n', '').strip(),
    ]

    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")

    ids = [
        f"candidate{i}_experience",
        f"candidate{i}_experience_projects",
        f"candidate{i}_experience_skills",
        f"candidate{i}_education",
        f"candidate{i}_experience",
        f"candidate{i}_leadership",
        f"candidate{i}_certifications",
        f"candidate{i}_extra_curriculum"
    ]

    index.add(embeddings)
    metadata.extend(ids)

    save_faiss(index, metadata, RESUME_INDEX_FILE, RESUME_METADATA_FILE)
    print(f"Resume embeddings for candidate {i} saved.")
