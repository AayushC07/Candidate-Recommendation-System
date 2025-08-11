import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Files for FAISS
JD_INDEX_FILE = "faiss_jd.index"
JD_METADATA_FILE = "faiss_jd_metadata.pkl"

RESUME_INDEX_FILE = "faiss_resume.index"
RESUME_METADATA_FILE = "faiss_resume_metadata.pkl"

def load_faiss(index_file, metadata_file, embedding_dim):
    index = faiss.read_index(index_file)
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def get_vector_by_id(doc_id, index, metadata):
    try:
        idx = metadata.index(doc_id)
        vec = index.reconstruct(idx)
        return vec
    except ValueError:
        return None

def matching_score(i):
    embedding_dim = 384  # all-MiniLM-L6-v2 dimension

    resume_index, resume_metadata = load_faiss(RESUME_INDEX_FILE, RESUME_METADATA_FILE, embedding_dim)
    jd_index, jd_metadata = load_faiss(JD_INDEX_FILE, JD_METADATA_FILE, embedding_dim)

    # IDs to compare, matching original logic
    pairs = [
        (f"candidate{i}_experience", "job_title"),
        (f"candidate{i}_experience_projects", "responsibilities"),
        (f"candidate{i}_experience_skills", "required_skills"),
        (f"candidate{i}_education", "educational_qualifications"),
        (f"candidate{i}_experience", "experience_level"),
        (f"candidate{i}_leadership", "leadership_experience"),
        (f"candidate{i}_certifications", "certifications"),
        (f"candidate{i}_extra_curriculum", "extra_curriculum"),
    ]

    scores = []
    for resume_id, jd_id in pairs:
        r_vec = get_vector_by_id(resume_id, resume_index, resume_metadata)
        j_vec = get_vector_by_id(jd_id, jd_index, jd_metadata)
        if r_vec is not None and j_vec is not None:
            sim = cosine_similarity([r_vec], [j_vec])[0][0]
        else:
            sim = 0.0
        scores.append(sim)

    section_weights = {
        "relation1": 0.05,
        "relation2": 0.30,
        "relation3": 0.35,
        "relation4": 0.10,
        "relation5": 0.10,
        "relation6": 0.05,
        "relation7": 0.035,
        "relation8": 0.015
    }

    final_score = 0.0
    for idx, weight in enumerate(section_weights.values(), start=1):
        final_score += scores[idx - 1] * weight

    print(*scores)
    return final_score
