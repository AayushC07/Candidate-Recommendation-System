import chromadb
from sklearn.metrics.pairwise import cosine_similarity

def matching_score(collection_resume, collection_jd, i):
    relation1_score = cosine_similarity(
        [collection_resume.get(ids=[f"candidate{i}_experience"], include=['embeddings'])['embeddings'][0]],
        [collection_jd.get(ids=["job_title"], include=['embeddings'])['embeddings'][0]]
    )[0][0].item()
    relation2_score = cosine_similarity(
        [collection_resume.get(ids=[f"candidate{i}_experience_projects"], include=['embeddings'])['embeddings'][0]],
        [collection_jd.get(ids=["responsibilities"], include=['embeddings'])['embeddings'][0]]
    )[0][0].item()
    relation3_score = cosine_similarity(
        [collection_resume.get(ids=[f"candidate{i}_experience_skills"], include=['embeddings'])['embeddings'][0]],
        [collection_jd.get(ids=["required_skills"], include=['embeddings'])['embeddings'][0]]
    )[0][0].item()
    relation4_score = cosine_similarity(
        [collection_resume.get(ids=[f"candidate{i}_education"], include=['embeddings'])['embeddings'][0]],
        [collection_jd.get(ids=["educational_qualifications"], include=['embeddings'])['embeddings'][0]]
    )[0][0].item()
    relation5_score = cosine_similarity(
        [collection_resume.get(ids=[f"candidate{i}_experience"], include=['embeddings'])['embeddings'][0]],
        [collection_jd.get(ids=["experience_level"], include=['embeddings'])['embeddings'][0]]
    )[0][0].item()
    relation6_score = cosine_similarity(
        [collection_resume.get(ids=[f"candidate{i}_leadership"], include=['embeddings'])['embeddings'][0]],
        [collection_jd.get(ids=["leadership_experience"], include=['embeddings'])['embeddings'][0]]
    )[0][0].item()
    relation7_score = cosine_similarity(
        [collection_resume.get(ids=[f"candidate{i}_certifications"], include=['embeddings'])['embeddings'][0]],
        [collection_jd.get(ids=["certifications"], include=['embeddings'])['embeddings'][0]]
    )[0][0].item()
    relation8_score = cosine_similarity(
        [collection_resume.get(ids=[f"candidate{i}_extra_curriculum"], include=['embeddings'])['embeddings'][0]],
        [collection_jd.get(ids=["extra_curriculum"], include=['embeddings'])['embeddings'][0]]
    )[0][0].item()
    print(relation1_score, relation2_score, relation3_score, relation4_score,
          relation5_score, relation6_score, relation7_score, relation8_score)
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
    for key, value in section_weights.items():
        final_score += locals()[key + "_score"] * value

    return final_score  # Return the matching score for relation1 only for simplicity
