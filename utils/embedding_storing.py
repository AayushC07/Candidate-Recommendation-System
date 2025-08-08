from sentence_transformers import SentenceTransformer
import chromadb

# Initialize ChromaDB client
client_chroma = chromadb.Client()
model = SentenceTransformer('all-MiniLM-L6-v2')

def jd_section_embeddings(jd_fields):
    jd_job_title_embedding = model.encode(jd_fields['Job Title'].replace('\n', '').strip())
    jd_responsibilities_embedding = model.encode(jd_fields['Responsibilities'].replace('\n', '').strip())
    jd_required_skills_embedding = model.encode(jd_fields['Required Skills'].replace('\n', '').strip())
    jd_educational_qualifications_embedding = model.encode(jd_fields['Educational Qualifications'].replace('\n', '').strip())
    jd_experience_level_embedding = model.encode(jd_fields['Experience Level'].replace('\n', '').strip())
    jd_leadership_experience_embedding = model.encode(jd_fields['Leadership Experience'].replace('\n', '').strip())
    jd_certifications_embedding = model.encode(jd_fields['Certifications'].replace('\n', '').strip())
    jd_extra_curriculum_embedding = model.encode(jd_fields['Extra Curriculum Activities'].replace('\n', '').strip())

    collection = client_chroma.get_or_create_collection(name="candidate_jd_embeddings")

    embeddings = [jd_job_title_embedding, jd_responsibilities_embedding, jd_required_skills_embedding,
                      jd_educational_qualifications_embedding, jd_experience_level_embedding,
                      jd_leadership_experience_embedding, jd_certifications_embedding,
                      jd_extra_curriculum_embedding]
    collection.add(
        embeddings = embeddings,
        documents = ["Job Title : " + jd_fields['Job Title'],
                    "Responsibilities : " + jd_fields['Responsibilities'],
                    "Required Skills : " + jd_fields['Required Skills'],
                    "Educational Qualifications : " + jd_fields['Educational Qualifications'],
                    "Experience Level : " + jd_fields['Experience Level'],
                    "Leadership Experience : " + jd_fields['Leadership Experience'],
                    "Certifications : " + jd_fields['Certifications'],
                    "Extra Curriculum Activities : " + jd_fields['Extra Curriculum Activities']],
        metadatas = [{"section": "Job Title", "source": "Job Description"},
                     {"section": "Responsibilities", "source": "Job Description"},
                     {"section": "Required Skills", "source": "Job Description"},
                     {"section": "Educational Qualifications", "source": "Job Description"},
                     {"section": "Experience Level", "source": "Job Description"},
                     {"section": "Leadership Experience", "source": "Job Description"},
                     {"section": "Certifications", "source": "Job Description"},
                     {"section": "Extra Curriculum Activities", "source": "Job Description"}],
        ids = ["job_title", "responsibilities", "required_skills", "educational_qualifications",
               "experience_level", "leadership_experience", "certifications", "extra_curriculum"]
    )

def resume_section_embeddings(resume_fields,i):
    resume_name_embedding = model.encode(resume_fields['Candidate Name'].replace('\n', '').strip())
    resume_email_embedding = model.encode(resume_fields['Email ID'].replace('\n', '').strip())
    resume_education_embedding = model.encode(resume_fields['Education'].replace('\n', '').strip())
    resume_experience_embedding = model.encode(resume_fields['Experience'].replace('\n', '').strip())
    resume_projects_embedding = model.encode(resume_fields['Projects'].replace('\n', '').strip())
    resume_skills_embedding = model.encode(resume_fields['Skills'].replace('\n', '').strip())
    resume_leadership_embedding = model.encode(resume_fields['Leadership'].replace('\n', '').strip())
    resume_certifications_embedding = model.encode(resume_fields['Certifications'].replace('\n', '').strip())
    resume_extra_curriculum_embedding = model.encode(resume_fields['Extra Curriculum Activities'].replace('\n', '').strip())
    resume_projexperience_embedding = model.encode(resume_fields['Projects'] + " " + resume_fields['Experience'].replace('\n', '').strip())
    resume_expskills_embedding = model.encode(resume_fields['Experience'] + " " + resume_fields['Skills'].replace('\n', '').strip())
    
    collection = client_chroma.get_or_create_collection(name="candidate_resume_embeddings")

    collection.add(
        #collection_name = "candidate_resume_embeddings",
        embeddings = [resume_name_embedding, resume_email_embedding, resume_education_embedding,
                      resume_experience_embedding, resume_projects_embedding, resume_skills_embedding,
                      resume_leadership_embedding, resume_certifications_embedding,
                      resume_extra_curriculum_embedding, resume_projexperience_embedding, resume_expskills_embedding],
        documents = ["Candidate Name : " + resume_fields['Candidate Name'],
                    "Email ID : " + resume_fields['Email ID'],
                    "Education : " + resume_fields['Education'],
                    "Experience : " + resume_fields['Experience'],
                    "Projects : " + resume_fields['Projects'],
                    "Skills : " + resume_fields['Skills'],
                    "Leadership : " + resume_fields['Leadership'],
                    "Certifications : " + resume_fields['Certifications'],
                    "Extra Curriculum Activities : " + resume_fields['Extra Curriculum Activities'],
                    "Projects and Experience Combined : " + resume_fields['Projects'] + " " + resume_fields['Experience'],
                    "Experience and Skills Combined : " + resume_fields['Experience'] + " " + resume_fields['Skills']],
        metadatas = [{"section": "Candidate Name", "source": "Resume"},
                     {"section": "Email ID", "source": "Resume"},
                     {"section": "Education", "source": "Resume"},
                     {"section": "Experience", "source": "Resume"},
                     {"section": "Projects", "source": "Resume"},
                     {"section": "Skills", "source": "Resume"},
                     {"section": "Leadership", "source": "Resume"},
                     {"section": "Certifications", "source": "Resume"},
                     {"section": "Extra Curriculum Activities", "source": "Resume"},
                     {"section": "Projects and Experience Combined", "source": "Resume"},
                     {"section": "Experience and Skills Combined", "source": "Resume"}],
        ids = [f"candidate{i}_name", f"candidate{i}_email", f"candidate{i}_education",
               f"candidate{i}_experience", f"candidate{i}_projects", f"candidate{i}_skills",
               f"candidate{i}_leadership", f"candidate{i}_certifications",
               f"candidate{i}_extra_curriculum", f"candidate{i}_experience_projects", f"candidate{i}_experience_skills"]
    )