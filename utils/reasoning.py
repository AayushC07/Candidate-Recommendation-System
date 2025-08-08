import os
from huggingface_hub import InferenceClient

# Initialize Hugging Face InferenceClient
client = InferenceClient(
    provider="novita",
    api_key=os.environ["HF_TOKEN"],
)

def reasoning_function(resume_fields, jd_fields):
    prompt = f"""
    You are given below the extracted fields in dictionary format from a candidate's resume and a job description. Your task is to evaluate how well the candidate's qualifications match the job requirements based on the provided fields.
    Here is also the relations that are mapped between the resume and job description fields:
    Relation 1 : Job Title ↔ Experience
    Relation 2 : Responsibilities ↔ Projects + Experience
    Relation 3 : Required Skills ↔ Skills + Experience
    Relation 4 : Educational Qualifications ↔ Education
    Relation 5 : Experience Level ↔ Experience
    Relation 6 : Leadership Experience ↔ Leadership
    Relation 7 : Certifications ↔ Certifications
    Relation 8 : Extra Curriculum Activities ↔ Extra Curriculum Activities
    The scoring is based on the following criteria:
    Relation 1 : 0.05,
    Relation 2 : 0.30,
    Relation 3 : 0.35,
    Relation 4 : 0.10,
    Relation 5 : 0.10,
    Relation 6 : 0.05,
    Relation 7 : 0.035,
    Relation 8 : 0.015
    Give me reason as to why the candidate is a good fit for the job based on the relations and the scoring criteria. I dont want the score but I just want the reason that highlights why the candidate is a good fit for the job.
    Resume Fields:
    {resume_fields}
    Job Description Fields:
    {jd_fields}
    """
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # Extract and return the generated text from the model
    reasoning_text = completion.choices[0].message["content"].strip()
    return reasoning_text