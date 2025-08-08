import os
import json
from huggingface_hub import InferenceClient

# Initialize Hugging Face InferenceClient
client = InferenceClient(
    provider="novita",
    api_key=os.environ["HF_TOKEN"],
)

# Function to extract fields from a resume using Hugging Face model
def extract_resume_fields(resume_text):
    try:
        prompt_resume = """
        You are a Data Extraction Specialist with expertise in parsing and extracting structured information from resumes. Your task is to extract specific fields from the provided resume text and return them in a JSON format. Follow the instructions below carefully.

        Instructions:
        - Candidate Name: Extract the full name of the candidate.
        - Contact Information: Extract the candidate's email address.
        - Education: Extract all educational qualifications, including institutions, degrees, and years.
        - Experience: Extract all work experience, including job titles, companies, employment periods (start and end dates), and job descriptions.
        - Projects: Extract the entire section related to projects, including project names, descriptions, and technologies used.
        - Skills: Extract the list of skills mentioned (both technical and soft skills).
        - Leadership: Extract any leadership roles or experiences, such as managing teams or mentoring.
        - Certifications: Extract all certifications listed.
        - Extra-Curricular Activities: Extract any extra-curricular activities mentioned, such as volunteer work, clubs, etc.

        **Guidelines for Output:**
        - Your response should be in **JSON format** with the exact structure outlined below.
        - If a section is missing in the resume, return an empty string for that section.
        - Do not summarize or paraphrase the content; extract it exactly as it appears.
        - Return only the required fields and ensure the output is free of extra text or formatting.
        
        The output should be a JSON object with the following structure:
        {
            "Candidate Name": "Candidate Name",
            "Email ID": "Contact Information",
            "Education": "Education Section Content",
            "Experience": "Experience Section Content",
            "Projects": "Projects Section Content",
            "Skills": "Skills Section Content",
            "Leadership": "Leadership Section Content",
            "Certifications": "Certifications Section Content",
            "Extra Curriculum Activities": "Extra Curriculum Activities Section Content"
        }
        Do not include any other text, just the above mentioned output format.
        If any section is not present in the resume, return it as an empty string.
        Do not add ```json or any other formatting to the output.

        Resume Text:

        """

        prompt_with_resume = prompt_resume + resume_text.strip()  # Append the resume text to the prompt
        
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {"role": "user", "content": prompt_with_resume}
            ]
        )
        # Extract and return the generated text from the model
        text1 =  completion.choices[0].message["content"].strip()
        try:
            resume_dict = json.loads(text1)
        except json.JSONDecodeError:
            resume_dict = {}
        return resume_dict
    except Exception as e:
        print(f"Error extracting resume fields: {e}")
        # If there's an error, return an empty dictionary
        return {}
        
    
# Calling Hugging Face model for job description field extraction
def extract_jd_fields(jd):
    prompt_jd = """
    Given the following job description, please extract the following sections:
    - Job Title: Full title of the job position.
    - Responsibilities: Complete list of responsibilities or duties or qualities mentioned for the job role.
    - Required Skills: A list of skills (technical or soft skills) or qualities required for the job.
    - Educational Qualifications: Educational qualifications or degrees required for the position. (Completed degrees like Bachelors, Masters, B.Tech, M.Tech, MBA, B.E., M.E., HR, etc.)
    - Experience Level: The level of experience and skills required for the job (e.g., entry-level, mid-level, senior).
    - Leadership Experience: Leadership responsibilities or experiences expected from the candidate.
    - Certifications: Certifications or specialized training required or preferred for the role.
    - Extra Curriculum Activities: Any soft skills or behavioral expectations listed in the job description (e.g., “Strong communication skills,” “Proven ability to collaborate effectively with cross-functional teams,” “Demonstrates initiative and self-motivation,” “Problem-solving capabilities”)
    The output should be a JSON object with the following structure:
    {
        "Job Title": "Job Title",
        "Responsibilities": "Responsibilities Section Content",
        "Required Skills": "Skills / Qualitites in the candidate Section Content",
        "Educational Qualifications": "Educational Qualifications Section Content",
        "Experience Level": "Experience Level Section Content",
        "Leadership Experience": "Leadership Experience Section Content",
        "Certifications": "Certifications Section Content",
        "Extra Curriculum Activities": "Extra Curriculum Activities Section Content"
    }
    Do not include any other text, just the above mentioned output format.
    If any section is not present in the job description, return it as an empty string.
    Do not add ```json or any other formatting to the output.
    Job Description Text:

    """
    prompt_with_jd = prompt_jd + jd.strip()  # Append the job description text to the prompt
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "user", "content": prompt_with_jd}
        ]
    )
    # Extract and return the generated text from the model
    text2 = completion.choices[0].message["content"].strip()
    try:
        jd_dict = json.loads(text2)
    except json.JSONDecodeError:
        jd_dict = {}

    return jd_dict