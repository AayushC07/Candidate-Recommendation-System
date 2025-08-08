# Candidate Recommendation System
## Overview
The Candidate Recommendation System is a web-based application built using Streamlit to recommend top candidates for a job description by comparing resumes (PDF, DOC, DOCX, or text files) with job requirements. The system utilizes the LLama3 API model for field extraction from resumes and job descriptions, Sentence Transformers for generating embeddings, and ChromaDB for storing and comparing those embeddings.

## Demonstration Video
[Click here to watch the demonstartion](https://www.youtube.com/watch?v=DWFs6aqknqw)

## Features and Functionalities
### Resume Upload and Parsing:

The application allows users to upload multiple resume files in PDF, DOC, DOCX, and Text formats.

Each resume is parsed to extract key fields such as Candidate Name, Email, Experience, Education, Skills, etc.

### Job Description Input:

Users can input the job description for which they want to find the best-fit candidates.

The job description is processed to extract essential fields, including Job Title, Responsibilities, Required Skills, Experience Level, etc.

### Field Extraction Using LLama3 API Model:

The system uses the LLama3 API model (via Hugging Face's InferenceClient) to extract specific fields from the uploaded resumes and job descriptions. This extraction includes sections like Experience, Education, Skills, etc., and returns them in a structured JSON format.

### Embedding Generation and Matching:

Sentence Transformers are used to generate embeddings for both resume and job description fields.

These embeddings are stored in ChromaDB collections for efficient similarity comparison.

Cosine similarity is computed between resume embeddings and job description embeddings to evaluate how well a candidate fits the job description.

### Candidate Scoring:

The application calculates a matching score based on various factors, including:

Job Title ↔ Experience

Responsibilities ↔ Projects + Experience

Required Skills ↔ Skills + Experience

Educational Qualifications ↔ Education

Experience Level ↔ Experience

Leadership Experience ↔ Leadership

Certifications ↔ Certifications

Extra Curriculum Activities ↔ Extra Curriculum Activities

### Ranking of Candidates:

After calculating the matching scores, the candidates are ranked based on their fit for the job.

The top candidates are displayed with their scores and reasons for eligibility.

### Additional Features (Extras Folder):

In the extras/ folder, we have an alternative approach for matching resumes with job descriptions, where we attempt to match the entire resume text with the job description text.

However, this approach yields very low results due to the complexity and variety in resume formats. Therefore, we’ve moved to key-specific matching for better results, focusing on specific fields like Experience, Skills, Education, etc.

Folder Structure
├── data
│   └── job_descriptions.txt
├── extras
│   └── overall_matching.py
├── main.py
├── requirements.txt
└── utils
    ├── __pycache__
    │   ├── embedding_storing.cpython-313.pyc
    │   ├── field_extraction.cpython-313.pyc
    │   ├── pdf_processing.cpython-313.pyc
    │   ├── reasoning.cpython-313.pyc
    │   └── similarity_scoring.cpython-313.pyc
    ├── embedding_storing.py
    ├── field_extraction.py
    ├── pdf_processing.py
    ├── reasoning.py
    └── similarity_scoring.py
bash
Copy
/Candidate Recommendation System
├── app/
│   └── main.py                  # Streamlit app for UI
├── utils/
│   ├── pdf_processing.py        # Functions to extract text from PDFs
│   ├── field_extraction.py      # Functions to extract fields from resumes and job descriptions
│   ├── embedding_processing.py  # Functions to generate embeddings for resumes and job descriptions
│   ├── matching.py              # Functions to compute matching scores
│   └── reasoning.py             # Functions to generate reasons for candidate fit
├── extras/                      # Code for overall matching (low results)
│   └── overall_matching.py      # Matches entire resume text with job description text
└── requirements.txt             # Python dependencies
Setup and Installation
Clone the Repository:

bash
Copy
git clone <your-repository-url>
cd <your-project-directory>
Install Dependencies:
Create a virtual environment and install the required dependencies.

bash
Copy
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
Set Environment Variables:
Make sure to set the Hugging Face API key for the LLama3 API model inference:

bash
Copy
export HF_TOKEN=<your-hugging-face-token>
Run the Streamlit App:
Start the Streamlit application:

bash
Copy
streamlit run app/main.py
Your app will be available at http://localhost:8501.

Explanation of the Code
Main Logic:
Resume and Job Description Extraction:

The system extracts fields from resumes and job descriptions using the LLama3 API model from Hugging Face.

The text from the uploaded resume is parsed using pdfplumber (for PDFs), python-docx (for DOCX), and pywin32 (for DOC files).

Embedding Generation:

After extracting key sections from the resumes and job descriptions, Sentence Transformers are used to generate embeddings for each section.

The embeddings are stored in ChromaDB, which allows fast retrieval and similarity comparison.

Matching and Ranking:

Cosine similarity is computed between the resume embeddings and the job description embeddings to generate a matching score.

The results are ranked based on how well the candidate's resume matches the job description.

Extras Folder:
Overall Matching Approach:

The extras/overall_matching.py file contains a code that matches the entire resume text with the job description text. This method does not work well in practice and provides low results due to the mismatch in resume formats and the diversity of job descriptions. Hence, the key-specific matching approach (based on field extraction) is preferred and used in the main application.

Future Improvements
Support for .doc Files: The current implementation only supports .docx files for resume extraction. Future improvements can include adding support for older .doc files.

Advanced Matching Algorithms: Incorporate more sophisticated matching algorithms or machine learning models to improve the accuracy of matching candidates with job descriptions.

License
This project is licensed under the MIT License - see the LICENSE file for details.

