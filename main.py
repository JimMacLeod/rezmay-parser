import os
import re
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import docx2txt
import openai

# OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Enable CORS for your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rezmay.co"],  # Update if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTH = os.getenv("BASIC_AUTH_TOKEN", "")


def extract_text_from_file(filename: str, content: bytes) -> str:
    ext = filename.lower().split('.')[-1]
    if ext == 'pdf':
        with open("/tmp/temp.pdf", "wb") as f:
            f.write(content)
        doc = fitz.open("/tmp/temp.pdf")
        return "\n".join(page.get_text() for page in doc)
    elif ext == 'docx':
        with open("/tmp/temp.docx", "wb") as f:
            f.write(content)
        return docx2txt.process("/tmp/temp.docx")
    else:
        raise ValueError("Unsupported file type")


def extract_name(text: str) -> str:
    lines = text.strip().split("\n")
    return lines[0].strip() if lines else ""


def extract_email(text: str) -> str:
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = re.search(r'(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', text)
    return match.group(0) if match else ""


def generate_experience_extraction_prompt(resume_text: str) -> str:
    prompt = f"""
You are a resume parser. Extract ONLY what is explicitly stated in the following resume text.
Do not guess, infer, or add any information. If a field is missing, leave it blank or omit it.
Return a list of experience entries in strict JSON format with the following structure:

[
  {{
    "title": "Job title",
    "company": "Company name",
    "location": "City, State or Remote",
    "years": "Start – End",
    "bullets": [
      "Responsibility or achievement 1",
      "Responsibility or achievement 2"
    ]
  }},
  ...
]

Here is the resume text:
\"\"\"
{resume_text}
\"\"\"
Return only the JSON, nothing else.
"""
    return prompt.strip()


def extract_experience_sections(resume_text: str) -> list:
    prompt = generate_experience_extraction_prompt(resume_text)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response['choices'][0]['message']['content'].strip()
        print("GPT raw response:\n", content)  # Logging for debug
        parsed = json.loads(content)
        return parsed if isinstance(parsed, list) else []
    except Exception as e:
        print(f"AI experience parsing failed: {e}")
        return []


def extract_education(text: str) -> list:
    lines = text.split('\n')
    education = []
    in_education = False

    degree_keywords = re.compile(r'\b(Bachelor|Master|BS|MS|MBA|PhD|Certificate|Associate)\b', re.IGNORECASE)

    for i, line in enumerate(lines):
        line = line.strip()

        # Enter education section
        if not in_education and 'education' in line.lower():
            in_education = True
            continue

        if in_education:
            if degree_keywords.search(line):
                degree = line
                school = lines[i - 1].strip() if i > 0 else ''
                if school and degree:
                    education.append({
                        "school": school,
                        "degree": degree
                    })

            # Stop parsing when hitting a likely new section
            if line == '' or re.match(r'^(experience|skills|summary)', line, re.IGNORECASE):
                break

    return education


def extract_skills(text: str) -> list:
    common_skills = [
        "Python", "JavaScript", "Marketing", "Leadership",
        "Design", "Copywriting", "UX", "Analytics"
    ]
    found = [skill for skill in common_skills if skill.lower() in text.lower()]
    return sorted(set(found))


def compare_with_job_description(resume_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0]


@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    if AUTH and authorization != f"Bearer {AUTH}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    content = await file.read()
    try:
        resume_text = extract_text_from_file(file.filename, content)
    except Exception as e:
        print(f"Error while extracting text: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    data = {
        "name": extract_name(resume_text),
        "email": extract_email(resume_text),
        "phone": extract_phone(resume_text),
        "experience": extract_experience_sections(resume_text),
        "education": extract_education(resume_text),
        "skills": extract_skills(resume_text)
    }

    if job_description:
        score = compare_with_job_description(resume_text, job_description)
        data["match_score"] = round(score * 100, 2)

    return data