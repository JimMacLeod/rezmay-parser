import os
import re
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import docx2txt

app = FastAPI()

# Enable CORS for your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rezmay.co"],  # Replace with your live WP domain
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


def extract_experience_sections(text: str) -> list:
    experience_entries = []
    lines = text.split('\n')

    current = None
    bullets = []

    for line in lines:
        line = line.strip()

        # Match a new role entry
        title_match = re.search(
            r'(?P<title>.*?(Manager|Director|VP|Engineer|Designer|Marketing|Developer|Lead).*?)'
            r'( at (?P<company>.*?))?'
            r'( \((?P<years>[^)]+)\))?$',
            line,
            re.IGNORECASE
        )

        # If the line looks like a bullet or continuation
        is_bullet = line.startswith(('•', '-', '*')) or (current and not title_match)

        if title_match and title_match.group('title'):
            # Save the previous entry
            if current:
                current['bullets'] = bullets
                experience_entries.append(current)
                bullets = []

            current = {
                "title": title_match.group('title').strip(),
                "company": title_match.group('company') or "",
                "years": title_match.group('years') or ""
            }

        elif is_bullet and current:
            clean_line = line.lstrip('•-* ').strip()
            if clean_line:
                bullets.append(clean_line)

    if current:
        current['bullets'] = bullets
        experience_entries.append(current)

    return experience_entries


def extract_education(text: str) -> list:
    matches = re.findall(r'(BA|BS|MA|MBA|PhD|Associate)[^,\n]*,? ?[^,\n]+', text, re.IGNORECASE)
    return [{"degree": match, "school": ""} for match in matches]


def extract_skills(text: str) -> list:
    common_skills = ["Python", "JavaScript", "Marketing", "Leadership", "Design", "Copywriting", "UX", "Analytics"]
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