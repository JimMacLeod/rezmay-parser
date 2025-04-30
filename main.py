import os
import re
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import fitz  # PyMuPDF
import docx2txt

app = FastAPI()

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
    current = {}
    bullets = []

    for line in lines:
        line = line.strip()

        if re.match(r'.*\b(Manager|Director|VP|Engineer|Designer|Marketing|Developer|Lead)\b.*', line, re.IGNORECASE):
            if current:
                current['bullets'] = bullets
                experience_entries.append(current)
                bullets = []

            current = {
                "title": line,
                "company": "",  # optional refinement
                "years": "",    # optional refinement
            }

        elif line.startswith(('•', '-', '*')):
            bullets.append(line.lstrip('•-* ').strip())

    if current:
        current['bullets'] = bullets
        experience_entries.append(current)

    return experience_entries


def extract_education(text: str) -> list:
    matches = re.findall(r'(BA|BS|MA|MBA|PhD|Associate)[^,\n]*,? ?[^,\n]+', text, re.IGNORECASE)
    return [{"degree": match, "school": ""} for match in matches]


def extract_skills(text: str) -> list:
    # Just a basic skill matcher for now
    common_skills = ["Python", "JavaScript", "Marketing", "Leadership", "Design", "Copywriting", "UX", "Analytics"]
    found = [skill for skill in common_skills if skill.lower() in text.lower()]
    return sorted(set(found))


@app.post("/parse")
async def parse(file: UploadFile = File(...), authorization: str | None = Header(None)):
    if AUTH and authorization != f"Bearer {AUTH}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    content = await file.read()
    try:
        text = extract_text_from_file(file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "experience": extract_experience_sections(text),
        "education": extract_education(text),
        "skills": extract_skills(text)
    }