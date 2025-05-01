import os
import re
from fastapi import FastAPI, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import List
from pypdf import PdfReader
from docx import Document
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rezmay.co"],  # or ["*"] for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI()
AUTH = os.getenv('BASIC_AUTH_TOKEN', '')

def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(file: UploadFile) -> str:
    doc = Document(file.file)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_experience(text: str) -> List[dict]:
    experience = []
    lines = text.splitlines()
    in_experience = False
    current = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.search(r'\bexperience\b', line.lower()):
            in_experience = True
            continue

        if in_experience:
            job_match = re.match(r'^(.+?)\s+[,–-]\s+(.+?)\s+\((\d{2}/\d{2})\s*[-–]\s*(\d{2}/\d{2}|present)\)', line)
            if job_match:
                if current:
                    experience.append(current)
                current = {
                    "title": job_match.group(1).strip(),
                    "company": job_match.group(2).strip(),
                    "start_date": job_match.group(3),
                    "end_date": job_match.group(4),
                    "bullets": []
                }
                continue

            if current and line.startswith(("•", "-", "◦")):
                current["bullets"].append(line.lstrip("•-◦ ").strip())

            if re.match(r'^\s*(education|skills|projects|certifications)\b', line.lower()):
                if current:
                    experience.append(current)
                break

    if current:
        experience.append(current)

    return experience

@app.post("/parse")
async def parse(file: UploadFile, authorization: str | None = Header(None)):
    if AUTH and authorization != f"Bearer {AUTH}":
        raise HTTPException(401, detail="Invalid token")

    ext = file.filename.lower()
    if ext.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif ext.endswith(".docx"):
        text = extract_text_from_docx(file)
    else:
        raise HTTPException(400, detail="Unsupported file type")

    experience = extract_experience(text)

    return JSONResponse({
        "experience": experience
    })