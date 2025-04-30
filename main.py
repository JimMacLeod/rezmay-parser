# main.py  –  FastAPI résumé parser for Rezmay
import os, re, json, textwrap
from typing import List, Optional, Union

import fitz            # PyMuPDF
import docx2txt
import openai

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ────────────────────────────────
# ENV / Auth
# ────────────────────────────────
client   = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
AUTH_KEY = os.getenv("BASIC_AUTH_TOKEN", "")

# ────────────────────────────────
# FastAPI app
# ────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rezmay.co"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────
# Pydantic response models
# ────────────────────────────────
class ExperienceItem(BaseModel):
    title: str = ""
    company: str = ""
    location: str = ""
    years: str = ""
    bullets: List[str] = []

class ResumeResponse(BaseModel):
    name:   str = ""
    email:  str = ""
    phone:  str = ""
    experience: List[ExperienceItem] = []
    education: List[str] = []
    skills:    List[str] = []
    match_score: Optional[float] = None

    # If GPT accidentally returns a dict for experience, coerce it into list
    @field_validator("experience", mode="before")
    @classmethod
    def _exp_must_be_list(cls, v):
        if isinstance(v, dict):
            return [v]
        return v or []

# ────────────────────────────────
# Helper: extract raw text
# ────────────────────────────────
def extract_text_from_file(fname: str, content: bytes) -> str:
    ext = fname.lower().split(".")[-1]
    if ext == "pdf":
        tmp = "/tmp/doc.pdf"; open(tmp, "wb").write(content)
        return "\n".join(page.get_text() for page in fitz.open(tmp))
    if ext == "docx":
        tmp = "/tmp/doc.docx"; open(tmp, "wb").write(content)
        return docx2txt.process(tmp)
    raise ValueError("Only PDF or DOCX files are supported")

# ────────────────────────────────
# Simple regex extractors
# ────────────────────────────────
def extract_email(txt: str):
    m = re.search(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}", txt); return m[0] if m else ""

def extract_phone(txt: str):
    m = re.search(r"(\+?\d{1,2}\s*)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", txt)
    return m[0] if m else ""

def extract_name(txt: str):
    # crudest possible – first non-blank line that isn’t email/phone
    for line in txt.splitlines():
        line=line.strip()
        if not line or "@" in line or re.search(r"\d", line): continue
        return line
    return ""

# ────────────────────────────────
# GPT-assisted experience parser
# ────────────────────────────────
GPT_PROMPT = textwrap.dedent("""
You are a resume parser. Return ONLY JSON in the form:

[
  {{
    "title": "...",
    "company": "...",
    "location": "...",
    "years": "Start – End",
    "bullets": ["...", "..."]
  }},
  ...
]

Do NOT guess. If something is missing, leave it blank. Here is the text:
""").strip()

def gpt_extract_experience(chunk: str) -> List[dict]:
    msg = [{"role":"user","content": f"{GPT_PROMPT}\n\n{chunk}\n\nJSON:"}]
    try:
        raw = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=msg).choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        print("⚠️ GPT chunk failed:", e)
        return []

def extract_experience_sections(full_text: str) -> List[dict]:
    # Chunk by ~700 tokens (~2800 chars) to stay well under GPT-3.5 limit
    chunks = textwrap.wrap(full_text, 2800, break_long_words=False, break_on_hyphens=False)
    exp: List[dict] = []
    for c in chunks:
        exp.extend(gpt_extract_experience(c))
    # Deduplicate by (title, company, years)
    seen = set()
    uniq = []
    for item in exp:
        key = (item.get("title",""), item.get("company",""), item.get("years",""))
        if key not in seen and any(key):
            seen.add(key); uniq.append(item)
    return uniq

# ────────────────────────────────
# Very light education parser
# ────────────────────────────────
DEGREE_RE = re.compile(r"\b(BS|MS|MBA|B\.?A\.?|M\.?A\.?|PhD|Bachelor|Master|Associate)\b", re.I)

def extract_education(txt: str) -> List[str]:
    edu = []
    for line in txt.splitlines():
        if DEGREE_RE.search(line):
            edu.append(line.strip())
    return edu

# ────────────────────────────────
# Skills (toy example)
# ────────────────────────────────
COMMON_SKILLS = ["Python","JavaScript","Marketing","Design","Leadership",
                 "Content","UX","SEO","Analytics","Copywriting"]

def extract_skills(txt:str)->List[str]:
    return sorted({s for s in COMMON_SKILLS if s.lower() in txt.lower()})

# ────────────────────────────────
# JD similarity
# ────────────────────────────────
def compare_with_jd(resume:str,jd:str)->float:
    vec=TfidfVectorizer(stop_words='english').fit_transform([resume,jd])
    return cosine_similarity(vec[0:1],vec[1:2])[0][0]

# ────────────────────────────────
# FastAPI route
# ────────────────────────────────
@app.post("/parse", response_model=ResumeResponse)
async def parse(
    file:UploadFile=File(...),
    job_description:Optional[str]=Form(None),
    authorization:Optional[str]=Header(None)
):
    # Basic bearer auth (optional)
    if AUTH_KEY and authorization!=f"Bearer {AUTH_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    raw = await file.read()
    try:
        txt = extract_text_from_file(file.filename, raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    data = {
        "name": extract_name(txt),
        "email": extract_email(txt),
        "phone": extract_phone(txt),
        "experience": extract_experience_sections(txt),
        "education": extract_education(txt),
        "skills": extract_skills(txt)
    }
    if job_description:
        data["match_score"] = round(compare_with_jd(txt, job_description)*100, 2)

    return data