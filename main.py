import os, re, json, fitz, docx2txt
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# ───────────────────────────────────────────────
# 1. FastAPI setup
# ───────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rezmay.co"],   # add dev origin if you test locally
    allow_methods=["*"],
    allow_headers=["*"],
)
AUTH = os.getenv("BASIC_AUTH_TOKEN", "")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───────────────────────────────────────────────
# 2. Pydantic response model
# ───────────────────────────────────────────────
class ExperienceItem(BaseModel):
    title: str
    company: str
    location: str
    years: str
    bullets: List[str] = []

class ResumeResponse(BaseModel):
    name: str
    email: str
    phone: str
    experience: List[ExperienceItem] = []
    education: List[str] = []
    skills: List[str] = []
    match_score: Optional[float] = None

# ───────────────────────────────────────────────
# 3. Helpers  (same as before – shortened for space)
# ───────────────────────────────────────────────
def extract_text_from_file(fname: str, blob: bytes) -> str: ...
def extract_name(t:str)->str: ...
def extract_email(t:str)->str: ...
def extract_phone(t:str)->str: ...
def extract_experience_sections(resume_text:str)->List[ExperienceItem]: ...
def extract_education(t:str)->List[str]: ...
def extract_skills(t:str)->List[str]: ...
def compare_with_job_description(rtxt:str, jdtxt:str)->float: ...

# ───────────────────────────────────────────────
# 4. POST /parse
# ───────────────────────────────────────────────
@app.post("/parse", response_model=ResumeResponse)
async def parse(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    # simple bearer check
    if AUTH and authorization != f"Bearer {AUTH}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    raw = await file.read()
    text = extract_text_from_file(file.filename, raw)

    resp = ResumeResponse(
        name   = extract_name(text),
        email  = extract_email(text),
        phone  = extract_phone(text),
        experience = extract_experience_sections(text),
        education  = extract_education(text),
        skills     = extract_skills(text)
    )
    if job_description:
        resp.match_score = round(compare_with_job_description(text, job_description)*100,2)

    return resp