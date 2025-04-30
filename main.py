import os, re, json, textwrap, tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz                      # PyMuPDF
import docx2txt
import openai

# ── OpenAI setup ────────────────────────────────────────────────────────────────
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── FastAPI setup ──────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rezmay.co"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTH = os.getenv("BASIC_AUTH_TOKEN", "")

# ── Pydantic response model (adds automatic 422 validation) ────────────────────
class ResumeResponse(BaseModel):
    name:   str
    email:  str
    phone:  str
    experience: list
    education: list
    skills: list
    match_score: Optional[float] = None

# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_text_from_file(filename: str, data: bytes) -> str:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(data)
            doc = fitz.open(f.name)
            return "\n".join(p.get_text() for p in doc)
    elif ext == "docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
            f.write(data)
            return docx2txt.process(f.name)
    raise ValueError("Unsupported file type")

def extract_name(txt: str)  -> str:
    return txt.strip().split("\n",1)[0]

def extract_email(txt: str) -> str:
    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w{2,}", txt);   return m.group(0) if m else ""

def extract_phone(txt: str) -> str:
    m = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", txt)
    return m.group(0) if m else ""

# ── AI experience / education (chunked) ────────────────────────────────────────
PROMPT = """
You are a resume parser. Extract ONLY what is explicitly present.  
Return JSON **only** (no markdown) in the format:
{{
  "experience":[{{"title":"","company":"","location":"","years":"","bullets":[]}},…],
  "education":[{{"school":"","degree_type":"","field":""}},…]
}}

Resume text:
"""

def gpt_chunk_parse(resume_text: str, chunk_tokens: int = 1800) -> dict:
    exp, edu = [], []

    # naive chunk by characters (~4 chars per token for most English text)
    for part in textwrap.wrap(resume_text, chunk_tokens*4):
        prompt = PROMPT.format(chunk=part)
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}],
                temperature=0
            )
            parsed = json.loads(resp.choices[0].message.content)
            exp.extend(parsed.get("experience", []))
            edu.extend(parsed.get("education",  []))
        except Exception as e:
            print("⚠️ GPT chunk failed:", e)

    return {"experience": exp, "education": edu}

# ── Regex education fallback ───────────────────────────────────────────────────
_deg = re.compile(r"\b(Bachelor|Master|BS|MS|MBA|PhD|Certificate|Associate)\b", re.I)

def extract_education(txt: str) -> List[dict]:
    lines = [l.strip() for l in txt.splitlines()]
    edu, grab = [], False
    for i,l in enumerate(lines):
        if not grab and "education" in l.lower(): grab=True; continue
        if grab and l and _deg.search(l):
            school = lines[i-1] if i>0 else ""
            parts  = [p.strip() for p in l.split(",",1)]
            edu.append({"school":school,
                        "degree_type":parts[0],
                        "field":parts[1] if len(parts)>1 else ""})
        if grab and l.lower().startswith(("experience","skills","summary")): break
    return edu

# ── Simple skills extractor (keyword list) ─────────────────────────────────────
def extract_skills(txt:str)->list:
    keywords = ["Python","JavaScript","Marketing","Leadership",
                "Design","Copywriting","UX","Analytics","SEO","Content"]
    return sorted(set(k for k in keywords if k.lower() in txt.lower()))

# ── JD match score ─────────────────────────────────────────────────────────────
def jd_similarity(resume:str,jd:str)->float:
    v = TfidfVectorizer(stop_words="english").fit_transform([resume,jd])
    return cosine_similarity(v[0:1],v[1:2])[0][0]

# ── API Endpoint ───────────────────────────────────────────────────────────────
@app.post("/parse", response_model=ResumeResponse)
async def parse(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    if AUTH and authorization != f"Bearer {AUTH}":
        raise HTTPException(401, "Unauthorized")

    text = extract_text_from_file(file.filename, await file.read())

    gpt_data = gpt_chunk_parse(text)
    experience = gpt_data["experience"]
    education  = gpt_data["education"] or extract_education(text)   # fallback

    data = {
        "name": extract_name(text),
        "email":extract_email(text),
        "phone":extract_phone(text),
        "experience":experience,
        "education":education,
        "skills":extract_skills(text)
    }
    if job_description:
        data["match_score"] = round(jd_similarity(text, job_description)*100,2)

    return data