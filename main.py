import os, re, json, textwrap, tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz              # PyMuPDF
import docx2txt
import openai

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rezmay.co"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
AUTH = os.getenv("BASIC_AUTH_TOKEN", "")

class ResumeResponse(BaseModel):
    name:str; email:str; phone:str
    experience:list; education:list; skills:list
    match_score: Optional[float] = None

# ─── file → raw text ───────────────────────────────────────────────────────────
def extract_text_from_file(fname:str, data:bytes)->str:
    ext = fname.lower().split('.')[-1]
    if ext=="pdf":
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as f:
            f.write(data); doc=fitz.open(f.name); return "\n".join(p.get_text() for p in doc)
    if ext=="docx":
        with tempfile.NamedTemporaryFile(delete=False,suffix=".docx") as f:
            f.write(data); return docx2txt.process(f.name)
    raise ValueError("Unsupported file type")

def extract_name(t:str)->str:  return t.split("\n",1)[0].strip()
def extract_email(t:str)->str: m=re.search(r'[\w\.-]+@[\w\.-]+\.\w{2,}',t); return m.group(0) if m else ""
def extract_phone(t:str)->str: m=re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',t); return m.group(0) if m else ""

# ─── GPT chunk parser ──────────────────────────────────────────────────────────
PROMPT = """
You are a resume parser. Extract ONLY explicit info; do not add or guess.
Return pure JSON in this shape:
{{
 "experience":[{{"title":"","company":"","location":"","years":"","bullets":[]}} …],
 "education":[{{"school":"","degree_type":"","field":""}} …]
}}
Resume text:
"""

def gpt_chunk_parse(resume_text:str, chunk_tokens:int=1800)->dict:
    exp, edu = [], []
    for chunk in textwrap.wrap(resume_text, chunk_tokens*4):
        try:
            rsp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":PROMPT.format(chunk=chunk)}],
                temperature=0
            ).choices[0].message.content.strip()
            parsed=json.loads(rsp)
            # ★ fix: verify keys exist & are list before extending
            exp.extend(parsed.get("experience",[]) if isinstance(parsed.get("experience"),list) else [])
            edu.extend(parsed.get("education", []) if isinstance(parsed.get("education"),list) else [])
        except Exception as e:                 # JSON error or API error
            print("⚠️  Skipped chunk:", e)
            continue
    return {"experience":exp, "education":edu}

# ─── fallback regex education ─────────────────────────────────────────────────
_deg=re.compile(r'\b(Bachelor|Master|BS|MS|MBA|PhD|Certificate|Associate)\b',re.I)
def extract_education(t:str)->List[dict]:
    lines=[l.strip() for l in t.splitlines()]
    edu,grab=[],False
    for i,l in enumerate(lines):
        if not grab and "education" in l.lower(): grab=True; continue
        if grab and _deg.search(l):
            school=lines[i-1] if i else ""
            parts=[p.strip() for p in l.split(",",1)]
            edu.append({"school":school,"degree_type":parts[0],"field":parts[1] if len(parts)>1 else ""})
        if grab and l.lower().startswith(("experience","skills","summary")): break
    return edu

def extract_skills(t:str)->list:
    kw=["Python","JavaScript","Marketing","Leadership","Design","Copywriting","UX","Analytics","SEO","Content"]
    return sorted({k for k in kw if k.lower() in t.lower()})

def jd_similarity(resume, jd)->float:
    v=TfidfVectorizer(stop_words="english").fit_transform([resume,jd])
    return cosine_similarity(v[0:1],v[1:2])[0][0]

# ─── API route ────────────────────────────────────────────────────────────────
@app.post("/parse", response_model=ResumeResponse)
async def parse(
    file:UploadFile=File(...),
    job_description:Optional[str]=Form(None),
    authorization:Optional[str]=Header(None)
):
    if AUTH and authorization!=f"Bearer {AUTH}":
        raise HTTPException(401,"Unauthorized")

    txt = extract_text_from_file(file.filename, await file.read())
    parsed = gpt_chunk_parse(txt)
    experience = parsed["experience"]
    education  = parsed["education"] or extract_education(txt)  # fallback

    data = dict(
        name = extract_name(txt),
        email= extract_email(txt),
        phone= extract_phone(txt),
        experience=experience,
        education =education,
        skills    =extract_skills(txt)
    )
    if job_description:
        data["match_score"]=round(jd_similarity(txt,job_description)*100,2)
    return data