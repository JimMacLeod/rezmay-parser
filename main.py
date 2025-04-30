import os, re, io, json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz           # PyMuPDF
import docx2txt
import openai

# ────────────────────────────────
# Config
# ────────────────────────────────
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
AUTH       = os.getenv("BASIC_AUTH_TOKEN", "")

client = openai.OpenAI(api_key=OPENAI_KEY)

app = FastAPI(title="Rezmay Parser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rezmay.co"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────
# Utility functions
# ────────────────────────────────
EMAIL_RE  = re.compile(r'[\w\.-]+@[\w\.-]+')
PHONE_RE  = re.compile(r'(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}')

def extract_text(filename: str, data: bytes) -> str:
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext == "pdf":
        with fitz.open(stream=data, filetype="pdf") as pdf:
            return "\n".join(p.get_text() for p in pdf)
    elif ext == "docx":
        tmp = "/tmp/upload.docx"
        with open(tmp, "wb") as f:
            f.write(data)
        return docx2txt.process(tmp) or ""
    raise ValueError("Unsupported file type")

def extract_basic_fields(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = "\n".join(lines)
    return {
        "name" : lines[0] if lines else "",
        "email": EMAIL_RE.search(joined).group(0) if EMAIL_RE.search(joined) else "",
        "phone": PHONE_RE.search(joined).group(0) if PHONE_RE.search(joined) else ""
    }

def ai_extract_experience(text: str) -> list:
    prompt = f"""
You are a strict JSON resume parser. Extract each job with fields:
"title", "company", "location", "years". No guessing.

Return:
[
  {{"title":"...","company":"...","location":"...","years":"..."}},
  ...
]

Resume text:
\"\"\"{text}\"\"\"
Return only JSON.
"""
    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print("AI experience parse failed:", e)
        return []

def extract_education(text: str) -> list:
    degree_kw = re.compile(r'\b(Bachelor|Master|BS|MS|MBA|PhD|Certificate|Associate)\b', re.I)
    edu, capture = [], False
    for line in text.splitlines():
        l = line.strip()
        if 'education' in l.lower():
            capture = True; continue
        if capture:
            if degree_kw.search(l):
                edu.append(l)
            elif l.lower().startswith(('experience','skills','summary')):
                break
    return edu

def extract_skills(text: str) -> list:
    keywords = {"Marketing","Design","SEO","Analytics","Content","Leadership",
                "Python","Java","HTML","CSS","WordPress","Copywriting"}
    return sorted({k for k in keywords if k.lower() in text.lower()})

def jd_similarity(resume: str, jd: str) -> float:
    vect = TfidfVectorizer(stop_words='english').fit_transform([resume, jd])
    return cosine_similarity(vect[0:1], vect[1:2])[0][0]

# ────────────────────────────────
# Routes
# ────────────────────────────────
@app.get("/")
def health():
    return {"message": "It works!"}

@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    if AUTH and authorization != f"Bearer {AUTH}":
        raise HTTPException(401, "Unauthorized")

    data = await file.read()
    try:
        text = extract_text(file.filename, data)
    except Exception as e:
        raise HTTPException(400, str(e))

    result = extract_basic_fields(text)
    result["experience"] = ai_extract_experience(text)
    result["education"]  = extract_education(text)
    result["skills"]     = extract_skills(text)

    if job_description:
        match = jd_similarity(text, job_description)
        result["match_score"] = round(match * 100, 2)

    return JSONResponse(result)