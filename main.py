import os, re, json, tempfile
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz          # PyMuPDF
import docx2txt
import openai

# ───────────────────────── CORS / FastAPI ─────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rezmay.co", "https://www.rezmay.co"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTH = os.getenv("BASIC_AUTH_TOKEN", "")

# ───────────────────────── Pydantic model (OPTION B) ──────────────
class Experience(BaseModel):
    title: str
    company: str
    location: str
    years: str
    bullets: List[str] = []

class ResumeResponse(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    experience: List[Experience] = []
    education: List[str] = []
    skills: List[str] = []
    match_score: Optional[float] = None

# ───────────────────────── Helper functions ───────────────────────
def extract_text_from_file(filename: str, content: bytes) -> str:
    ext = filename.lower().split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(content); tmp_path = tmp.name
    if ext == "pdf":
        doc = fitz.open(tmp_path)
        return "\n".join(p.get_text() for p in doc)
    if ext == "docx":
        return docx2txt.process(tmp_path)
    raise ValueError("Unsupported file type")

def extract_name(text: str)  -> str:
    first_line = text.strip().split("\n", 1)[0]
    return first_line if "@" not in first_line else ""

def extract_email(text: str) -> str:
    m = re.search(r'[\w\.-]+@[\w\.-]+\.\w{2,}', text)
    return m.group(0) if m else ""

def extract_phone(text: str) -> str:
    m = re.search(r'(\+?\d{1,2}\s*)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', text)
    return m.group(0) if m else ""

def extract_skills(text: str) -> List[str]:
    common = ["Python","JavaScript","Marketing","Leadership",
              "Design","Copywriting","UX","Analytics","SEO","Content"]
    return sorted({s for s in common if s.lower() in text.lower()})

# ⁂  GPT call to structure “Experience”  ⁂
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def extract_experience_sections(text: str) -> List[dict]:
    prompt = f"""
You are a resume parser. Return ONLY valid JSON array like:
[{{"title":"","company":"","location":"","years":"","bullets":[]}}]

Resume:
\"\"\"{text[:6000]}\"\"\"   # truncate long resumes
"""
    try:
        r = client.chat.completions.create(
            model="gpt-3.5-turbo", temperature=0,
            messages=[{"role":"user","content":prompt}]
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        print("⚠️ GPT parse error →", e)
        return []

def extract_education(text: str) -> List[str]:
    section = re.split(r'\bexperience\b', text, flags=re.I)[0]
    deg = re.findall(r'(?:Bachelor|Master|BS|MS|MBA|PhD)[^\n]{0,120}', section, flags=re.I)
    return [d.strip() for d in deg]

def jd_similarity(resume: str, jd: str) -> float:
    vec  = TfidfVectorizer(stop_words='english').fit_transform([resume, jd])
    return cosine_similarity(vec[0:1], vec[1:2])[0][0]

# ───────────────────────── Endpoint ───────────────────────────────
@app.post("/parse", response_model=ResumeResponse)          # ← keep if you choose OPTION B
# @app.post("/parse")                                      # ← use this line instead for OPTION A
async def parse(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    if AUTH and authorization != f"Bearer {AUTH}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    raw = await file.read()
    txt  = extract_text_from_file(file.filename, raw)

    data = {
        "name":       extract_name(txt),
        "email":      extract_email(txt),
        "phone":      extract_phone(txt),
        "experience": extract_experience_sections(txt),
        "education":  extract_education(txt),
        "skills":     extract_skills(txt)
    }
    if job_description:
        data["match_score"] = round(jd_similarity(txt, job_description)*100, 2)

    # OPTION B – return object FastAPI can validate
    return ResumeResponse(**data)

    # OPTION A – just return raw dict (remove response_model above)
    # return data