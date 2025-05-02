"""
Rezmay résumé parser – v1.0
Stable base: GPT-extracts experience, regex-gets contact/edu/skills, JD scoring.
"""

import os, re, json, textwrap, docx2txt
from typing import List, Optional
from pypdf import PdfReader

import openai
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────  APP + CORS + AUTH  ────────────────
client   = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
AUTH_KEY = os.getenv("BASIC_AUTH_TOKEN", "")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rezmay.co"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────  RESPONSE MODEL s  ────────────────
class ExpItem(BaseModel):
    title: str = ""
    company: str = ""
    location: str = ""
    years: str = ""
    bullets: List[str] = []

class Resume(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    experience: List[ExpItem] = []
    education: List[str] = []
    skills: List[str] = []
    match_score: Optional[float] = None

    @field_validator("experience", mode="before")
    @classmethod
    def force_list(cls, v):
        return v if isinstance(v, list) else [v] if v else []

# ────────────────  TEXT EXTRACTION  ────────────────
def extract_text(fname: str, data: bytes) -> str:
    ext = fname.split(".")[-1].lower()
    path = "/tmp/doc." + ext
    with open(path, "wb") as f: f.write(data)

    if ext == "pdf":
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    if ext == "docx":
        return docx2txt.process(path)
    raise ValueError("Only PDF/DOCX allowed")

# ────────────────  CONTACT INFO  ────────────────
EMAIL_RE  = re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}")
PHONE_RE  = re.compile(r"(\+?\d{1,2}\s*)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")

def first_match(regex, txt): m=regex.search(txt); return m[0] if m else ""

def extract_name(txt:str):
    for ln in txt.splitlines():
        ln=ln.strip()
        if ln and not EMAIL_RE.search(ln) and not PHONE_RE.search(ln): return ln
    return ""

# ────────────────  GPT EXPERIENCE  ────────────────
PROMPT_HEAD = textwrap.dedent("""
Return ONLY JSON; each object must look like:
{{
 "title": "", "company": "", "location": "", "years": "Start – End",
 "bullets": ["...", "..."]
}}
If info missing leave it blank. Resume chunk:
""").strip()

def gpt_exp(chunk:str)->List[dict]:
    print("GPT chunk start ===")
    print(chunk[:300])
    print("...end ===")
    msg=[{"role":"user","content":f"{PROMPT_HEAD}\n\n{chunk}\n\nJSON:"}]
    try:
        raw=client.chat.completions.create(
            model="gpt-3.5-turbo", temperature=0, messages=msg
        ).choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        print("GPT exp chunk error:",e)
        return []

def refine_header(item:dict):
    title=item.get("title","")
    if "•" in title:
        head, *rest = title.split("•")
        item["title"]=head.strip("–—- ").strip()
        item.setdefault("bullets",[]).insert(0,"•".join(rest).strip())
    if item["company"] and item["location"]:
        item["company"] += ","
    return item

def extract_exp(full:str)->List[dict]:
    chunks=textwrap.wrap(full, 2500, break_long_words=False)
    items=[]
    for c in chunks: items += gpt_exp(c)
    seen, out = set(), []
    for it in items:
        it = refine_header(it)
        key = (it["title"], it["company"], it["years"])
        if any(key) and key not in seen:
            seen.add(key)
            out.append(it)
    return out

# ────────────────  EDUCATION (regex)  ────────────────
DEGREE=re.compile(r"(Bachelor|Master|Associate|B\.?S\.?|M\.?S\.?|MBA|PhD)",re.I)
SCHOOL=re.compile(r"(University|College|Institute|School)",re.I)

def extract_edu(txt:str)->List[str]:
    edu=[]
    for ln in txt.splitlines():
        if (DEGREE.search(ln) or SCHOOL.search(ln)) and len(ln.strip()) > 10:
            edu.append(ln.strip(" •-"))
    return edu

# ────────────────  SKILLS (expanded)  ────────────────
SKILLS = [
    "Python", "JavaScript", "HTML", "CSS", "SQL", "Leadership", "UX", "UI",
    "Content", "SEO", "Marketing", "Analytics", "Copywriting", "Product",
    "Design", "Strategy", "Branding", "AI", "Campaigns", "CRM", "CMS"
]

def extract_sk(txt): return sorted({s for s in SKILLS if s.lower() in txt.lower()})

# ────────────────  JD SIMILARITY  ────────────────
def jd_score(resume,jd):
    vec=TfidfVectorizer(stop_words='english').fit_transform([resume,jd])
    return round(cosine_similarity(vec[0:1],vec[1:2])[0][0]*100,2)

# ────────────────  ROUTE  ────────────────
@app.post("/parse", response_model=Resume)
async def parse(
    file:UploadFile=File(...),
    job_description:Optional[str]=Form(None),
    authorization:Optional[str]=Header(None)
):
    if AUTH_KEY and authorization!=f"Bearer {AUTH_KEY}":
        raise HTTPException(401,"Unauthorized")

    txt=extract_text(file.filename, await file.read())
    data={
        "name":extract_name(txt),
        "email":first_match(EMAIL_RE,txt),
        "phone":first_match(PHONE_RE,txt),
        "experience":extract_exp(txt),
        "education":extract_edu(txt),
        "skills":extract_sk(txt)
    }
    if job_description: data["match_score"]=jd_score(txt,job_description)
    return data