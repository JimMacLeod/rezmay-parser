"""
Rezmay résumé parser – v0.9
FastAPI on Railway; parses PDF/DOCX, GPT-extracts experience,
regex-extracts education/skills, optional JD similarity.
"""

import os, re, json, textwrap, fitz, docx2txt
from typing import List, Optional

import openai
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────  ENV / APP  ──────────────────────────
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

# ────────────────────────  RESPONSE MODEL  ──────────────────────
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

# ─────────────────────  RAW-TEXT EXTRACTION  ────────────────────
def extract_text(fname: str, data: bytes) -> str:
    ext = fname.split(".")[-1].lower()
    path = "/tmp/doc." + ext
    with open(path, "wb") as f: f.write(data)

    if ext == "pdf":
        return "\n".join(p.get_text() for p in fitz.open(path))
    if ext == "docx":
        return docx2txt.process(path)
    raise ValueError("Only PDF/DOCX allowed")

# ─────────────────────────  LIGHT REGEX  ────────────────────────
EMAIL_RE  = re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}")
PHONE_RE  = re.compile(r"(\+?\d{1,2}\s*)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")

def first_match(regex, txt): m=regex.search(txt); return m[0] if m else ""

def extract_name(txt:str):
    for ln in txt.splitlines():
        ln=ln.strip()
        if ln and not EMAIL_RE.search(ln) and not PHONE_RE.search(ln): return ln
    return ""

# ─────────────────  GPT-ASSISTED EXPERIENCE  ────────────────────
PROMPT_HEAD = textwrap.dedent("""
Return ONLY JSON; each object must look like:
{{
 "title": "", "company": "", "location": "", "years": "Start – End",
 "bullets": ["...", "..."]
}}
If info missing leave it blank. Resume chunk:
""").strip()

def gpt_exp(chunk:str)->List[dict]:
    msg=[{"role":"user","content":f"{PROMPT_HEAD}\n\n{chunk}\n\nJSON:"}]
    try:
        raw=client.chat.completions.create(model="gpt-3.5-turbo",temp=0,messages=msg
              ).choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        print("GPT exp chunk error:",e); return []

def refine_header(item:dict):
    """If first bullet got glued to header, detach it."""
    title=item.get("title","")
    if "•" in title:
        head, *rest = title.split("•")
        item["title"]=head.strip("–—- ").strip()
        item.setdefault("bullets",[]).insert(0,"•".join(rest).strip())
    # put comma after company if location exists
    if item["company"] and item["location"]:
        item["company"]=item["company"]+","
    return item

def extract_exp(full:str)->List[dict]:
    chunks=textwrap.wrap(full,2800,break_long_words=False)
    items=[]
    for c in chunks: items+=gpt_exp(c)
    seen,setitems=set(),[]
    for it in items:
        it=refine_header(it)
        key=(it["title"],it["company"],it["years"])
        if any(key) and key not in seen:
            seen.add(key); setitems.append(it)
    return setitems

# ─────────────────────  EDUCATION (regex)  ──────────────────────
DEGREE=re.compile(r"(Bachelor|Master|Associate|B\.?S\.?|M\.?S\.?|MBA|PhD)",re.I)
SCHOOL=re.compile(r"(University|College|Institute|School)",re.I)

def extract_edu(txt:str)->List[str]:
    edu=[]
    for ln in txt.splitlines():
        if DEGREE.search(ln) and SCHOOL.search(ln):
            edu.append(ln.strip(" •-"))
    return edu

# ───────────────────────  SKILLS (toy list)  ────────────────────
SKILLS=["Python","JavaScript","Marketing","Design","Leadership",
        "Content","UX","SEO","Analytics","Copywriting"]

def extract_sk(txt): return sorted({s for s in SKILLS if s.lower() in txt.lower()})

# ─────────────────────  JD SIMILARITY  ──────────────────────────
def jd_score(resume,jd):
    vec=TfidfVectorizer(stop_words='english').fit_transform([resume,jd])
    return round(cosine_similarity(vec[0:1],vec[1:2])[0][0]*100,2)

# ─────────────────────────  ROUTE  ─────────────────────────────
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