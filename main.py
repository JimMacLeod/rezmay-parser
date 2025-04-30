import os, re, json
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz, docx2txt, openai

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
AUTH   = os.getenv("BASIC_AUTH_TOKEN", "")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rezmay.co"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ────────── file helpers 2 ──────────
def extract_text_from_file(fname: str, blob: bytes) -> str:
    ext = fname.lower().split(".")[-1]
    tmp = f"/tmp/temp.{ext}"
    open(tmp, "wb").write(blob)
    if ext == "pdf":
        return "\n".join(pg.get_text() for pg in fitz.open(tmp))
    if ext == "docx":
        return docx2txt.process(tmp)
    raise ValueError("Unsupported file type")

# ────────── quick regex fields ──────────
def extract_name(t):  return t.split("\n",1)[0].strip()
def extract_email(t): m=re.search(r'[\w\.-]+@[\w\.-]+\.\w+',t);return m.group(0) if m else""
def extract_phone(t): m=re.search(r'(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}',t);return m.group(0) if m else""

# ────────── EXPERIENCE via GPT (with bullets) ──────────
def extract_experience_sections(resume_text:str)->List[Dict]:
    prompt=f"""
You are an ATS parser. Extract ONLY what is explicitly in the text.
Return JSON list like:

[
 {{"title":"","company":"","location":"","years":"","bullets":["• ...","• ..."]}},
 ...
]

Text:
\"\"\"{resume_text}\"\"\""""
    try:
        rsp=client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{{"role":"user","content":prompt}}],
            temperature=0
        )
        return json.loads(rsp.choices[0].message.content)
    except Exception as e:
        print("AI exp error:",e);return[]

# ────────── EDUCATION regex pass ──────────
deg_kw=re.compile(r'\b(Bachelor|Master|Associate|BS|BA|MS|MBA|MA|PhD|Doctor|Certificate)\b',re.I)
def extract_education(txt:str)->List[Dict]:
    lines=[l.strip() for l in txt.split("\n")]
    capture=False; out=[]
    for i,l in enumerate(lines):
        if not capture and "education" in l.lower():
            capture=True;continue
        if capture and re.match(r'^\s*(experience|skills|summary|core)\b',l,re.I):
            break
        if capture and l:
            m=deg_kw.search(l)
            if m:
                degree_type=m.group(0)
                rest=l[m.end():].strip(" ,-")
                school=lines[i-1].strip()
                # cleanup stray words
                school=re.sub(r'\bEducation\b','',school,flags=re.I).strip()
                field=rest
                out.append({ "school":school, "degree_type":degree_type, "field":field })
    # dedupe
    dedup=[]
    for e in out:
        if e not in dedup: dedup.append(e)
    return dedup

def extract_skills(t:str)->List[str]:
    common=["Python","JavaScript","Marketing","Leadership","Design","Copywriting","UX","Analytics","SEO","Content"]
    return sorted({s for s in common if s.lower() in t.lower()})

def jd_similarity(r,j):
    v=TfidfVectorizer(stop_words="english").fit_transform([r,j])
    return cosine_similarity(v[0:1],v[1:2])[0][0]

# ────────── API ──────────
@app.post("/parse")
async def parse(
    file:UploadFile=File(...),
    job_description:Optional[str]=Form(None),
    authorization:Optional[str]=Header(None)
):
    if AUTH and authorization!=f"Bearer {AUTH}": raise HTTPException(401,"Unauthorized")
    txt=extract_text_from_file(file.filename,await file.read())

    data={{"name":extract_name(txt),"email":extract_email(txt),"phone":extract_phone(txt),
           "experience":extract_experience_sections(txt),
           "education":extract_education(txt),
           "skills":extract_skills(txt)}}

    if job_description: data["match_score"]=round(jd_similarity(txt,job_description)*100,2)
    return data