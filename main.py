import os
import re
import json
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz               # PyMuPDF
import docx2txt
import openai

# ───────────────────
#  FastAPI + CORS
# ───────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rezmay.co"],  # ← adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────
#  OpenAI
# ───────────────────
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
AUTH   = os.getenv("BASIC_AUTH_TOKEN", "")   # optional bearer

# ───────────────────
#  Helpers
# ───────────────────
def extract_text_from_file(fname: str, content: bytes) -> str:
    ext = fname.lower().split(".")[-1]
    if ext == "pdf":
        tmp = "/tmp/temp.pdf"
        open(tmp, "wb").write(content)
        doc = fitz.open(tmp)
        return "\n".join(pg.get_text() for pg in doc)
    elif ext == "docx":
        tmp = "/tmp/temp.docx"
        open(tmp, "wb").write(content)
        return docx2txt.process(tmp)
    raise ValueError("Unsupported file type")

def extract_name(text: str) -> str:
    first_line = text.strip().split("\n", 1)[0]
    return first_line.strip()

def extract_email(text: str) -> str:
    m = re.search(r'[\w\.-]+@[\w\.-]+\.[\w]+', text)
    return m.group(0) if m else ""

def extract_phone(text: str) -> str:
    m = re.search(r'(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', text)
    return m.group(0) if m else ""

# ───────────────────
#  AI-assisted EXPERIENCE
# ───────────────────
def extract_experience_sections(resume_text: str) -> list:
    prompt = f"""
You are an ATS résumé parser. Extract ONLY what is explicitly present.
Return JSON list:

[{{"title":"","company":"","location":"","years":""}}, …]

Résumé:
\"\"\"{resume_text}\"\"\"
JSON only:
"""
    try:
        rsp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return json.loads(rsp.choices[0].message.content)
    except Exception as e:
        print("AI exp parsing failed:", e)
        return []

# ───────────────────
#  Education  (improved / tolerant)
# ───────────────────
def extract_education(text: str) -> list:
    lines = [l.strip() for l in text.split("\n")]
    out, capture, buf = [], False, []

    deg_kw = re.compile(
        r'\b(Bachelor|Master|Associate|BS|BA|MS|MBA|MA|PhD|Doctor|Certificate)\b',
        re.I
    )

    for line in lines:
        if not capture and "education" in line.lower():
            capture = True
            continue

        if capture:
            if re.match(r'^\s*(experience|skills|summary|core)\b', line, re.I):
                break
            if line:
                buf.append(line)

            if not line or len(buf) >= 4:
                joined = " ".join(buf)
                m = deg_kw.search(joined)
                if m:
                    degree = joined[m.start():].split(",")[0].strip()
                    school = joined[:m.start()].strip()
                    field  = joined[m.end():].strip(" ,-")
                    out.append({
                        "school": school,
                        "degree_type": degree,
                        "field": field
                    })
                buf = []
    return out

def extract_skills(text: str) -> list:
    common = ["Python","JavaScript","Marketing","Leadership",
              "Design","Copywriting","UX","Analytics","SEO","Content"]
    return sorted({s for s in common if s.lower() in text.lower()})

def compare_with_job_description(resume_text: str, jd_text: str) -> float:
    vec = TfidfVectorizer(stop_words="english").fit_transform([resume_text, jd_text])
    return cosine_similarity(vec[0:1], vec[1:2])[0][0]

# ───────────────────
#  FastAPI endpoint
# ───────────────────
@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    if AUTH and authorization != f"Bearer {AUTH}":
        raise HTTPException(401, "Unauthorized")

    content = await file.read()
    resume_text = extract_text_from_file(file.filename, content)

    data = {
        "name": extract_name(resume_text),
        "email": extract_email(resume_text),
        "phone": extract_phone(resume_text),
        "experience": extract_experience_sections(resume_text),
        "education": extract_education(resume_text),
        "skills": extract_skills(resume_text)
    }

    if job_description:
        data["match_score"] = round(compare_with_job_description(resume_text, job_description)*100, 2)

    return data