from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import re, io, os, docx2txt
import fitz  # PyMuPDF

app = FastAPI(title="Rezmay Parser API")

# ────────────────────────────────
# Helpers
# ────────────────────────────────
EMAIL_RE  = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE  = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

DATE_RE   = re.compile(r"\d{2}/\d{2}\s*[–-]\s*(Present|\d{2}/\d{2})")
DEGREE_RE = re.compile(r"(Bachelor|Master|BS|MS|MBA|PhD|Certificate)", re.I)

SKILL_KEYWORDS = {
    "Marketing", "Design", "SEO", "Analytics", "Content", "Leadership",
    "Python", "Java", "HTML", "CSS", "WordPress", "Copywriting"
}

def extract_pdf_text(buffer: bytes) -> str:
    """Extract raw text from PDF bytes."""
    with fitz.open(stream=buffer, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

def extract_docx_text(buffer: bytes) -> str:
    """Extract raw text from DOCX bytes."""
    tmp = "_upload.docx"
    with open(tmp, "wb") as f:
        f.write(buffer)
    text = docx2txt.process(tmp)
    os.remove(tmp)
    return text or ""

def parse_resume(text: str) -> dict:
    """Very lightweight parser – good enough for MVP."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = "\n".join(lines)

    # Contact
    email  = EMAIL_RE.search(joined)
    phone  = PHONE_RE.search(joined)
    name   = lines[0] if lines else ""

    # Experience (grab Title / Company line followed by date line)
    experience = []
    for i, ln in enumerate(lines):
        if DATE_RE.search(ln) and i:
            title_company = lines[i-1]
            dates = DATE_RE.search(ln).group(0)
            experience.append(f"{title_company} ({dates})")

    # Education
    education = []
    edu_flag = False
    for ln in lines:
        if "education" in ln.lower():
            edu_flag = True
            continue
        if edu_flag:
            if DEGREE_RE.search(ln):
                education.append(ln)
            elif ln.lower().startswith(("skills", "experience")):
                break

    # Skills
    skills = sorted({kw for kw in SKILL_KEYWORDS if kw.lower() in joined.lower()})

    return {
        "name":   name,
        "email":  email.group(0)  if email  else "",
        "phone":  phone.group(0)  if phone  else "",
        "experience": experience,
        "education":  education,
        "skills":     skills
    }

# ────────────────────────────────
# Routes
# ────────────────────────────────
@app.get("/")
def health():
    return {"message": "It works!"}

@app.post("/upload")
async def upload_resume(resume: UploadFile = File(...)):
    ext = resume.filename.split(".")[-1].lower()
    if ext not in {"pdf", "docx"}:
        raise HTTPException(400, "Only PDF or DOCX allowed")

    buffer = await resume.read()
    text   = extract_pdf_text(buffer) if ext == "pdf" else extract_docx_text(buffer)
    data   = parse_resume(text)

    return JSONResponse(content=data)

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_resume(resume: UploadFile = File(...)):
    content = await resume.read()
    return {"filename": resume.filename, "size": len(content)}