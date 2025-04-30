from fastapi import FastAPI, UploadFile, File, HTTPException
import fitz  # PyMuPDF
import docx2txt, io

app = FastAPI()

def extract_text(data: bytes, filename: str) -> str:
    ext = filename.split('.')[-1].lower()
    if ext == "pdf":
        pdf = fitz.open(stream=data, filetype="pdf")
        return "".join(p.get_text() for p in pdf)
    if ext == "docx":
        return docx2txt.process(io.BytesIO(data))
    raise ValueError("Unsupported file type")

@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    try:
        text = extract_text(await file.read(), file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"excerpt": text[:400]}