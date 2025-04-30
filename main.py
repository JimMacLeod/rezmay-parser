# main.py

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_resume(resume: UploadFile = File(...)):
    content = await resume.read()
    return {
        "filename": resume.filename,
        "size": len(content),
        "content_type": resume.content_type
    }

# Trigger redeploy: dummy comment