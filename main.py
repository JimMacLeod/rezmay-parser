from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.get("/")
def root():
    return {"message": "ready for in-memory upload"}

@app.post("/upload")
async def upload_resume(resume: UploadFile = File(...)):
    try:
        filename = resume.filename
        content_type = resume.content_type
        # just read a few bytes to avoid crashing
        preview = await resume.read(100)
        return {
            "filename": filename,
            "content_type": content_type,
            "preview": preview.decode(errors="ignore")
        }
    except Exception as e:
        return {"error": str(e)}