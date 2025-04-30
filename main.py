from fastapi import FastAPI, UploadFile, File
import shutil

app = FastAPI()

@app.get("/")
def root():
    return {"message": "upload test"}

@app.post("/upload")
async def upload_resume(resume: UploadFile = File(...)):
    try:
        file_location = f"/tmp/{resume.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(resume.file, buffer)
        return {
            "filename": resume.filename,
            "saved_to": file_location,
            "content_type": resume.content_type
        }
    except Exception as e:
        return {"error": str(e)}