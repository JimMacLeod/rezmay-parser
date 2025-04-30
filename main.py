from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.get("/")
def root():
    return {"message": "still working"}

@app.post("/upload")
async def upload_resume(resume: UploadFile = File(...)):
    try:
        print(f"Received file: {resume.filename}")
        content = await resume.read()
        size = len(content)
        print(f"File size: {size} bytes")
        return {
            "filename": resume.filename,
            "size": size,
            "content_type": resume.content_type
        }
    except Exception as e:
        print(f"Error while processing upload: {e}")
        return {"error": str(e)}