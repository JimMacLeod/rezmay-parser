from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ... [existing imports and functions stay unchanged] ...

@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    authorization: str | None = Header(None)
):
    if AUTH and authorization != f"Bearer {AUTH}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    content = await file.read()
    try:
        resume_text = extract_text_from_file(file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    data = {
        "name": extract_name(resume_text),
        "email": extract_email(resume_text),
        "phone": extract_phone(resume_text),
        "experience": extract_experience_sections(resume_text),
        "education": extract_education(resume_text),
        "skills": extract_skills(resume_text)
    }

    if job_description:
        score = compare_with_job_description(resume_text, job_description)
        data["match_score"] = round(score * 100, 2)  # e.g. 76.43

    return data


def compare_with_job_description(resume_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0]