# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from ml_logic import compute_efficiency, find_swaps, extract_features  # Your ML

app = FastAPI(title="Talent AI Backend")

# CORS for frontend (Lovable React)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/upload-resumes")
async def process_resumes(
    files: list[UploadFile] = File(...),
    job_desc: str = Form(...)
):
    # Parse PDFs/CSVs to DF (use PyPDF2/pandas for resumes)
    df_list = []
    for file in files:
        content = await file.read()
        # Simulate: df = pd.read_csv(io.StringIO(content.decode()))  # Or PDF parse
        df_list.append(pd.DataFrame({'Resume_Text': [content.decode()], 'Candidate': [file.filename]}))
    df = pd.concat(df_list)
    
    job_skills = ['Python', 'Machine Learning']  # Extract from job_desc via regex/NLP
    
    # Run ML
    scored_df = compute_efficiency(df, job_skills=job_skills)
    swaps = find_swaps(scored_df)
    
    return {
        "rankings": scored_df[['Candidate', 'Efficiency_Score']].to_dict('records'),
        "swaps": swaps,
        "features_summary": scored_df.describe().to_dict()  # For debug
    }

@app.get("/health")
def health():
    return {"status": "Backend + ML Ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
