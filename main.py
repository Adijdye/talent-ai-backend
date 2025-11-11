from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import re
from PyPDF2 import PdfReader  # For PDF parsing
from typing import List
from ml_logic import compute_efficiency, find_swaps  # Your ML functions

app = FastAPI(title="Talent AI Backend")

# CORS for frontend (Lovable React) - Restrict origins in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["http://localhost:3000", "https://your-app.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple function to extract skills from job_desc (expand with spaCy/BERT)
def extract_job_skills(job_desc: str) -> List[str]:
    # Common skills patterns (add more as needed)
    patterns = [
        r'\bpython\b', r'\bmachine learning\b', r'\bml\b', r'\bsql\b', r'\banalytics\b',
        r'\bjava\b', r'\bjavascript\b', r'\breact\b', r'\baws\b', r'\bdata science\b'
    ]
    text_lower = job_desc.lower()
    matches = set(re.findall('|'.join(patterns), text_lower))
    return list(matches) if matches else ['Python', 'Machine Learning']  # Fallback

@app.post("/upload-resumes")
async def process_resumes(
    files: List[UploadFile] = File(...),
    job_desc: str = Form(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # File size limit (total ~10MB)
    total_size = sum(len(await file.read()) for file in files)
    if total_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Total file size exceeds 10MB")
    
    # Reset file pointers (after size check)
    for file in files:
        await file.seek(0)
    
    df_list = []
    for file in files:
        try:
            content = await file.read()
            filename = file.filename or "unknown"
            
            if filename.endswith('.pdf'):
                # PDF parsing
                pdf_reader = PdfReader(io.BytesIO(content))
                text = ''.join(page.extract_text() or '' for page in pdf_reader.pages)
                df_list.append(pd.DataFrame({'Resume_Text': [text], 'Candidate': [filename]}))
            elif filename.endswith('.csv'):
                # CSV parsing
                csv_df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                csv_df['Candidate'] = filename  # Add candidate col if missing
                csv_df['Resume_Text'] = csv_df.apply(lambda row: ' '.join(row.astype(str)), axis=1)  # Concat to text
                df_list.append(csv_df[['Resume_Text', 'Candidate']])
            else:
                # Text files: assume UTF-8
                text = content.decode('utf-8', errors='ignore')
                df_list.append(pd.DataFrame({'Resume_Text': [text], 'Candidate': [filename]}))
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing {filename}: {str(e)}")
    
    if not df_list:
        raise HTTPException(status_code=400, detail="No valid files processed")
    
    df = pd.concat(df_list, ignore_index=True)
    if df.empty:
        raise HTTPException(status_code=400, detail="No data extracted from files")
    
    # Dynamic skills from job_desc
    job_skills = extract_job_skills(job_desc)
    
    try:
        # Run ML (assumes ml_logic handles df with 'Resume_Text' and 'Candidate')
        scored_df = compute_efficiency(df, job_skills=job_skills)
        swaps = find_swaps(scored_df)
        
        # Normalize scores if not already (0-100 for UI)
        if 'Efficiency_Score' in scored_df.columns:
            min_score = scored_df['Efficiency_Score'].min()
            max_score = scored_df['Efficiency_Score'].max()
            if max_score > min_score:
                scored_df['Efficiency_Score'] = 100 * (scored_df['Efficiency_Score'] - min_score) / (max_score - min_score)
        
        return {
            "rankings": scored_df[['Candidate', 'Efficiency_Score']].to_dict('records'),
            "swaps": swaps,
            "job_skills_extracted": job_skills,  # For transparency
            "features_summary": scored_df.describe(include='all').to_dict() if len(scored_df) > 0 else {},  # Debug
            "processed_candidates": len(scored_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML processing error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "Backend + ML Ready", "timestamp": pd.Timestamp.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
