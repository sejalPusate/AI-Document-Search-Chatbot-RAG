from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ingestion import ingest_pdf
from retrieval import retrieve_answer
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    ingest_pdf(file_path)
    return {"message": f"{file.filename} ingested successfully."}

@app.post("/ask-question/")
async def ask_question(question: str = Form(...)):
    answer, sources = retrieve_answer(question)
    return {"answer": answer, "sources": sources}
