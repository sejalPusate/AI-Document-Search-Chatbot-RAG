from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from supabase import create_client
from utils.pdf_parser import extract_text_from_pdf
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

def ingest_pdf(file_path: str, table_name: str = "pdf_embeddings"):
    # Extract text
    text = extract_text_from_pdf(file_path)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # Create embeddings and store in Supabase
    for chunk in chunks:
        vector = embeddings.embed(chunk)
        supabase.table(table_name).insert({
            "content": chunk,
            "embedding": vector
        }).execute()

    print(f"✅ PDF ingested with {len(chunks)} chunks.")
