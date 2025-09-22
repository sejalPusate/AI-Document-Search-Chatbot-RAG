from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from supabase import create_client
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_answer(query: str, table_name: str = "pdf_embeddings", top_k: int = 5):
    # Embed the query
    query_vector = embeddings.embed(query)

    # Fetch all embeddings from Supabase
    response = supabase.table(table_name).select("*").execute()
    docs = response.data

    # Compute similarity
    for doc in docs:
        doc["score"] = cosine_similarity(np.array(doc["embedding"]), np.array(query_vector))

    # Sort by highest similarity
    top_docs = sorted(docs, key=lambda x: x["score"], reverse=True)[:top_k]

    # Combine chunks
    context = "\n\n".join([d["content"] for d in top_docs])

    # Ask LLM
    answer = llm.predict(f"Answer the question based on context:\n{context}\nQuestion: {query}")
    return answer, top_docs
