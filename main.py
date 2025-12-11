import os
import re
import json
import dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import uuid
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import PyPDF2
from io import BytesIO
import docx

dotenv.load_dotenv()

try:
    import PyPDF2
except:
    PyPDF2 = None

try:
    import docx
except:
    docx = None


LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemini-2.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CHUNK_LENGTH = int(os.environ.get("CHUNK_LENGTH", 1000))
RAG_DATA_DIR = os.environ.get("RAG_DATA-DIR", "./data")
HF_API_KEY = os.environ.get("HF_API_KEY")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
PORT = int(os.environ.get("PORT", 8000))
CHROMA_DB_HOST = os.environ.get("CHROMA_DB_HOST", "localhost")


app = FastAPI(title="Chroma RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel(LLM_MODEL_NAME)

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
os.makedirs(RAG_DATA_DIR, exist_ok=True)

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY
)

if CHROMA_DB_HOST and CHROMA_DB_HOST != "localhost":
    db_client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=PORT)
else:
    db_client = chromadb.PersistentClient(path=RAG_DATA_DIR)


collection = db_client.get_or_create_collection(name="my_docs")

def get_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".txt") or lower.endswith(".md"):
        try:
            return content.decode("utf-8")
        except:
            return content.decode("latin-1", errors="ignore")


    if lower.endswith(".pdf"):
        if PyPDF2 is None:
            raise HTTPException(500, "PyPDF2 not installed")

        reader = PyPDF2.PdfReader(BytesIO(content))
        txt = []
        for p in reader.pages:
            try:
                txt.append(p.extract_text() or "")
            except:
                txt.append("")

        return "\n".join(txt)


    if lower.endswith(".docx"):
        if docx is None:
            raise HTTPException(500, "python-docx not installed")

        d = docx.Document(BytesIO(content))
        return "\n".join([p.text for p in d.paragraphs])


    try:
        return content.decode("utf-8")
    except:
        return content.decode("latin-1", errors="ignore")

def chunk_text(text, target_words=CHUNK_LENGTH):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunks = []
    current_words = 0
    chunk_start = 0

    for sentence in sentences:
        words_in_sentence = len(sentence.split())

        if current_words + words_in_sentence > target_words and current_chunks:
            chunk_text = " ".join(current_chunks)
            chunks.append((chunk_text, chunk_start, chunk_start + len(chunk_text)))
            chunk_start += len(chunk_text) + 1
            current_chunks = [sentence]
            current_words = words_in_sentence
        else:
            current_chunks.append(sentence)
            current_words += words_in_sentence

    if current_chunks:
        chunk_text = " ".join(current_chunks)
        chunks.append((chunk_text, chunk_start, chunk_start + len(chunk_text)))

    return chunks


@app.post("/upload")
def upload_files(files: List[UploadFile] = File(...), context: Optional[str] = Form(None)):
    if context is None:
        context = f"ctx-{uuid.uuid4().hex[:8]}"

    ctx_dir = os.path.join(RAG_DATA_DIR, context)
    os.makedirs(ctx_dir, exist_ok=True)
    file_dir = os.path.join(ctx_dir, "files")
    os.makedirs(file_dir, exist_ok=True)

    metadata_path = os.path.join(ctx_dir, "metadata.json")
    metadata = []
    if os.path.exists(metadata_path):
        metadata = json.load(open(metadata_path))

    new_vectors = []

    for f in files:
        content = f.file.read()
        text = get_text(f.filename, content)
        chunks = chunk_text(text)

        dest = os.path.join(file_dir, f.filename)
        with open(dest, "wb") as out:
            out.write(content)

        for chunk, s, e in chunks:
            vec = embed_model.encode(chunk).tolist()
            cid = str(uuid.uuid4().hex)
            meta = {
                "id": cid,
                "context": context,
                "filename": f.filename,
                "offset_start": s,
                "offset_end": e,
                "text": chunk,
            }
            metadata.append(meta)
   
    collection.upsert(ids=[cid], documents=[chunk], metadatas=[meta], embeddings=[vec])
    json.dump(metadata, open(metadata_path, "w"), indent=2)

    return {"context": context, "chunks": len(new_vectors)}

@app.post("/prompt")
def chat(context: str = Form(...), query: str = Form(...)):
    qvec = embed_model.encode(query).tolist()
    results = collection.query(query_embeddings=qvec, n_results=5)

    context_block = "\n".join(results["documents"][0])


    prompt = f"""
        Context: {context_block}

        Question: {query}
            
            Based on the context provided above, generate a succint and direct answer to the query above.
        """
    response = llm_model.generate_content(prompt)

    return {"answer": response.text, "context": context_block}

@app.get("/contexts")
def list_contexts():
    return [d for d in os.listdir(RAG_DATA_DIR) if os.path.isdir(os.path.join(RAG_DATA_DIR, d))]


@app.get("/context/{name}/metadata")
def get_metadata(name: str):
    p = os.path.join(RAG_DATA_DIR, name, "metadata.json")
    if not os.path.exists(p):
        raise HTTPException(404, "Context not found")
    return json.load(open(p))