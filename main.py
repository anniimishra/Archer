from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

import pypdf
import io
import uuid
import numpy as np


# ══════════════════════════════════════════════════════════════════
#  IN-MEMORY VECTOR STORE
#  { session_id: { "filename": str, "chunks": list, "embeddings": np.ndarray } }
# ══════════════════════════════════════════════════════════════════
VECTOR_STORE: dict = {}


# ══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════

KEY_DETAILS_PROMPT = """
You are a precise data-extraction engine for the Archer GRC RACE solution.
Extract ONLY these fields from the context. If absent, write "Not found".

📅 Document Title         :
📅 Document Date          :
📅 Assessment / Audit Date:
📅 Review / Next Review   :
📅 Due Dates / Deadlines  :
⚠️  Risk Score(s)          :
⚠️  Risk Rating(s)         :
⚠️  Inherent Risk          :
⚠️  Residual Risk          :
🔑 Control ID(s)          :
🔑 Finding ID(s)          :
📋 Compliance Frameworks  :
🏢 Business Unit / Dept   :
👤 Owner(s) / Assignee(s) :
📊 Status                 :
💰 Financial Impact       :
🏭 Vendor / Third-Party   :

Return ONLY the labeled list above. No commentary. No extra text.
"""

SUMMARY_PROMPT = """
You are an Archer GRC RACE solution analyst.
Produce a structured bullet-point summary using ONLY the context provided.

Group bullets under these headings (skip if not relevant):
🔴 Risk Management
🔵 Audit Management
🟢 Policy & Compliance
🟡 Vendor / Third-Party Risk
🟠 Incident Management
⚪ Business Continuity
🔷 Archer Workflow Recommendations

Rules:
- Bullet points ONLY — no prose paragraphs
- Use Archer terms: Applications, Levels, Fields, iViews, DDEs, Questionnaires, Data Feeds
- Under Archer Workflow Recommendations: suggest Applications, DDEs, Notifications, Questionnaires
- 15–25 concise actionable bullets total
- Never invent features not in Archer GRC
"""

CHAT_PROMPT = """
You are an Archer GRC RACE assistant.
Answer questions using ONLY the document context provided.
If the answer is not in the context, say:
"This information is not available in the uploaded document."
Frame answers in Archer GRC terminology where relevant.
Be concise and professional.
End with: "Would you like more detail on any step?"
"""


# ══════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Archer GRC RACE RAG Assistant",
    description="Upload PDF once → get session_id → use for all 3 endpoints.",
    version="7.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════

def get_client(api_key: str) -> genai.Client:
    """Return a new Google GenAI client for the given API key."""
    return genai.Client(api_key=api_key)


def extract_text(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract full text from PDF. Returns (text, page_count)."""
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text and text.strip():
            parts.append(f"[Page {i}]\n{text.strip()}")
    if not parts:
        raise ValueError(
            "No extractable text found. "
            "This may be a scanned PDF — please use a text-based PDF."
        )
    return "\n\n".join(parts), len(reader.pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def embed_texts(api_key: str, texts: list[str], task: str) -> np.ndarray:
    """
    Embed a list of texts using gemini-embedding-001 via the new Google GenAI SDK.
    task: 'RETRIEVAL_DOCUMENT' for chunks, 'RETRIEVAL_QUERY' for questions.
    """
    client = get_client(api_key)
    vectors = []
    for text in texts:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(task_type=task),
        )
        vectors.append(response.embeddings[0].values)
    return np.array(vectors, dtype=np.float32)


def retrieve_context(api_key: str, query: str, session_id: str, top_k: int) -> str:
    """
    Embed the query and return top-k most similar chunks
    from the session's vector store via cosine similarity.
    """
    store     = VECTOR_STORE[session_id]
    query_vec = embed_texts(api_key, [query], task="RETRIEVAL_QUERY")[0]

    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    m = store["embeddings"] / (np.linalg.norm(store["embeddings"], axis=1, keepdims=True) + 1e-10)
    scores  = m @ q
    indices = np.argsort(scores)[::-1][:top_k]

    return "\n\n---\n\n".join(store["chunks"][i] for i in indices)


def call_gemini(api_key: str, system_prompt: str, context: str,
                user_message: str, temperature: float = 0.3,
                max_tokens: int = 1500) -> str:
    """Inject retrieved context and call Gemini Flash via new SDK."""
    client = get_client(api_key)
    prompt = (
        f"Use ONLY the document context below to respond.\n\n"
        f"━━━━ DOCUMENT CONTEXT ━━━━\n{context}\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{user_message}"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text.strip()


def get_session(session_id: str) -> dict:
    """Fetch a session or raise a clean 404."""
    if session_id not in VECTOR_STORE:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Please upload your PDF first via POST /pdf/upload.",
        )
    return VECTOR_STORE[session_id]


# ══════════════════════════════════════════════════════════════════
#  ROOT & HEALTH
# ══════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "service" : "Archer GRC RACE RAG Assistant",
        "version" : "7.0.0",
        "flow": [
            "1. POST /pdf/upload      → upload PDF once, get session_id",
            "2. POST /pdf/key-details → session_id → key details (dates, scores, names …)",
            "3. POST /pdf/summary     → session_id → Archer RACE bullet-point summary",
            "4. POST /pdf/chat        → session_id + question → grounded answer from embeddings",
        ],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status"          : "ok",
        "version"         : "7.0.0",
        "active_sessions" : len(VECTOR_STORE),
    }


# ══════════════════════════════════════════════════════════════════
#  ENDPOINT 1 — POST /pdf/upload
#  Upload PDF once → chunk → embed → store → return session_id
# ══════════════════════════════════════════════════════════════════
@app.post("/pdf/upload")
async def upload_pdf(
    api_key : str        = Form(..., description="Your Gemini API key"),
    file    : UploadFile = File(..., description="PDF file to ingest"),
):
    """
    Upload your PDF once.
    Extracts text, creates chunks, generates embeddings (gemini-embedding-001),
    and stores everything under a session_id.
    Use that session_id for /key-details, /summary, and /chat.
    """
    if not api_key.strip():
        raise HTTPException(status_code=400, detail="Gemini API key cannot be empty.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(pdf_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large — max 20 MB.")

    try:
        # Step 1 — Extract text
        pdf_text, pages = extract_text(pdf_bytes)

        # Step 2 — Chunk
        chunks = chunk_text(pdf_text, chunk_size=800, overlap=100)

        # Step 3 — Embed all chunks
        embeddings = embed_texts(api_key, chunks, task="RETRIEVAL_DOCUMENT")

        # Step 4 — Store in memory
        session_id = str(uuid.uuid4())
        VECTOR_STORE[session_id] = {
            "filename"   : file.filename,
            "chunks"     : chunks,
            "embeddings" : embeddings,    # np.ndarray (n_chunks × embedding_dim)
        }

        return {
            "message"         : "PDF uploaded and embeddings created successfully.",
            "session_id"      : session_id,
            "filename"        : file.filename,
            "pages"           : pages,
            "chunks_created"  : len(chunks),
            "embedding_model" : "gemini-embedding-001",
            "next_steps": {
                "key_details" : f"POST /pdf/key-details  →  session_id: {session_id}",
                "summary"     : f"POST /pdf/summary      →  session_id: {session_id}",
                "chat"        : f"POST /pdf/chat         →  session_id: {session_id}",
            },
        }

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        print(f"\n[/pdf/upload ERROR] {e}\n")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════
#  ENDPOINT 2 — POST /pdf/key-details
#  session_id → retrieve relevant chunks → extract key fields
# ══════════════════════════════════════════════════════════════════
class SessionRequest(BaseModel):
    api_key    : str
    session_id : str

@app.post("/pdf/key-details")
def pdf_key_details(req: SessionRequest):
    """
    Extract key structured data: dates, risk scores, owner names,
    control IDs, compliance frameworks, statuses, financial impact, etc.
    Uses embedding search to find the most relevant chunks.
    """
    if not req.api_key.strip():
        raise HTTPException(status_code=400, detail="Gemini API key cannot be empty.")
    get_session(req.session_id)

    try:
        query   = (
            "document date assessment date risk score risk rating control ID "
            "finding ID compliance framework owner assignee status financial impact vendor"
        )
        context = retrieve_context(req.api_key, query, req.session_id, top_k=8)
        result  = call_gemini(
            api_key       = req.api_key,
            system_prompt = KEY_DETAILS_PROMPT,
            context       = context,
            user_message  = "Extract all key details from the context above.",
            temperature   = 0.1,
            max_tokens    = 800,
        )
        return {
            "endpoint"    : "/pdf/key-details",
            "session_id"  : req.session_id,
            "filename"    : VECTOR_STORE[req.session_id]["filename"],
            "model"       : "gemini-2.5-flash",
            "key_details" : result,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n[/pdf/key-details ERROR] {e}\n")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════
#  ENDPOINT 3 — POST /pdf/summary
#  session_id → retrieve representative chunks → RACE bullet summary
# ══════════════════════════════════════════════════════════════════
@app.post("/pdf/summary")
def pdf_summary(req: SessionRequest):
    """
    Generate an Archer RACE structured bullet-point summary.
    Groups bullets under Risk, Audit, Compliance, Vendor,
    Incident, BCP, and Archer Workflow Recommendations.
    """
    if not req.api_key.strip():
        raise HTTPException(status_code=400, detail="Gemini API key cannot be empty.")
    get_session(req.session_id)

    try:
        query   = (
            "risk audit compliance vendor incident business continuity "
            "findings recommendations controls policy"
        )
        context = retrieve_context(req.api_key, query, req.session_id, top_k=10)
        result  = call_gemini(
            api_key       = req.api_key,
            system_prompt = SUMMARY_PROMPT,
            context       = context,
            user_message  = "Summarise this document as Archer RACE bullet points.",
            temperature   = 0.4,
            max_tokens    = 2000,
        )
        return {
            "endpoint"            : "/pdf/summary",
            "session_id"          : req.session_id,
            "filename"            : VECTOR_STORE[req.session_id]["filename"],
            "model"               : "gemini-2.5-flash",
            "archer_race_summary" : result,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n[/pdf/summary ERROR] {e}\n")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════
#  ENDPOINT 4 — POST /pdf/chat
#  session_id + question → embed question → top-k search → answer
# ══════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    api_key    : str
    session_id : str
    question   : str

@app.post("/pdf/chat")
def pdf_chat(req: ChatRequest):
    """
    Ask any question about the uploaded PDF.
    Embeds your question, runs cosine similarity over stored chunk
    embeddings, and answers strictly from the top matching chunks.
    No PDF re-upload needed — just session_id + question.
    """
    if not req.api_key.strip():
        raise HTTPException(status_code=400, detail="Gemini API key cannot be empty.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    get_session(req.session_id)

    try:
        context = retrieve_context(req.api_key, req.question, req.session_id, top_k=6)
        answer  = call_gemini(
            api_key       = req.api_key,
            system_prompt = CHAT_PROMPT,
            context       = context,
            user_message  = f"Question: {req.question}",
            temperature   = 0.4,
            max_tokens    = 1200,
        )
        return {
            "endpoint"   : "/pdf/chat",
            "session_id" : req.session_id,
            "filename"   : VECTOR_STORE[req.session_id]["filename"],
            "question"   : req.question,
            "answer"     : answer,
            "model"      : "gemini-2.5-flash",
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n[/pdf/chat ERROR] {e}\n")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════
#  Run Locally
# ══════════════════════════════════════════════════════════════════
#
#  IMPORTANT — install the NEW Google GenAI SDK (not the old one):
#    pip install fastapi uvicorn google-genai pypdf python-multipart numpy
#
#  If you had the old SDK installed, uninstall it first:
#    pip uninstall google-generativeai -y
#
#  Run:
#    uvicorn main:app --reload
#
#  Swagger UI:
#    http://127.0.0.1:8000/docs
#
# ──────────────────────────────────────────────────────────────────
#  COMPLETE FLOW
# ──────────────────────────────────────────────────────────────────
#
#  STEP 1 — Upload PDF (do this once)
#  curl -X POST http://localhost:8000/pdf/upload \
#       -F "api_key=YOUR_KEY" -F "file=@report.pdf"
#  → returns session_id
#
#  STEP 2a — Key Details
#  curl -X POST http://localhost:8000/pdf/key-details \
#       -H "Content-Type: application/json" \
#       -d '{"api_key":"YOUR_KEY","session_id":"SESSION_ID"}'
#
#  STEP 2b — Summary
#  curl -X POST http://localhost:8000/pdf/summary \
#       -H "Content-Type: application/json" \
#       -d '{"api_key":"YOUR_KEY","session_id":"SESSION_ID"}'
#
#  STEP 2c — Chat (repeat as many times as needed, no re-upload)
#  curl -X POST http://localhost:8000/pdf/chat \
#       -H "Content-Type: application/json" \
#       -d '{"api_key":"YOUR_KEY","session_id":"SESSION_ID","question":"What is the residual risk?"}'
#