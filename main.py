from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai


# ──────────────────────────────────────────────
# Archer GRC System Prompt
# ──────────────────────────────────────────────
ARCHER_GRC_SYSTEM_PROMPT = """
You are an expert AI assistant exclusively dedicated to the Archer GRC platform
(Governance, Risk, and Compliance) by RSA Security / ArcherIRM.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR IDENTITY & SCOPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You ONLY respond to questions, tasks, and discussions that are directly related
to the Archer GRC platform. Your areas of expertise include:

• Archer Platform Architecture & Administration
• Archer GRC Modules & Use Cases
• Archer Configuration & Customization
• Archer Integrations & Data Management
• Archer Reporting & Analytics
• Archer Best Practices & Troubleshooting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT BEHAVIOR RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. REFUSE OFF-TOPIC QUESTIONS:
If unrelated to Archer GRC, respond exactly:

"I'm exclusively configured to assist with the Archer GRC platform.
Your question appears to be outside that scope. Please ask me anything
related to Archer GRC — modules, configuration, integrations, reporting,
risk frameworks, or administration."

2. NEVER INVENT FEATURES:
Do not fabricate Archer modules or capabilities.

3. USE CORRECT TERMINOLOGY:
Use official Archer terms like Applications, Levels, Fields,
iViews, DDEs, Questionnaires, and Data Feeds.

4. FRAMEWORK CONTEXT:
Discuss frameworks only in relation to Archer usage.

5. VERSION AWARENESS:
Mention SaaS vs On-Prem differences when relevant.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Keep ALL responses under 100 words
- Give complete answers
- Be concise and professional
- Use bullets or numbered steps when useful
- Start directly with the answer
- End with:
  "Would you like more detail on any step?"
"""


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Archer GRC AI Assistant",
    description="AI-powered assistant for Archer GRC using Gemini SDK.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    api_key: str
    message: str


class ChatResponse(BaseModel):
    response: str
    model: str = "gemini-2.5-flash"
    scope: str = "Archer GRC Platform"


# ──────────────────────────────────────────────
# Root Endpoint
# ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Archer GRC AI Assistant",
        "description": "Ask anything about the Archer GRC platform.",
        "docs": "/docs",
        "endpoints": {
            "POST /chat": "Single-turn Archer GRC chat",
            "GET /health": "Health check",
            "GET /topics": "Supported Archer GRC topics",
        },
    }


# ──────────────────────────────────────────────
# Health Endpoint
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "scope": "Archer GRC Only",
    }


# ──────────────────────────────────────────────
# Topics Endpoint
# ──────────────────────────────────────────────
@app.get("/topics")
def topics():
    return {
        "supported_topics": [
            "Archer Administration & Architecture",
            "Risk Management",
            "Policy & Compliance Management",
            "Audit Management",
            "Vendor / Third-Party Risk Management",
            "Incident Management",
            "Business Continuity",
            "Archer REST API",
            "Data Feeds & Integrations",
            "Dashboards & Reporting",
            "DDEs & Notifications",
            "Questionnaires & Assessments",
            "Archer SaaS vs On-Premise",
            "Archer Exchange",
            "Troubleshooting & Best Practices",
        ]
    }


# ──────────────────────────────────────────────
# Chat Endpoint
# ──────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    # Validate API key
    if not request.api_key.strip():
        raise HTTPException(
            status_code=400,
            detail="Gemini API key cannot be empty."
        )

    # Validate message
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # Limit message size
    if len(request.message) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Message too long."
        )

    try:
        # Configure Gemini SDK
        genai.configure(api_key=request.api_key)

        # Load Gemini model
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=ARCHER_GRC_SYSTEM_PROMPT,
        )

        # Generate response
        response = model.generate_content(
            request.message,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 1000,
            }
        )

        response_text = response.text.strip()

        return ChatResponse(response=response_text)

    except Exception as e:

        error_msg = str(e)

        print("\n========== GEMINI ERROR ==========")
        print(error_msg)
        print("==================================\n")

        error_lower = error_msg.lower()

        # Invalid API key
        if any(x in error_lower for x in [
            "api_key_invalid",
            "401",
            "403"
        ]):
            raise HTTPException(
                status_code=401,
                detail="Invalid Gemini API key."
            )

        # Rate limit / quota exceeded
        elif any(x in error_lower for x in [
            "429",
            "quota",
            "resourceexhausted",
            "rate limit",
            "too many requests"
        ]):
            raise HTTPException(
                status_code=429,
                detail="Gemini API rate limit exceeded."
            )

        # Model not found
        elif "404" in error_lower or "not_found" in error_lower:
            raise HTTPException(
                status_code=404,
                detail="Gemini model not found or unsupported."
            )

        # Generic error
        else:
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )


# ──────────────────────────────────────────────
# Run Locally
# ──────────────────────────────────────────────
#
# Save as: main.py
#
# Run:
#
# uvicorn main:app --reload
#
# Swagger Docs:
#
# http://127.0.0.1:8000/docs
#