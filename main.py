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
- Responses must be COMPLETE and informative
- Target approximately 80–100 words
- Use numbered steps or bullets when useful
- Start directly with the answer
- Use professional Archer GRC terminology
- Never leave incomplete sentences
- Always finish the response naturally
- End with:
  "Would you like more detail on any step?"
- Remember one thing give detailed explanation about the topic with 250 words
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
# Request Model (POST)
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    api_key: str
    message: str


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
            "GET /chat": "Chat using query parameters",
            "POST /chat": "Chat using JSON body",
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
# Shared Gemini Function
# ──────────────────────────────────────────────
def generate_archer_response(api_key: str, message: str):

    # Configure Gemini SDK
    genai.configure(api_key=api_key)

    # Load Gemini model
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=ARCHER_GRC_SYSTEM_PROMPT,
    )

    # Generate response
    response = model.generate_content(
        message,
        generation_config={
            "temperature": 0.5,
            "max_output_tokens": 500,
        }
    )

    return response.text.strip()


# ──────────────────────────────────────────────
# GET Chat Endpoint
# Example:
# /chat?api_key=KEY&message=How do I create a Data Feed?
# ──────────────────────────────────────────────
@app.get("/chat")
def chat_get(api_key: str, message: str):

    # Validate API key
    if not api_key.strip():
        raise HTTPException(
            status_code=400,
            detail="Gemini API key cannot be empty."
        )

    # Validate message
    if not message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:

        response_text = generate_archer_response(api_key, message)

        return {
            "response": response_text,
            "model": "gemini-2.5-flash",
            "scope": "Archer GRC Platform"
        }

    except Exception as e:

        error_msg = str(e)

        print("\n========== GEMINI ERROR ==========")
        print(error_msg)
        print("==================================\n")

        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


# ──────────────────────────────────────────────
# POST Chat Endpoint
# ──────────────────────────────────────────────



# ──────────────────────────────────────────────
# Run Locally
# ──────────────────────────────────────────────
#
# Save as: main.py
#
# Install:
#
# pip install -r requirements.txt
#
# Run:
#
# uvicorn main:app --reload
#
# Swagger Docs:
#
# http://127.0.0.1:8000/docs
#
# Example GET Request:
#
# http://127.0.0.1:8000/chat?api_key=YOUR_KEY&message=How do I create a Data Feed in Archer?
#