# 🏛️ Archer GRC AI Assistant

A **FastAPI** + **LangChain** + **Gemini** powered AI assistant that is **exclusively scoped** to the **Archer GRC platform** (Governance, Risk & Compliance) by RSA / ArcherIRM.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
uvicorn main:app --reload
```

- API: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`

---

## 📡 Endpoints

| Method | Endpoint   | Description                              |
|--------|------------|------------------------------------------|
| GET    | `/`        | Service info                             |
| GET    | `/health`  | Health check                             |
| GET    | `/topics`  | List of all supported Archer GRC topics  |
| POST   | `/chat`    | Chat with the Archer GRC AI assistant    |

---

## 💬 Using `/chat`

### Single-turn request
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "YOUR_GEMINI_API_KEY",
    "message": "How do I create a Data Feed in Archer?"
  }'
```

### Multi-turn (with history)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "YOUR_GEMINI_API_KEY",
    "message": "Can you show me an example with JSON format?",
    "history": [
      {"role": "human", "content": "How do I create a Data Feed in Archer?"},
      {"role": "ai",    "content": "To create a Data Feed in Archer, navigate to ..."}
    ]
  }'
```

---

## 🔒 Scope Enforcement

The assistant is **strictly restricted** to Archer GRC topics:

✅ Allowed:
- Archer modules (Risk, Audit, Policy, TPRM, Incidents, BIA)
- Archer configuration (Applications, Fields, DDEs, Questionnaires)
- REST API & Data Feeds
- Reporting & Dashboards
- Compliance frameworks **within Archer context** (NIST, ISO 27001, SOX, GDPR, HIPAA, PCI-DSS)
- Archer SaaS vs On-Premise differences
- Troubleshooting & best practices

❌ Refused (politely redirected):
- General software development questions
- Non-Archer GRC topics
- Unrelated compliance questions outside Archer context

---

## 🏗️ Architecture

```
User Request
    │
    ▼
FastAPI (/chat endpoint)
    │
    ▼
LangChain ChatPromptTemplate
    ├── System Prompt  ──► Archer GRC Scope Enforcement
    ├── History        ──► Multi-turn conversation context
    └── Human Input    ──► User's question
    │
    ▼
ChatGoogleGenerativeAI (Gemini 2.0 Flash)
    │
    ▼
StrOutputParser ──► JSON Response
```

---

## 🔑 Getting a Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **Create API Key**
4. Use it in the `api_key` field of your requests

---

## 📚 Supported Archer GRC Topics

- Archer Platform Administration & Architecture  
- Risk Management (Enterprise, Operational, IT)  
- Policy & Compliance Management  
- Audit Management  
- Vendor / Third-Party Risk Management (TPRM)  
- Incident Management & Business Continuity  
- Regulatory Frameworks in Archer (NIST, ISO 27001, SOX, GDPR, HIPAA, PCI-DSS)  
- Application Builder & Configuration  
- Data Feeds & REST API Integration  
- Reporting, Dashboards & iViews  
- Data-Driven Events (DDEs) & Notifications  
- Questionnaires & Assessments  
- Archer Exchange & Packaged Content  
- Troubleshooting & Best Practices  
