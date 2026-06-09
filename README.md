---
title: Multi-Agent Financial Analyst
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI earnings report analyzer — multi-agent
---

# 📊 Multi-Agent Financial Analyst

An AI-powered financial analysis platform that processes company earnings reports (PDFs), extracts financial KPIs, analyzes risks, summarizes reports, and compares quarterly performance using a multi-agent architecture.

**Powered by [Groq](https://console.groq.com) (free API) — no local GPU needed.**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **KPI Extraction** | Hybrid regex + LLM extraction of revenue, EPS, margins, cash flow, debt, and more |
| ⚠️ **Risk Analysis** | Explainable risk scoring (0-100) with 6 weighted factors and evidence-backed reasoning |
| 📝 **AI Summary** | Executive summaries with management highlights, growth drivers, and challenges |
| 📈 **QoQ Comparison** | Quarter-over-quarter comparison with trend indicators and narrative insights |
| 🔍 **Smart OCR** | PaddleOCR for scanned/image-heavy PDFs with automatic detection |
| 📄 **PDF Processing** | pdfplumber + PyMuPDF for robust text and table extraction |
| 🧠 **RAG Pipeline** | FAISS vector store with SentenceTransformers for context-aware analysis |
| 🎨 **Premium Dashboard** | Streamlit frontend with glassmorphism UI, Plotly charts, and dark theme |

---

## 🔑 Setup (Hugging Face Spaces)

Set the following **Secrets** in your Space settings:

| Secret | Value |
|--------|-------|
| `GROQ_API_KEY` | Your free Groq API key from [console.groq.com](https://console.groq.com) |
| `LLM_PROVIDER` | `groq` |
| `GROQ_MODEL` | `llama-3.1-8b-instant` |

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
PDF Extraction (pdfplumber → PyMuPDF → PaddleOCR)
    │
    ▼
Text Cleaning & Chunking
    │
    ▼
FAISS Vector Store (SentenceTransformers)
    │
    ├────────────────┬─────────────────┐
    ▼                ▼                 ▼
KPI Agent      Risk Agent      Summary Agent
(regex+LLM)    (deterministic    (RAG+LLM)
               +LLM sentiment)
    │                │                 │
    └────────────────┴─────────────────┘
                     │
                     ▼
            Comparison Agent
            (QoQ analysis)
                     │
                     ▼
           Final Report Generator
```

**Agent Framework:** LangGraph StateGraph with typed Pydantic state models.

---

## 🚀 Local Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env
echo "LLM_PROVIDER=groq" >> .env
echo "GROQ_MODEL=llama-3.1-8b-instant" >> .env

# Launch dashboard
python run.py frontend
```

---

## 📊 Risk Scoring Algorithm

```
Overall Risk Score = Σ (factor_score × weight)

Factors:
├── Revenue Decline     × 0.25  (deterministic)
├── Profit Decline      × 0.20  (deterministic)
├── Margin Compression  × 0.15  (deterministic)
├── Debt Increase       × 0.15  (deterministic)
├── Cash Flow Health    × 0.15  (deterministic)
└── Sentiment Analysis  × 0.10  (LLM-based)

Risk Levels:
├── 0-30:   LOW      🟢
├── 30-60:  MEDIUM   🟡
├── 60-80:  HIGH     🟠
└── 80-100: CRITICAL 🔴
```

---

## 🔧 Supported LLM Providers

| Provider | Setup | Speed |
|----------|-------|-------|
| **Groq** (recommended for HF) | Add `GROQ_API_KEY` secret | ⚡ Very fast |
| **Ollama** (local only) | `ollama serve` + `ollama pull qwen3:8b` | Medium |

---

## 📄 License

MIT — for educational and research purposes.
