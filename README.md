# RAG System

A simple RAG system with FastAPI backend and Gradio frontend.

## Installation

### 1. Install uv

`uv` is a fast Python package manager.

**Option A - Official installer (recommended):**

**Windows:**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Mac/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Option B - Using pip:**
```bash
pip install uv
```

Or download from: https://github.com/astral-sh/uv/releases

### 2. Install project dependencies

```bash
uv sync
```


## Quick Start

### 1. Create .env file

Create a `.env` file in the project root and add your API key:

```
OPENAI_API_KEY=sk-your-api-key-here
DEEPLAKE_API_KEY=sk-your-api-key-here

```

### 2. Run both services

**Terminal 1 - Backend (API):**
```bash
uv run backend
```
Runs on `http://localhost:8000`

**Terminal 2 - Frontend (Web interface):**
```bash
uv run frontend
```
Opens at `http://localhost:7860`

---

## Using the system

1. Open in browser: **http://localhost:7860**
2. Enter your question about AI and RAG theme in the input field 
3. The system will find relevant documents and provide an answer
4. Use the settings button to adjust parameters (Top-K, Temperature)

---

## Parameters

- **Top-K** - number of similar documents to use (1-20)
- **Temperature** - answer creativity (0.0 = precise, 1.0 = creative)
- **Retrieval metrics ** - simple cosine similarity or enchanced cosine similarity with spacy
---

## API Documentation

When the backend is running, view the documentation at:
- Swagger UI: **http://localhost:8000/docs**


## Troubleshooting

**Error: "Cannot connect to backend"**
- Make sure you started the backend in a separate terminal: `uv run backend`

**Error: "OPENAI_API_KEY not found"**
- Check that the `.env` file exists in the project root with the correct key
- Extension must be `.env` (not `.env.txt`)

**Slow performance**
- This is normal on first run (models and indexes are loading)
- Subsequent requests will be faster
