# Research Paper Intelligence System (Agentic AI)

An agentic AI system for exploring and analyzing ML research papers from ArXiv. It combines semantic + keyword hybrid search, keyword/entity extraction, and an LLM-powered agent with custom tools to answer natural-language questions about papers, extract keywords, and compare papers side by side.

## Features

- **Dataset**: Loads the `CShorten/ML-ArXiv-Papers` dataset (~117k ML paper titles + abstracts) via Hugging Face `datasets`.
- **Semantic Search**: Generates dense embeddings for paper abstracts using `sentence-transformers` and indexes them with `FAISS` for fast similarity search.
- **Hybrid Retrieval**: Combines semantic (embedding) search with keyword/NER-based filtering for more precise retrieval.
- **Keyword Extraction**: Uses `KeyBERT` to pull out top keywords from any paper abstract or user-supplied text.
- **Named Entity Recognition**: Extracts entities (models, datasets, techniques, etc.) from abstracts using a transformer-based NER pipeline.
- **Agentic Tool Use**: A LangChain agent (backed by `langchain-groq`) is equipped with custom tools to:
  - Extract keywords from a topic/paper
  - Retrieve the closest matching paper via FAISS
  - Compare two papers across research objective, methodology, key contributions, advantages, limitations, and applications
- **LLM Reasoning**: The agent decides which tool to call, executes it, and synthesizes a natural-language response grounded in the tool output.

## Architecture

```
User Query
   │
   ▼
LangChain Agent (Groq LLM)
   │
   ├── Tool: keyword_extraction (KeyBERT)
   ├── Tool: semantic_search (SentenceTransformers + FAISS)
   └── Tool: compare_papers (FAISS retrieval + LLM comparison)
   │
   ▼
Final Natural-Language Response
```

## Tech Stack

| Component            | Library/Tool                         |
|-----------------------|---------------------------------------|
| Dataset loading       | `datasets` (Hugging Face)             |
| Embeddings             | `sentence-transformers`               |
| Vector search          | `faiss-cpu`                           |
| Keyword extraction      | `keybert`                              |
| NER / transformers      | `transformers`                         |
| Agent framework          | `langchain`, `langchain-community`, `langchain-core`, `langchain-huggingface` |
| LLM backend              | `langchain-groq`                        |
| Data handling             | `pandas`, `numpy`, `scikit-learn`        |

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

You'll also need a Groq API key set as an environment variable:

```bash
export GROQ_API_KEY="your-key-here"
```

## Usage

Open `Coding_Blocks_Research_Paper_Intelligence_System.ipynb` in Jupyter or Google Colab and run the cells in order:

1. Install dependencies
2. Load and prepare the ArXiv ML papers dataset
3. Generate embeddings and build the FAISS index
4. Set up the NER and keyword extraction pipelines
5. Define agent tools (keyword extraction, paper comparison)
6. Run the agent with a natural-language query

Example query:
```
Extract the top 5 keywords from "Deep Learning for Medical Image Reconstruction".
```

## Project Structure

```
.
├── Coding_Blocks_Research_Paper_Intelligence_System.ipynb   # Main notebook
├── requirements.txt                                          # Python dependencies
└── README.md                                                  # This file
```

## Future Improvements

- Add a web UI (Streamlit/Gradio) for interactive querying
- Persist the FAISS index to disk instead of rebuilding on every run
- Add caching for repeated queries
- Expand tool set (e.g., citation lookup, summarization tool)
- Add automated tests for tool outputs

## Notes

This notebook was run interactively in Google Colab; some cells (particularly agent invocation calls) may need to be re-run in sequence for full reproducibility, as one execution was interrupted mid-run in the saved version.
