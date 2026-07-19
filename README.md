# Agentic AI Research Paper Intelligence System

An agentic AI system for exploring and analyzing ML research papers from ArXiv. It combines semantic + keyword hybrid search, keyword/entity extraction, and an LLM-powered ReAct agent with custom tools to answer natural-language questions about papers, extract keywords, and compare papers side by side.

## Features

- **Dataset**: Loads the `CShorten/ML-ArXiv-Papers` dataset (~117k ML paper titles + abstracts) via Hugging Face `datasets`.
- **Semantic Search**: Generates dense embeddings for paper abstracts using `sentence-transformers` and indexes them with `FAISS` for fast similarity search.
- **Hybrid Retrieval**: Combines semantic (embedding) search with keyword/NER-based filtering for more precise retrieval.
- **Keyword Extraction**: Uses `KeyBERT` to pull out top keywords from any paper abstract or user-supplied text.
- **Named Entity Recognition**: Extracts entities (models, datasets, techniques, etc.) from abstracts using a transformer-based NER pipeline.
- **Summarization**: Generates abstractive summaries of papers using a `facebook/bart-large-cnn` summarization pipeline wrapped as a LangChain `HuggingFacePipeline`.
- **Agentic Tool Use**: A LangChain ReAct agent (`create_react_agent` + `AgentExecutor`, backed by a Groq LLM) is equipped with custom tools to:
  - Extract keywords from a topic/paper
  - Retrieve the closest matching paper via FAISS
  - Compare two papers based on their abstracts (research objective, methodology, key contributions, advantages, limitations, and applications)
- **LLM Reasoning**: The agent decides which tool to call, executes it, and synthesizes a natural-language response grounded in the tool output.

## Architecture

```
User Query
   │
   ▼
LangChain ReAct Agent (AgentExecutor, Groq LLM)
   │
   ├── Tool: keyword_extraction (KeyBERT)
   ├── Tool: semantic_search (SentenceTransformers + FAISS)
   ├── Tool: summarization (BART via HuggingFacePipeline)
   └── Tool: compare_papers (FAISS retrieval + LLM comparison)
   │
   ▼
Final Natural-Language Response (response["output"])
```

## Tech Stack

| Component            | Library/Tool                         |
|-----------------------|---------------------------------------|
| Dataset loading       | `datasets` (Hugging Face)             |
| Embeddings             | `sentence-transformers`               |
| Vector search          | `faiss-cpu`                           |
| Keyword extraction      | `keybert`                              |
| NER / summarization      | `transformers` (`facebook/bart-large-cnn`) |
| Agent framework          | `langchain`, `langchain-community`, `langchain-core`, `langchain-huggingface` |
| LLM backend              | `langchain-groq`                        |
| Data handling             | `pandas`, `numpy`, `scikit-learn`        |

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

You'll also need API keys/tokens set as environment variables:

```bash
export GROQ_API_KEY="your-groq-key-here"
export HF_TOKEN="your-huggingface-token-here"
```

(If running in Google Colab, these are read via `google.colab.userdata` instead.)

## Usage

Open `Agentic_AI_Research_Paper_Intelligence_System.ipynb` in Jupyter or Google Colab and run the cells in order:

1. Install dependencies
2. Set API keys (Groq + Hugging Face)
3. Load and prepare the ArXiv ML papers dataset
4. Generate embeddings and build the FAISS index
5. Set up the NER, summarization, and keyword extraction pipelines
6. Define agent tools (keyword extraction, paper comparison)
7. Build the ReAct agent (`create_react_agent` + `AgentExecutor`)
8. Run the agent with a natural-language query

Example queries:
```
Extract the top 5 keywords from Deep Learning for Medical Image Reconstruction.
Find the top 3 research papers on Vision Transformer.
```

## Project Structure

```
.
├── Agentic_AI_Research_Paper_Intelligence_System.ipynb   # Main notebook
├── requirements.txt                                        # Python dependencies
└── README.md                                                # This file
```

## Future Improvements

- Add a web UI (Streamlit/Gradio) for interactive querying
- Persist the FAISS index to disk instead of rebuilding on every run
- Add caching for repeated queries
- Expand tool set (e.g., citation lookup)
- Add automated tests for tool outputs

## Notes

This notebook was developed and run interactively in Google Colab across multiple sessions, so cell execution order in the saved file isn't strictly linear. For full reproducibility, restart the runtime and run all cells top to bottom before relying on it as a clean end-to-end demo.
