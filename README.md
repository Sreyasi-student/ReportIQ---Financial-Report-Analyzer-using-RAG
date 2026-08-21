# ReportIQ — Financial Report Analyzer using RAG

A RAG-based application to query and summarize American Express annual reports using natural language. Built with LangChain, FAISS, Ollama, and Streamlit.

## Features

- Loads and parses PDF financial reports
- Chunks text and generates embeddings using Ollama
- Retrieves relevant sections via FAISS vector search
- Answers queries using a local Llama model
- Interactive Streamlit UI with source page references

## Prerequisites

- Python 3.9+
- Ollama installed and running, with models pulled:
  ```bash
  ollama pull llama3.2:1b
  ollama pull nomic-embed-text
  ```
- `Amex_Report.pdf` placed in the project root

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Dependencies

```
streamlit
pandas
langchain-community
langchain-ollama
langchain-text-splitters
langchain-core
pypdf
```
