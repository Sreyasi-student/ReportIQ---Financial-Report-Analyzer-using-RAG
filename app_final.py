import os
import streamlit as st
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(page_title="ReportIQ: American Express Report Analyser", layout="wide")
st.title("ReportIQ: American Express Report Analyser")

# ---------------------------------------------------------------------------
# 1. Get the PDF (hardcoded path — must sit next to this script, or set the
#    full absolute path below)
# ---------------------------------------------------------------------------
PDF_PATH = r"C:\Users\Sreyasi Dey\Desktop\Rag Project\Amex_Report.pdf"

if not os.path.exists(PDF_PATH):
    st.error(f"PDF not found at: {PDF_PATH}\nMake sure the file is at that exact path.")
    st.stop()

with open(PDF_PATH, "rb") as f:
    pdf_bytes = f.read()
pdf_source = PDF_PATH


# ---------------------------------------------------------------------------
# 2. Extract text page by page
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def extract_pages(pdf_bytes: bytes):
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


# ---------------------------------------------------------------------------
# 3. Build the vector store (Ollama embeddings + FAISS)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_retriever(pdf_bytes: bytes):
    pages = extract_pages(pdf_bytes)
    if not pages:
        return None, "No extractable text found in this PDF (it may be scanned/image-only)."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = []
    for p in pages:
        for chunk in splitter.split_text(p["text"]):
            docs.append(Document(page_content=chunk, metadata={"page": p["page"]}))

    if not docs:
        return None, "Could not split the extracted text into chunks."

    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        db = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        return None, (
            "Could not reach Ollama for embeddings. Make sure Ollama is installed and running "
            f"('ollama serve') and that you've pulled the model with 'ollama pull nomic-embed-text'.\n\nDetails: {e}"
        )

    return db.as_retriever(search_kwargs={"k": 5}), None


with st.spinner("Reading PDF and building the search index..."):
    retriever, build_error = build_retriever(pdf_bytes)

if build_error:
    st.error(build_error)
    st.stop()

st.success(f"Loaded and indexed: {pdf_source}")

# ---------------------------------------------------------------------------
# 4. Query UI
# ---------------------------------------------------------------------------
query = st.text_input("Enter your question (e.g., 'Give a summary of the report')")

if query:
    with st.spinner("Finding relevant sections and generating an answer..."):
        try:
            docs = retriever.invoke(query)

            if not docs:
                st.warning("No relevant sections found.")
            else:
                context_text = ""
                for d in docs:
                    context_text += f"\nPage {d.metadata.get('page', 'N/A')}:\n{d.page_content}\n---\n"

                system_prompt = (
                    "You are a helpful assistant analyzing a business report.\n"
                    "From the provided context, answer the user's question clearly and concisely.\n"
                    "Use only information from the context provided.\n"
                    "If the context doesn't contain relevant information, say you don't have enough information to answer.\n\n"
                    f"Context:\n{context_text}"
                )

                llm = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=query),
                ])

                st.markdown("### 📊 Analysis Results")
                st.markdown(response.content)

                with st.expander("📄 View Source Pages"):
                    for d in docs:
                        st.markdown(f"**Page {d.metadata.get('page', 'N/A')}**")
                        st.write(d.page_content)
                        st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info(
                "Make sure Ollama is running ('ollama serve') and both models are pulled: "
                "'ollama pull llama3.2:1b' and 'ollama pull nomic-embed-text'."
            )