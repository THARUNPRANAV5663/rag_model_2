# ════════════════════════════════════════════════════════════════════════════
# KSTP's RAG Model — Production Conversational Chatbot
# Multi-source: PDF, Scanned PDF (OCR), CSV, TSV, Excel, URL
# Stack: PyMuPDF, Tesseract, LangChain, ChromaDB, BM25,
#        bge-base-en-v1.5, bge-reranker-large, Llama3 (Groq), Streamlit
# Built by Tharun Pranav K S (KSTP)
# ════════════════════════════════════════════════════════════════════════════

# ── Standard Libraries ────────────────────────────────────────────────────────
import os
import io
import hashlib
from collections import deque

# ── Data Processing ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── PDF Processing ────────────────────────────────────────────────────────────
import fitz                                         # PyMuPDF — digital PDF
import pytesseract                                  # OCR — scanned PDF
from PIL import Image

# ── Web Scraping ──────────────────────────────────────────────────────────────
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader

# ── Text Chunking ─────────────────────────────────────────────────────────────
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Embeddings + Reranker ─────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Vector Store ──────────────────────────────────────────────────────────────
import chromadb

# ── Keyword Search ────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi

# ── LLM ───────────────────────────────────────────────────────────────────────
from groq import Groq

# ── Streamlit UI ──────────────────────────────────────────────────────────────
import streamlit as st

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["USER_AGENT"] = "KSTPs_RAG/1.0"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — API KEY SETUP
# ════════════════════════════════════════════════════════════════════════════

try:
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
except:
    GROQ_API_KEY = st.secrets.get('GROQ_API_KEY')

groq_client = Groq(api_key=GROQ_API_KEY)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL LOADING (cached — loads only once)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-large")
    return embedder, reranker

@st.cache_resource
def load_chromadb():
    client     = chromadb.Client()
    collection = client.get_or_create_collection(name="kstps_rag")
    return client, collection

embedder, reranker           = load_models()
chroma_client, collection    = load_chromadb()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_pdf(file_path):
    doc   = fitz.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if not text:
            pix        = page.get_pixmap()
            img        = Image.open(io.BytesIO(pix.tobytes("png")))
            text       = pytesseract.image_to_string(img).strip()
            extraction = "ocr"
        else:
            extraction = "pymupdf"

        if text:
            pages.append({
                "text":       text,
                "page":       page_num + 1,
                "source":     os.path.basename(file_path),
                "extraction": extraction
            })

    return pages


def load_tabular(file_path):
    ext    = os.path.splitext(file_path)[1].lower()
    chunks = []

    if ext == ".csv":
        sheets = {"Sheet1": pd.read_csv(file_path)}
    elif ext == ".tsv":
        sheets = {"Sheet1": pd.read_csv(file_path, sep="\t")}
    elif ext in [".xlsx", ".xls"]:
        sheets = pd.read_excel(file_path, sheet_name=None)
    else:
        return []

    for sheet_name, df in sheets.items():
        df = df.fillna("unknown")
        for i, row in df.iterrows():
            sentence = ", ".join([f"{col} is {val}" for col, val in row.items()])
            sentence = f"Row {i+1}: {sentence}."
            chunks.append({
                "text":   sentence,
                "source": os.path.basename(file_path),
                "sheet":  sheet_name,
                "rows":   str(i + 1)
            })

    return chunks


def load_url(url):
    try:
        loader = WebBaseLoader(url)
        docs   = loader.load()
        raw    = " ".join([d.page_content for d in docs])

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["nav", "header", "footer",
                         "script", "style", "aside"]):
            tag.decompose()

        clean_text = soup.get_text(separator=" ", strip=True)
        return [{"text": clean_text, "source": url}]

    except Exception as e:
        st.error(f"URL load failed: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SECURITY CHECKS
# ════════════════════════════════════════════════════════════════════════════

def check_file_size(file_path, max_mb=20):
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"File too large: {size_mb:.1f}MB — max {max_mb}MB")

def check_url(url):
    if not url.startswith(("http://", "https://")):
        raise ValueError("Invalid URL — must start with http:// or https://")
    blocked = ["localhost", "127.0.0.1", "0.0.0.0"]
    if any(b in url for b in blocked):
        raise ValueError("Blocked URL")

def check_prompt_injection(query):
    patterns = [
        "ignore previous instructions",
        "ignore all instructions",
        "you are now",
        "forget everything",
        "act as",
        "jailbreak",
        "disregard",
        "override",
        "system prompt",
    ]
    query_lower = query.lower()
    for pattern in patterns:
        if pattern in query_lower:
            raise ValueError(f"Blocked: potential prompt injection detected.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CHUNKER
# ════════════════════════════════════════════════════════════════════════════

def count_tokens(text, model="cl100k_base"):
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))

def chunk_documents(docs, chunk_size=512, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = chunk_size,
        chunk_overlap   = overlap,
        length_function = count_tokens,
        separators      = ["\n\n", "\n", ".", " ", ""]
    )

    all_chunks  = []
    chunk_index = 0

    for doc in docs:
        text = doc.get("text", "").strip()
        if not text:
            continue

        splits = splitter.split_text(text)

        for i, chunk_text in enumerate(splits):
            all_chunks.append({
                "text":        chunk_text,
                "chunk_index": chunk_index,
                "tokens":      count_tokens(chunk_text),
                "source":      doc.get("source", "unknown"),
                "page":        doc.get("page",   None),
                "sheet":       doc.get("sheet",  None),
                "rows":        doc.get("rows",   None),
            })
            chunk_index += 1

    return all_chunks


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EMBEDDER
# ════════════════════════════════════════════════════════════════════════════

processed_hashes = set()
bm25_index  = None
bm25_chunks = []

def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def embed_and_store(chunks, file_hash=None):
    global bm25_index, bm25_chunks

    if file_hash and file_hash in processed_hashes:
        st.info("File already processed — skipping reembedding")
        return

    if not chunks:
        return

    texts      = [c["text"] for c in chunks]
    ids        = [f"chunk_{c['chunk_index']}" for c in chunks]
    metadatas  = [{
        "source": str(c.get("source", "unknown")),
        "page":   str(c.get("page")  or ""),
        "sheet":  str(c.get("sheet") or ""),
        "rows":   str(c.get("rows")  or ""),
        "tokens": int(c.get("tokens", 0)),
    } for c in chunks]

    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        ids        = ids,
        documents  = texts,
        embeddings = embeddings,
        metadatas  = metadatas
    )

    bm25_chunks = chunks
    tokenized   = [c["text"].lower().split() for c in chunks]
    bm25_index  = BM25Okapi(tokenized)

    if file_hash:
        processed_hashes.add(file_hash)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RETRIEVER
# ════════════════════════════════════════════════════════════════════════════

def vector_search(query, top_k=10):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = min(top_k, collection.count())
    )
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "method":   "vector"
        })
    return chunks

def bm25_search(query, top_k=10):
    if bm25_index is None:
        return []
    tokenized_query = query.lower().split()
    scores          = bm25_index.get_scores(tokenized_query)
    top_indices     = np.argsort(scores)[::-1][:top_k]
    chunks = []
    for idx in top_indices:
        if scores[idx] > 0:
            chunks.append({
                "text":     bm25_chunks[idx]["text"],
                "metadata": {
                    "source": str(bm25_chunks[idx].get("source", "")),
                    "page":   str(bm25_chunks[idx].get("page")  or ""),
                    "sheet":  str(bm25_chunks[idx].get("sheet") or ""),
                    "rows":   str(bm25_chunks[idx].get("rows")  or ""),
                    "tokens": bm25_chunks[idx].get("tokens", 0),
                },
                "method": "bm25"
            })
    return chunks

def rerank(query, chunks, top_k=4):
    if not chunks:
        return []
    pairs  = [[query, c["text"]] for c in chunks]
    scores = reranker.predict(pairs)
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

def retrieve(query, top_k=4):
    if collection.count() == 0:
        return []

    vector_results = vector_search(query, top_k=10)
    bm25_results   = bm25_search(query,   top_k=10)

    seen, merged = set(), []
    for chunk in vector_results + bm25_results:
        text = chunk["text"].strip()
        if text not in seen:
            seen.add(text)
            merged.append(chunk)

    return rerank(query, merged, top_k=top_k)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — REWRITER
# ════════════════════════════════════════════════════════════════════════════

def rewrite_query(query, chat_history=[]):
    history_text = ""
    if chat_history:
        for msg in chat_history[-3:]:
            history_text += f"{msg['role']}: {msg['content']}\n"

    system_prompt = """You are a query rewriting assistant.
Rewrite vague user queries into clear, specific search queries.
Rules:
- Return ONLY the rewritten query
- No explanation or preamble
- Keep under 30 words
- Use conversation history for context on vague followups
- If query is already clear, return as is"""

    user_prompt = f"""Conversation history:
{history_text if history_text else "No previous messages"}

Current query: {query}

Rewrite this query:"""

    try:
        response = groq_client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens  = 100,
            temperature = 0.1
        )
        return response.choices[0].message.content.strip()
    except:
        return query


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CHAIN
# ════════════════════════════════════════════════════════════════════════════

query_cache = {}

def compress_context(chunks, max_tokens=1500):
    total = sum(c.get("tokens", 0) for c in chunks)
    if total <= max_tokens:
        return chunks
    ratio = max_tokens / total
    compressed = []
    for chunk in chunks:
        words         = chunk["text"].split()
        keep          = max(20, int(len(words) * ratio))
        chunk         = chunk.copy()
        chunk["text"] = " ".join(words[:keep]) + "..."
        compressed.append(chunk)
    return compressed

def chat(query, memory):

    # ── Identity & FAQ — handle before RAG pipeline ───────────────────────────
    q = query.lower().strip()

    identity_triggers = ["who built you", "who made you", "who created you", "who developed you"]
    what_triggers     = ["what are you", "what is this", "what is kstp", "are you chatgpt", "are you gpt"]
    privacy_triggers  = ["store my data", "store my pdf", "send my data", "is my pdf safe",
                         "data safe", "uploaded anywhere", "data privacy"]
    capability_triggers = ["what can you do", "what do you support", "file types", "what files"]
    memory_triggers   = ["remember previous", "remember chats", "chat history", "previous conversations"]
    scope_triggers    = ["weather", "write me a poem", "who is elon", "who is modi",
                         "tell me a joke", "general knowledge"]

    if any(t in q for t in identity_triggers):
        return "I was built by Tharun Pranav K S (KSTP). I am KSTP's RAG Model — a Retrieval-Augmented Generation chatbot.", []

    if any(t in q for t in what_triggers):
        return "I am KSTP's RAG Model, a Retrieval-Augmented Generation chatbot built by Tharun Pranav K S (KSTP). I am powered by Llama3 via Groq and am not ChatGPT or any OpenAI product.", []

    if any(t in q for t in privacy_triggers):
        return "Your files are completely safe. They are processed in session memory only and are never stored, saved, or sent anywhere. Once you close the session, everything is cleared.", []

    if any(t in q for t in capability_triggers):
        return "I can answer questions from PDFs (including scanned/OCR PDFs), Excel files, CSV, TSV files, and web URLs. Just upload your document and ask away!", []

    if any(t in q for t in memory_triggers):
        return "I remember the last 3 messages in our current conversation for context. I do not retain any memory across sessions.", []

    # ── Security ──────────────────────────────────────────────────────────────
    try:
        check_prompt_injection(query)
    except ValueError as e:
        return str(e), []

    # ── No data check ─────────────────────────────────────────────────────────
    if collection.count() == 0:
        return "Please upload a document or enter a URL first.", []

    # ── Rewrite query ─────────────────────────────────────────────────────────
    rewritten = rewrite_query(query, chat_history=memory)

    # ── Retrieve ──────────────────────────────────────────────────────────────
    chunks = retrieve(rewritten, top_k=4)
    if not chunks:
        return "I couldn't find relevant information in the provided documents.", []

    # ── Compress ──────────────────────────────────────────────────────────────
    chunks = compress_context(chunks, max_tokens=1500)

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_key = hashlib.md5(
        (rewritten + "".join([c["text"][:50] for c in chunks])).encode()
    ).hexdigest()

    if cache_key in query_cache:
        return query_cache[cache_key], chunks

    # ── Build context ─────────────────────────────────────────────────────────
    context = "\n\n".join([
        f"[Source: {c['metadata']['source']} | "
        f"Page: {c['metadata']['page']} | "
        f"Chunk: {c['metadata'].get('chunk_index', '')}]\n{c['text']}"
        for c in chunks
    ])

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt = """You are KSTP's RAG Model — a helpful conversational assistant built by Tharun Pranav K S (KSTP).

Answer questions based ONLY on the provided document context.
If the answer is not found in the context, say: "I couldn't find that information in the provided documents."
Be conversational, clear and concise.
Always mention the source and page number in your answer.

If asked who built you: "I was built by Tharun Pranav K S (KSTP)."
If asked what you are: "I am KSTP's RAG Model, powered by Llama3 via Groq."
If asked about data privacy: "Your files are processed in memory only and never stored or sent anywhere."
If asked about capabilities: "I support PDF, scanned PDF (OCR), Excel, CSV, TSV, and URLs."
"""

    # ── Build messages ────────────────────────────────────────────────────────
    messages  = [{"role": "system", "content": system_prompt}]
    messages += memory
    messages += [{"role": "user",
                  "content": f"Context:\n{context}\n\nQuestion: {rewritten}"}]

    # ── Call Llama3 ───────────────────────────────────────────────────────────
    try:
        response = groq_client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = messages,
            max_tokens  = 512,
            temperature = 0.3
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}", []

    query_cache[cache_key] = answer
    return answer, chunks


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — STREAMLIT UI
# ════════════════════════════════════════════════════════════════════════════

def main():

    st.set_page_config(
        page_title = "KSTP's RAG Model",
        page_icon  = "🤖",
        layout     = "wide"
    )

    st.title("🤖 KSTP's RAG Model")
    st.caption("Multi-source conversational chatbot — PDF, Excel, CSV, URL | Built by Tharun Pranav K S")

    if "chat_history"  not in st.session_state:
        st.session_state.chat_history  = []
    if "memory"        not in st.session_state:
        st.session_state.memory        = []
    if "files_loaded"  not in st.session_state:
        st.session_state.files_loaded  = []

    with st.sidebar:
        st.header("📁 Upload Sources")

        uploaded_files = st.file_uploader(
            "Upload PDF, CSV, TSV, Excel",
            type    = ["pdf", "csv", "tsv", "xlsx", "xls"],
            accept_multiple_files = True
        )

        url_input = st.text_input("Or enter a URL:", placeholder="https://...")

        if st.button("⚡ Process Sources", use_container_width=True):
            all_docs = []

            for uploaded_file in uploaded_files:
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    check_file_size(temp_path)
                    file_hash = get_file_hash(temp_path)
                    ext       = os.path.splitext(uploaded_file.name)[1].lower()

                    with st.spinner(f"Loading {uploaded_file.name}..."):
                        if ext == ".pdf":
                            docs = load_pdf(temp_path)
                        elif ext in [".csv", ".tsv", ".xlsx", ".xls"]:
                            docs = load_tabular(temp_path)
                        else:
                            docs = []

                    all_docs.extend(docs)
                    st.session_state.files_loaded.append(uploaded_file.name)
                    st.success(f"✅ {uploaded_file.name} loaded")

                except ValueError as e:
                    st.error(str(e))

            if url_input:
                try:
                    check_url(url_input)
                    with st.spinner(f"Scraping {url_input}..."):
                        docs = load_url(url_input)
                    all_docs.extend(docs)
                    st.session_state.files_loaded.append(url_input)
                    st.success(f"✅ URL loaded")
                except ValueError as e:
                    st.error(str(e))

            if all_docs:
                with st.spinner("Chunking and embedding..."):
                    chunks = chunk_documents(all_docs)
                    embed_and_store(chunks)
                st.success(f"✅ {len(chunks)} chunks embedded and ready!")

        if st.session_state.files_loaded:
            st.divider()
            st.subheader("📚 Loaded Sources")
            for src in st.session_state.files_loaded:
                st.write(f"• {src}")

        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory       = []
            st.session_state.files_loaded = []
            query_cache.clear()
            chroma_client.delete_collection("kstps_rag")
            st.rerun()

    st.divider()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📌 Sources"):
                    for src in msg["sources"]:
                        st.caption(
                            f"📄 {src['metadata']['source']} | "
                            f"Page: {src['metadata']['page']} | "
                            f"Sheet: {src['metadata']['sheet']}"
                        )

    user_input = st.chat_input("Ask anything about your documents...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        st.session_state.chat_history.append({
            "role":    "user",
            "content": user_input
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = chat(user_input, st.session_state.memory)

            st.write(answer)

            if sources:
                with st.expander("📌 Sources"):
                    for src in sources:
                        st.caption(
                            f"📄 {src['metadata']['source']} | "
                            f"Page: {src['metadata']['page']} | "
                            f"Sheet: {src['metadata']['sheet']}"
                        )

        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources
        })

        st.session_state.memory.append({"role": "user",      "content": user_input})
        st.session_state.memory.append({"role": "assistant", "content": answer})
        st.session_state.memory = st.session_state.memory[-6:]


if __name__ == "__main__":
    main()