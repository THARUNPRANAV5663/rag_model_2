# ════════════════════════════════════════════════════════════════════════════
# Retriva — Production Conversational Chatbot
# Multi-source: PDF, Scanned PDF (OCR), CSV, TSV, Excel, URL
# Stack: PyMuPDF, Tesseract, LangChain, ChromaDB, BM25,
#        bge-base-en-v1.5, bge-reranker-large, Llama3 (Groq), Streamlit
# Built by Tharun Pranav K S
# ════════════════════════════════════════════════════════════════════════════

import os
import io
import hashlib
from collections import deque

import numpy as np
import pandas as pd

import fitz
import pytesseract
from PIL import Image

from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer, CrossEncoder

import chromadb
from rank_bm25 import BM25Okapi
from groq import Groq

import streamlit as st

os.environ["USER_AGENT"] = "Retriva/1.0"


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
# SECTION 2 — MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-large")
    return embedder, reranker

def get_collection():
    """Always returns a valid collection — stored in session_state so it survives reruns."""
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = chromadb.EphemeralClient()
    client = st.session_state.chroma_client
    try:
        collection = client.get_or_create_collection(name="retriva")
    except Exception:
        # Client died — recreate everything
        st.session_state.chroma_client = chromadb.EphemeralClient()
        collection = st.session_state.chroma_client.get_or_create_collection(name="retriva")
    return st.session_state.chroma_client, collection

embedder, reranker = load_models()


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


MAX_ROWS_FULL    = 500    # embed all rows below this
MAX_ROWS_SAMPLE  = 3000  # sample cap for very large files
SAMPLE_THRESHOLD = 500   # trigger sampling above this

def _df_to_chunks(df, file_path, sheet_name):
    chunks = []
    df = df.fillna("unknown")
    n  = len(df)

    if n > SAMPLE_THRESHOLD:
        # Smart sampling: always keep first + last 100 rows for structure,
        # random sample the middle for coverage
        head = df.iloc[:100]
        tail = df.iloc[-100:]
        mid  = df.iloc[100:-100]
        sample_n = min(MAX_ROWS_SAMPLE - 200, len(mid))
        if sample_n > 0:
            mid = mid.sample(n=sample_n, random_state=42)
        df_sampled = pd.concat([head, mid, tail]).drop_duplicates()
        st.info(f"📊 Large file detected ({n:,} rows) — sampled {len(df_sampled):,} representative rows for fast embedding.")
    else:
        df_sampled = df

    for i, row in df_sampled.iterrows():
        sentence = ", ".join([f"{col} is {val}" for col, val in row.items()])
        sentence = f"Row {i+1}: {sentence}."
        chunks.append({
            "text":   sentence,
            "source": os.path.basename(file_path),
            "sheet":  sheet_name,
            "rows":   str(i + 1)
        })
    return chunks

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
        chunks.extend(_df_to_chunks(df, file_path, sheet_name))
    return chunks


def load_url(url):
    try:
        loader = WebBaseLoader(url)
        docs   = loader.load()
        raw    = " ".join([d.page_content for d in docs])
        soup   = BeautifulSoup(raw, "html.parser")
        for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()
        clean_text = soup.get_text(separator=" ", strip=True)
        return [{"text": clean_text, "source": url}]
    except Exception as e:
        st.error(f"URL load failed: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SECURITY
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
        "ignore previous instructions", "ignore all instructions",
        "you are now", "forget everything", "act as",
        "jailbreak", "disregard", "override", "system prompt",
    ]
    query_lower = query.lower()
    for pattern in patterns:
        if pattern in query_lower:
            raise ValueError("Blocked: potential prompt injection detected.")


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
        st.info("File already processed — skipping reembedding.")
        return
    if not chunks:
        return
    # Always get fresh collection reference to avoid stale cache issues
    _, fresh_collection = get_collection()
    texts     = [c["text"] for c in chunks]
    ids       = [hashlib.md5(c["text"].encode()).hexdigest() for c in chunks]
    metadatas = [{
        "source": str(c.get("source", "unknown")),
        "page":   str(c.get("page")  or ""),
        "sheet":  str(c.get("sheet") or ""),
        "rows":   str(c.get("rows")  or ""),
        "tokens": int(c.get("tokens", 0)),
    } for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    fresh_collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    bm25_chunks += chunks
    tokenized   = [c["text"].lower().split() for c in bm25_chunks]
    bm25_index  = BM25Okapi(tokenized)
    if file_hash:
        processed_hashes.add(file_hash)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RETRIEVER
# ════════════════════════════════════════════════════════════════════════════

def vector_search(query, top_k=10):
    _, fresh_collection = get_collection()
    query_embedding = embedder.encode([query]).tolist()
    results = fresh_collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, fresh_collection.count())
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
    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

def retrieve(query, top_k=4):
    _, fresh_collection = get_collection()
    if fresh_collection.count() == 0:
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
# SECTION 9 — FAQ HANDLER
# Catches identity, privacy, capability questions before RAG pipeline
# ════════════════════════════════════════════════════════════════════════════

# Order matters — specific phrases checked before broad single words
FAQ_TRIGGERS = [
    {
        "category": "identity",
        "keywords": [
            "who are you", "who r u", "who ru", "who are u", "whos this",
            "who's this", "who built you", "who made you", "who created you",
            "who developed you", "introduce yourself", "tell me about yourself",
            "what are you", "what is this", "what is retriva", "what r u",
            "are you chatgpt", "are you gpt", "are you ai", "are you a bot",
            "are you human", "are you real", "what kind of ai", "which ai",
            "what model are you", "what llm", "who is kstp", "what is kstp",
            "tell me about you", "about you", "about yourself",
            "are you a rag", "complete rag", "pretrained", "pre trained",
            "do you have knowledge", "general knowledge",
        ],
        "answer": (
            "Hey! I'm **Retriva** 👋 — a smart document chatbot built by **Tharun Pranav K S**.\n\n"
            "I'm not ChatGPT or any OpenAI product. I'm powered by **Llama3 (via Groq)** "
            "with hybrid retrieval (semantic + keyword search) and a reranker under the hood.\n\n"
            "In **RAG mode** I answer strictly from your uploaded documents. "
            "Switch to **Groq AI mode** (top-right toggle) to chat using Llama3's general knowledge.\n\n"
            "Upload a PDF, Excel, CSV, or paste a URL — and I'll answer anything from it. Let's go! 🚀"
        )
    },
    {
        "category": "capability",
        "keywords": [
            "what can you do", "what do you support", "file types", "what files",
            "can you read pdf", "can you read excel", "supported formats",
            "how does this work", "how do you work", "what's your purpose",
            "what is your purpose", "how to use", "get started", "help",
        ],
        "answer": (
            "Here's what I can do 💡\n\n"
            "📄 **PDFs** — digital & scanned (OCR supported)\n"
            "📊 **Excel** — .xlsx, .xls (multi-sheet)\n"
            "📋 **CSV / TSV** — tabular data\n"
            "🌐 **URLs** — scrape and answer from any webpage\n\n"
            "🔀 **Two modes** — RAG (document-only) or Groq AI (general knowledge) via the top-right toggle.\n\n"
            "Just upload your file or paste a URL, hit **⚡ Process Sources**, and ask me anything!"
        )
    },
    {
        "category": "privacy",
        "keywords": [
            "store my data", "store my pdf", "save my data", "send my data",
            "is my pdf safe", "data safe", "uploaded anywhere", "data privacy",
            "do you collect", "do you save", "is it safe", "safe to upload",
            "where does my data go", "who sees my data", "confidential",
            "privacy", "secure", "trust you",
        ],
        "answer": (
            "Your data is **completely safe** 🔒\n\n"
            "Everything you upload is processed **in-memory only** during your session. "
            "Nothing is saved to disk, sent to any external server, or stored anywhere. "
            "Once you close the session, it's all gone. You can safely upload confidential documents!"
        )
    },
    {
        "category": "memory",
        "keywords": [
            "remember previous", "remember chats", "chat history",
            "previous conversations", "do you remember", "your memory",
        ],
        "answer": (
            "I remember the last **3 messages** in our current conversation for context. "
            "I don't retain any memory across sessions — each session starts fresh."
        )
    },
    {
        "category": "confused",
        "keywords": [
            "i don't understand", "i dont understand", "confused",
            "what do you mean", "can you explain", "explain that",
            "how does that work", "lost", "pardon",
        ],
        "answer": (
            "No worries! 😊 Let me break it down:\n\n"
            "1. **Upload** a PDF, Excel, CSV, or paste a URL in the sidebar\n"
            "2. Hit **⚡ Process Sources** to let me read it\n"
            "3. **Ask me anything** about the document in the chat\n\n"
            "I'll find the answer and tell you exactly which page or row it came from!"
        )
    },
    {
        "category": "thanks",
        "keywords": [
            "thank you", "thanks", "thankyou", "thx", "ty", "thank u",
            "appreciate it", "appreciated", "cheers", "nice one", "good job",
            "well done", "great job", "you are great", "ur great", "you're great",
            "you are good", "you're good", "ur good", "retriva is great",
            "retriva is good", "awesome", "perfect", "great answer",
            "good answer", "well answered",
        ],
        "answer": (
            "Thank you, that means a lot! 😊\n\n"
            "Got more questions? Just ask — I'm here!"
        )
    },
    {
        "category": "farewell",
        "keywords": [
            "bye", "goodbye", "see you", "see ya", "cya", "later",
            "take care", "gotta go", "ttyl", "talk later", "good bye",
        ],
        "answer": (
            "Goodbye! 👋 Come back anytime with more documents to explore.\n\n"
            "Have a great day! 🌟"
        )
    },
    {
        "category": "howru",
        "keywords": [
            "how are you", "how r u", "how are u", "how r you",
            "how's it going", "hows it going", "how do you do",
            "you doing", "you okay", "you good", "all good",
        ],
        "answer": (
            "I'm doing great, thanks for asking! 😄\n\n"
            "Ready to dive into your documents. Upload a file or ask me anything!"
        )
    },
    {
        "category": "greeting",
        "keywords": [
            "hello", "hi", "hey", "heyy", "heyyy", "hii", "hiii",
            "good morning", "good afternoon", "good evening", "good night",
            "howdy", "sup", "what's up", "whats up", "wassup", "yo",
            "greetings", "namaste", "vanakkam", "hola",
        ],
        "answer": (
            "Hey there! 👋 Welcome to **Retriva**!\n\n"
            "I'm your smart document assistant. Upload a PDF, Excel, CSV, or paste a URL — "
            "and I'll answer anything from it instantly.\n\n"
            "Want to get started? Try uploading a file from the sidebar! 🚀"
        )
    },
]

def check_faq(query):
    """Returns (category, answer) if query matches any trigger, else (None, None)."""
    q = query.lower().strip()
    for faq in FAQ_TRIGGERS:
        if any(keyword in q for keyword in faq["keywords"]):
            return faq["category"], faq["answer"]
    return None, None


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CHAIN
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

def chat_groq_ai(query, memory):
    """Groq AI mode — answers from Llama3 general knowledge, no document context."""
    # Only flag as doc-specific if query contains SPECIFIC doc phrases, not just "data"
    doc_hints = [
        "my csv", "my pdf", "my excel", "my file", "my document", "my sheet",
        "uploaded file", "uploaded csv", "uploaded pdf", "the csv", "the pdf",
        "the excel", "the document", "the file", "in the sheet", "the table",
        "how many rows", "how many columns", "column names", "my data",
    ]
    q = query.lower()
    if any(hint in q for hint in doc_hints):
        return (
            "📄 That looks like a document-specific question!\n\n"
            "Switch to **RAG mode** using the toggle at the top-right, "
            "upload your file, and I'll answer it from your data."
        ), []

    system_prompt = """You are Retriva — a smart AI assistant powered by Llama3 (via Groq).
You are currently in Groq AI mode — answer from your general knowledge.
Be conversational, helpful, and concise. If you don't know something, say so honestly.
Note: your training data has a cutoff of early 2024 — for very recent events, mention this limitation."""

    messages  = [{"role": "system", "content": system_prompt}]
    messages += memory
    messages += [{"role": "user", "content": query}]

    try:
        response = groq_client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = messages,
            max_tokens  = 512,
            temperature = 0.5
        )
        return response.choices[0].message.content.strip(), []
    except Exception as e:
        return f"LLM error: {e}", []


def chat(query, memory, mode="rag"):

    # ── Groq AI mode — bypass RAG entirely ───────────────────────────────────
    if mode == "groq":
        # Only intercept identity/privacy/capability/memory FAQs in Groq mode
        # Greetings, howru, thanks, farewell → let Groq answer naturally
        GROQ_FAQ_CATEGORIES = {"identity", "privacy", "capability", "memory", "confused"}
        category, faq_answer = check_faq(query)
        if category in GROQ_FAQ_CATEGORIES and faq_answer:
            return faq_answer, []
        return chat_groq_ai(query, memory)

    # ── FAQ check — before RAG pipeline ───────────────────────────────────────
    _, faq_answer = check_faq(query)
    if faq_answer:
        return faq_answer, []

    # ── Security ──────────────────────────────────────────────────────────────
    try:
        check_prompt_injection(query)
    except ValueError as e:
        return str(e), []

    # ── No data check ─────────────────────────────────────────────────────────
    _, fresh_collection = get_collection()
    if fresh_collection.count() == 0:
        # Check if user had files loaded before — means session expired
        if st.session_state.get("files_loaded"):
            return (
                "⏰ **Session expired** — your documents were cleared due to inactivity.\n\n"
                "Streamlit Cloud sleeps after a few minutes of inactivity and resets all in-memory data.\n\n"
                "Please re-upload your files and hit **⚡ Process Sources** to continue. Sorry for the inconvenience!"
            ), []
        return (
            "No documents loaded yet! Please upload a PDF, Excel, CSV, or enter a URL first.\n\n"
            "Not sure how to start? Ask me **'how to use'** 😊"
        ), []

    # ── Rewrite ───────────────────────────────────────────────────────────────
    rewritten = rewrite_query(query, chat_history=memory)

    # ── Retrieve ──────────────────────────────────────────────────────────────
    chunks = retrieve(rewritten, top_k=4)
    if not chunks:
        return "I couldn't find relevant information in the provided documents.", []

    # ── Compress ──────────────────────────────────────────────────────────────
    chunks = compress_context(chunks, max_tokens=1500)

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_key = hashlib.md5(
        (rewritten + "".join([c["text"][:50] for c in chunks])).encode()
    ).hexdigest()
    if cache_key in query_cache:
        return query_cache[cache_key], chunks

    # ── Context ───────────────────────────────────────────────────────────────
    context = "\n\n".join([
        f"[Source: {c['metadata']['source']} | "
        f"Page: {c['metadata']['page']} | "
        f"Chunk: {c['metadata'].get('chunk_index', '')}]\n{c['text']}"
        for c in chunks
    ])

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt = """You are Retriva — a smart document chatbot built by Tharun Pranav K S.
Answer questions based ONLY on the provided document context.
If the answer is not found in the context, say: "I couldn't find that information in the provided documents."
Be conversational, clear and concise.
Always mention the source and page number in your answer."""

    messages  = [{"role": "system", "content": system_prompt}]
    messages += memory
    messages += [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {rewritten}"}]

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
# SECTION 11 — STREAMLIT UI
# ════════════════════════════════════════════════════════════════════════════

def reset_all():
    global bm25_index, bm25_chunks, processed_hashes
    st.session_state.chat_history = []
    st.session_state.memory       = []
    st.session_state.files_loaded = []
    query_cache.clear()
    bm25_index       = None
    bm25_chunks      = []
    processed_hashes = set()
    try:
        if "chroma_client" in st.session_state:
            st.session_state.chroma_client.delete_collection("retriva")
    except:
        pass
    st.session_state.pop("chroma_client", None)


def main():

    st.set_page_config(
        page_title = "Retriva",
        page_icon  = "🔍",
        layout     = "wide"
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero title ── */
    .retriva-hero {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 8px 0 4px 0;
    }
    .retriva-logo {
        font-size: 2.6rem;
        line-height: 1;
    }
    .retriva-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
    }
    .retriva-sub {
        font-size: 0.85rem;
        color: #888;
        margin-top: 2px;
    }

    /* ── Sticky mode toggle ── */
    .mode-toggle-wrap {
        position: fixed;
        top: 14px;
        right: 20px;
        z-index: 9999;
        background: #1a1a2e;
        border: 1px solid #2a2a4e;
        border-radius: 999px;
        padding: 5px 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #a0a0c0;
        box-shadow: 0 2px 12px rgba(108,99,255,0.15);
    }
    .mode-rag  { color: #48CAE4; }
    .mode-groq { color: #6C63FF; }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        border-radius: 14px !important;
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0f0f1a !important;
        border-right: 1px solid #1e1e2e;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #a0a0c0 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Process button ── */
    div[data-testid="stButton"] button[kind="secondary"] {
        background: linear-gradient(135deg, #6C63FF, #48CAE4) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 1px solid #2a2a3e !important;
        background: #0f0f1a !important;
    }

    /* ── Divider ── */
    hr { border-color: #1e1e2e !important; }
    </style>

    <div class="retriva-hero">
        <svg width="52" height="52" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0">
          <defs>
            <linearGradient id="gstar" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#6C63FF"/>
              <stop offset="60%" stop-color="#7B9FE0"/>
              <stop offset="100%" stop-color="#48CAE4"/>
            </linearGradient>
            <linearGradient id="gdark" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#4A44B0"/>
              <stop offset="100%" stop-color="#2A8AAF"/>
            </linearGradient>
          </defs>
          <g transform="translate(50,50)">
            <polygon points="0,-44 10,-10 44,0 10,10 0,44 -10,10 -44,0 -10,-10" fill="url(#gstar)"/>
            <polygon points="0,-20 14,0 0,20 -14,0" fill="url(#gdark)" opacity="0.6"/>
            <polygon points="0,-10 10,0 0,10 -10,0" fill="url(#gstar)" opacity="0.9"/>
            <circle cx="0" cy="0" r="48" fill="none" stroke="url(#gstar)" stroke-width="1" stroke-dasharray="4 4" opacity="0.3"/>
          </g>
        </svg>
        <div>
            <div class="retriva-title">Retriva</div>
            <div class="retriva-sub">Smart document chatbot — PDF, Excel, CSV, URL &nbsp;|&nbsp; Built by Tharun Pranav K S</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = []
    if "files_loaded" not in st.session_state:
        st.session_state.files_loaded = []
    if "mode" not in st.session_state:
        st.session_state.mode = "rag"

    # ── Sticky mode toggle (top-right) ────────────────────────────────────────
    col_spacer, col_toggle = st.columns([6, 1])
    with col_toggle:
        mode_label = "🗂️ RAG" if st.session_state.mode == "rag" else "🤖 Groq AI"
        if st.button(mode_label, help="Toggle between RAG (document) mode and Groq AI (general knowledge) mode"):
            st.session_state.mode = "groq" if st.session_state.mode == "rag" else "rag"
            st.rerun()

    # ── Mode indicator banner ─────────────────────────────────────────────────
    if st.session_state.mode == "groq":
        st.info("🤖 **Groq AI mode** — answering from Llama3 general knowledge. Switch to **RAG mode** to query your documents.")

    with st.sidebar:
        st.header("📁 Upload Sources")

        uploaded_files = st.file_uploader(
            "Upload PDF, CSV, TSV, Excel",
            type    = ["pdf", "csv", "tsv", "xlsx", "xls"],
            accept_multiple_files = True
        )

        url_input = st.text_input("Or enter a URL:", placeholder="https://...")

        if st.button("⚡ Process Sources", use_container_width=True):

            for uploaded_file in uploaded_files:
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                try:
                    check_file_size(temp_path)
                    file_hash = get_file_hash(temp_path)
                    if file_hash in processed_hashes:
                        st.info(f"⏭️ {uploaded_file.name} already processed — skipping.")
                        continue
                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    with st.spinner(f"Loading {uploaded_file.name}..."):
                        if ext == ".pdf":
                            docs = load_pdf(temp_path)
                        elif ext in [".csv", ".tsv", ".xlsx", ".xls"]:
                            docs = load_tabular(temp_path)
                        else:
                            docs = []
                    if docs:
                        with st.spinner(f"Embedding {uploaded_file.name}..."):
                            chunks = chunk_documents(docs)
                            embed_and_store(chunks, file_hash=file_hash)
                        if uploaded_file.name not in st.session_state.files_loaded:
                            st.session_state.files_loaded.append(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} — {len(chunks)} chunks embedded!")
                except ValueError as e:
                    st.error(str(e))

            if url_input:
                try:
                    check_url(url_input)
                    url_hash = hashlib.md5(url_input.encode()).hexdigest()
                    if url_hash in processed_hashes:
                        st.info(f"⏭️ URL already processed — skipping.")
                    else:
                        with st.spinner(f"Scraping {url_input}..."):
                            docs = load_url(url_input)
                        if docs:
                            with st.spinner("Embedding URL content..."):
                                chunks = chunk_documents(docs)
                                embed_and_store(chunks, file_hash=url_hash)
                            if url_input not in st.session_state.files_loaded:
                                st.session_state.files_loaded.append(url_input)
                            st.success(f"✅ URL — {len(chunks)} chunks embedded!")
                except ValueError as e:
                    st.error(str(e))

        if st.session_state.files_loaded:
            st.divider()
            st.subheader("📚 Loaded Sources")
            for src in st.session_state.files_loaded:
                st.write(f"• {src}")

        if st.button("🗑️ Clear All", use_container_width=True):
            reset_all()
            st.rerun()

    st.divider()

    # ── Session expired banner ────────────────────────────────────────────────
    _, _col = get_collection()
    if st.session_state.get("files_loaded") and _col.count() == 0:
        st.warning("⏰ **Session expired** — please re-upload your files and hit ⚡ Process Sources to continue.")

    for msg in st.session_state.chat_history:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🔍"
        with st.chat_message(msg["role"], avatar=avatar):
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
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(user_input)

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant", avatar="🔍"):
            with st.spinner("Thinking..."):
                answer, sources = chat(user_input, st.session_state.memory, mode=st.session_state.mode)
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