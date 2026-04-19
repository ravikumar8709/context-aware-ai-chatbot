import streamlit as st
import os
import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq

# ==============================
# 🔑 GROQ API KEY
# ==============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==============================
# 🤖 LLM (Groq)
# ==============================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)

# ==============================
# 🧠 EMBEDDINGS
# ==============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==============================
# 🌐 WEBSITE LOADER
# ==============================
def load_website(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "sup"]):
            tag.extract()

        texts = []

        # Infobox (important info like DOB, year)
        infobox = soup.find("table", {"class": "infobox"})
        if infobox:
            texts.append(infobox.get_text(" ", strip=True))

        # Paragraphs
        for p in soup.find_all("p"):
            text = p.get_text(" ", strip=True)
            if len(text) > 50:
                texts.append(text)

        return "\n".join(texts)

    except Exception:
        return ""

# ==============================
# 📄 PDF LOADER
# ==============================
def load_pdf(file):
    loader = PyPDFLoader(file)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = file

    return docs

# ==============================
# 🔄 VECTOR STORE
# ==============================
def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

# ==============================
# 🔍 ASK QUESTION (IMPROVED)
# ==============================
def ask_question(db, query):
    # 🔥 Query expansion
    enhanced_query = query + " year date number appointment details information"

    # 🔍 MMR search (better retrieval)
    docs = db.max_marginal_relevance_search(
        enhanced_query,
        k=6,
        fetch_k=20,
        lambda_mult=0.7
    )

    if not docs:
        return "No relevant data found.", [], []

    # 🔥 Limit context size (avoid token error)
    context = ""
    max_chars = 3200
    used_docs = []

    for doc in docs:
        if len(context) + len(doc.page_content) <= max_chars:
            context += doc.page_content + "\n"
            used_docs.append(doc)

    prompt = f"""
You are a precise QA assistant.

Answer ONLY using the context.

- Extract exact factual answer
- Focus on year/date/number
- Give short answer (1-2 lines)
- If not found, say: Not found in data

Context:
{context}

Question: {query}

Answer:
"""

    answer = llm.invoke(prompt).content

    sources = list(set([doc.metadata.get("source", "Unknown") for doc in used_docs]))

    return answer, sources, used_docs

# ==============================
# 🎨 STREAMLIT UI
# ==============================
st.title("AI Chatbot 🤖 (PDF + Website + Groq)")

pdf_file = st.file_uploader("Upload PDF", type="pdf")
url = st.text_input("Enter Website URL")

if "docs" not in st.session_state:
    st.session_state.docs = []

if "db" not in st.session_state:
    st.session_state.db = None

# ==============================
# 🔘 LOAD DATA
# ==============================
if st.button("Load Data"):
    st.session_state.docs = []

    # PDF
    if pdf_file:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())
        st.session_state.docs.extend(load_pdf("temp.pdf"))

    # Website
    if url:
        website_text = load_website(url)

        st.write("Website text length:", len(website_text))  # debug

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        chunks = splitter.split_text(website_text)

        for chunk in chunks:
            st.session_state.docs.append(
                Document(page_content=chunk, metadata={"source": url})
            )

    # Create vector DB
    if st.session_state.docs:
        st.session_state.db = create_vectorstore(st.session_state.docs)
        st.success("Data loaded successfully!")

# ==============================
# 💬 QUERY
# ==============================
if st.session_state.db:
    query = st.text_input("Ask something:")

    if query:
        answer, sources, used_docs = ask_question(st.session_state.db, query)

        st.write("### 🤖 Answer:")
        st.write(answer)

        st.write("### 📚 Sources:")
        for s in sources:
            st.write("-", s)

        # 🔍 Debug panel
        with st.expander("🔍 Retrieved Context (Debug)"):
            for i, doc in enumerate(used_docs):
                st.write(f"Chunk {i+1}:")
                st.write(doc.page_content[:400])
                st.write("-----")