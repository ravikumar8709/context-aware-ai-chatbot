import streamlit as st
import requests
from bs4 import BeautifulSoup
import tempfile
import os
import base64
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync

import speech_recognition as sr
from gtts import gTTS

# ==============================
# 🔑 GROQ API KEY
# ==============================
GROQ_API_KEY = ""

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
# 🛠️ SHARED HTML PARSER
# ==============================
def parse_html(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "sup", "nav", "footer",
                      "header", "noscript", "iframe", "ads"]):
        tag.extract()

    texts = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6",
                               "li", "div", "section", "article",
                               "span", "td", "th", "blockquote"]):
        text = tag.get_text(" ", strip=True)
        if len(text) > 40:
            texts.append(text)

    seen = set()
    unique_texts = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)

    return "\n".join(unique_texts)


# ==============================
# 🌐 METHOD 1: Requests (Fast)
# ==============================
def load_website_requests(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return ""
        return parse_html(response.text)
    except Exception as e:
        st.sidebar.warning(f"Requests error: {str(e)}")
        return ""


# ==============================
# 🌐 METHOD 2: Playwright Stealth
# ==============================
def load_website_playwright(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                ]
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800},
                java_script_enabled=True,
                locale="en-US",
                timezone_id="America/New_York",
            )
            page = context.new_page()
            stealth_sync(page)
            page.goto(url, timeout=30000, wait_until="networkidle")
            page.wait_for_timeout(4000)
            page.evaluate("""
                async () => {
                    await new Promise(resolve => {
                        let totalHeight = 0;
                        const distance = 300;
                        const timer = setInterval(() => {
                            window.scrollBy(0, distance);
                            totalHeight += distance;
                            if (totalHeight >= document.body.scrollHeight) {
                                clearInterval(timer);
                                resolve();
                            }
                        }, 200);
                    });
                }
            """)
            page.wait_for_timeout(2000)
            html = page.content()
            browser.close()
            return parse_html(html)
    except Exception as e:
        st.sidebar.error(f"Playwright error: {str(e)}")
        return ""


# ==============================
# 🌐 METHOD 3: Playwright Visible
# ==============================
def load_website_playwright_visible(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()
            stealth_sync(page)
            page.goto(url, timeout=30000, wait_until="networkidle")
            page.wait_for_timeout(5000)
            page.evaluate("""
                async () => {
                    await new Promise(resolve => {
                        let totalHeight = 0;
                        const distance = 300;
                        const timer = setInterval(() => {
                            window.scrollBy(0, distance);
                            totalHeight += distance;
                            if (totalHeight >= document.body.scrollHeight) {
                                clearInterval(timer);
                                resolve();
                            }
                        }, 200);
                    });
                }
            """)
            page.wait_for_timeout(2000)
            html = page.content()
            browser.close()
            return parse_html(html)
    except Exception as e:
        st.sidebar.error(f"Visible browser error: {str(e)}")
        return ""


# ==============================
# 🌐 SMART WEBSITE LOADER
# ==============================
def load_website(url):
    st.info("⏳ Stage 1: Trying fast loader (requests)...")
    text = load_website_requests(url)
    if len(text) >= 500:
        st.success("✅ Fast loader succeeded!")
        return text

    st.warning("⚠️ Stage 1 failed. Trying Playwright stealth (headless)...")
    text = load_website_playwright(url)
    if len(text) >= 500:
        st.success("✅ Playwright stealth loader succeeded!")
        return text

    st.warning("⚠️ Stage 2 failed. Trying visible browser (last resort)...")
    text = load_website_playwright_visible(url)
    if len(text) >= 500:
        st.success("✅ Visible browser loader succeeded!")
        return text

    return text


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
# 🔍 ASK QUESTION
# ==============================
def ask_question(db, query):
    docs = db.max_marginal_relevance_search(
        query,
        k=8,
        fetch_k=30,
        lambda_mult=0.6
    )

    if not docs:
        return "No relevant data found.", [], []

    context = ""
    max_chars = 4500
    used_docs = []

    for doc in docs:
        if len(context) + len(doc.page_content) <= max_chars:
            context += doc.page_content + "\n"
            used_docs.append(doc)

    prompt = f"""
You are a knowledgeable and helpful assistant.

Instructions:
- Answer clearly and concisely using ONLY the context below
- If the question is descriptive (e.g. "what is X"), give a full explanation
- If the question is factual (e.g. dates, numbers), give a direct answer
- If the answer is truly not in the context, say: "Not found in the provided data"
- Do NOT make up information

Context:
{context}

Question: {query}

Answer:
"""

    answer = llm.invoke(prompt).content
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in used_docs]))
    return answer, sources, used_docs


# ==============================
# 🎤 VOICE INPUT
# ==============================
def voice_input(language="en-IN"):
    """
    Record voice from microphone and convert to text
    language codes:
        English   -> en-IN
        Hindi     -> hi-IN
        Punjabi   -> pa-IN
        Bengali   -> bn-IN
        Tamil     -> ta-IN
        Telugu    -> te-IN
        Marathi   -> mr-IN
        Gujarati  -> gu-IN
        Kannada   -> kn-IN
        Malayalam -> ml-IN
        Urdu      -> ur-IN
        
    """
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=8, phrase_time_limit=15)

        st.info("⏳ Processing speech...")
        text = r.recognize_google(audio, language=language)
        return text

    except sr.WaitTimeoutError:
        st.warning("⏰ No speech detected. Please try again.")
        return ""
    except sr.UnknownValueError:
        st.warning("❓ Could not understand audio. Please speak clearly.")
        return ""
    except sr.RequestError as e:
        st.error(f"🌐 Speech recognition service error: {e}")
        return ""
    except Exception as e:
        st.error(f"Microphone error: {str(e)}")
        return ""


# ==============================
# 🔊 TEXT TO SPEECH
# ==============================
def text_to_speech(text, language="en"):
    """
    Convert text to speech and return autoplay HTML audio
    language codes:
            English   -> en-IN
            Hindi     -> hi-IN
            Punjabi   -> pa-IN
            Bengali   -> bn-IN
            Tamil     -> ta-IN
            Telugu    -> te-IN
            Marathi   -> mr-IN
            Gujarati  -> gu-IN
            Kannada   -> kn-IN
            Malayalam -> ml-IN
            Urdu      -> ur-IN
            
    """
    try:
        tts = gTTS(text=text, lang=language, slow=False)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts.save(f.name)
            audio_path = f.name

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        os.remove(audio_path)

        # Encode to base64 for inline HTML playback
        audio_b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio controls autoplay style="width:100%; margin-top:10px;">
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
        """
        return audio_html

    except Exception as e:
        st.error(f"TTS error: {str(e)}")
        return ""


# ==============================
# 🎨 STREAMLIT UI
# ==============================
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.title("AI Chatbot 🤖 (PDF + Website + Groq)")

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header("📥 Load Data")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    url = st.text_input("Enter Website URL", placeholder="https://example.com")
    load_btn = st.button("🔄 Load Data", use_container_width=True)

    st.markdown("---")

    # 🔊 Audio Settings
    st.header("🎵 Audio Settings")

    tts_enabled = st.toggle("🔊 Read answers aloud (TTS)", value=True)

    # Language mapping
    language_map = {
        "English": {"tts": "en", "stt": "en-IN"},
        "Hindi": {"tts": "hi", "stt": "hi-IN"},
        "Punjabi": {"tts": "pa", "stt": "pa-IN"},
        "Bengali": {"tts": "bn", "stt": "bn-IN"},
        "Tamil": {"tts": "ta", "stt": "ta-IN"},
        "Telugu": {"tts": "te", "stt": "te-IN"},
        "Marathi": {"tts": "mr", "stt": "mr-IN"},
        "Gujarati": {"tts": "gu", "stt": "gu-IN"},
        "Kannada": {"tts": "kn", "stt": "kn-IN"},
        "Malayalam": {"tts": "ml", "stt": "ml-IN"},
        "Urdu": {"tts": "ur", "stt": "ur-IN"}
    }

# Dropdown
    audio_language = st.selectbox(
        "🌐 Audio Language",
        options=list(language_map.keys()),
        index=0
    )

    # Get codes
    tts_lang = language_map[audio_language]["tts"]
    stt_lang = language_map[audio_language]["stt"]
    # Debug toggle
    if st.checkbox("Show raw scraped text (debug)"):
        st.session_state.show_debug = True
    else:
        st.session_state.show_debug = False

# ==============================
# SESSION STATE
# ==============================
if "docs" not in st.session_state:
    st.session_state.docs = []
if "db" not in st.session_state:
    st.session_state.db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "voice_query" not in st.session_state:
    st.session_state.voice_query = ""

# ==============================
# 🔘 LOAD DATA
# ==============================
if load_btn:
    st.session_state.docs = []
    st.session_state.chat_history = []

    with st.spinner("Loading data..."):

        if pdf_file:
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.read())
            pdf_docs = load_pdf("temp.pdf")
            st.session_state.docs.extend(pdf_docs)
            st.sidebar.success(f"✅ PDF loaded: {len(pdf_docs)} pages")

        if url:
            website_text = load_website(url)
            st.sidebar.write(f"📊 Website text length: **{len(website_text)}** chars")

            if st.session_state.get("show_debug") and website_text:
                with st.expander("🔍 Raw Scraped Text Preview"):
                    st.text(website_text[:3000])

            if len(website_text) < 100:
                st.sidebar.error("❌ Could not extract content. Site may be heavily protected.")
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150
                )
                chunks = splitter.split_text(website_text)
                for chunk in chunks:
                    st.session_state.docs.append(
                        Document(page_content=chunk, metadata={"source": url})
                    )
                st.sidebar.success(f"✅ Website loaded: {len(chunks)} chunks")

        if st.session_state.docs:
            st.session_state.db = create_vectorstore(st.session_state.docs)
            st.sidebar.success("✅ Vector store ready! Start chatting 👇")
        else:
            st.sidebar.warning("⚠️ No data loaded. Please upload a PDF or enter a URL.")

# ==============================
# 💬 CHAT INTERFACE
# ==============================
if st.session_state.db:

    st.markdown("---")
    st.subheader("💬 Chat")

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            with st.expander("📚 Sources"):
                for s in chat["sources"]:
                    st.write(f"- {s}")

    # ==============================
    # 🎤 VOICE INPUT BUTTON
    # ==============================
    col1, col2 = st.columns([1, 5])

    with col1:
        if st.button("🎤 Speak", use_container_width=True):
            spoken_text = voice_input(language=stt_lang)
            if spoken_text:
                st.session_state.voice_query = spoken_text
                st.success(f"✅ You said: **{spoken_text}**")

    with col2:
        st.caption(f"🌐 Voice language: **{audio_language}** | Click 'Speak' then ask your question")

    # ==============================
    # 💬 CHAT INPUT (text or voice)
    # ==============================
    # Use voice query if available, else wait for text input
    text_query = st.chat_input("Ask something about the loaded data...")

    # Determine final query (voice takes priority if just recorded)
    query = None
    if st.session_state.voice_query:
        query = st.session_state.voice_query
        st.session_state.voice_query = ""  # Clear after use
    elif text_query:
        query = text_query

    if query:
        with st.chat_message("user"):
            st.write(f"{'🎤 ' if not text_query else ''}{query}")

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources, used_docs = ask_question(st.session_state.db, query)

            st.write(answer)

            # 🔊 Text-to-Speech playback
            if tts_enabled and answer and answer != "Not found in the provided data":
                audio_html = text_to_speech(answer, language=tts_lang)
                if audio_html:
                    st.markdown(audio_html, unsafe_allow_html=True)

            with st.expander("📚 Sources"):
                for s in sources:
                    st.write(f"- {s}")

            with st.expander("🔍 Retrieved Chunks (Debug)"):
                for i, doc in enumerate(used_docs):
                    st.write(f"**Chunk {i+1}:**")
                    st.write(doc.page_content[:400])
                    st.divider()

        st.session_state.chat_history.append({
            "question": query,
            "answer": answer,
            "sources": sources
        })

else:
    st.info("👈 Please load a PDF or Website URL from the sidebar to start chatting.")