import asyncio
import nest_asyncio
import streamlit as st
import os
from dotenv import load_dotenv
import time
from pathlib import Path
import json
from datetime import datetime
import requests


from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI


try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
nest_asyncio.apply()


load_dotenv()


st.set_page_config(
    page_title="Anmol's Genius Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### Genius Pro v3.0 | Enterprise AI Solution"
    }
)

# PROFESSIONAL CSS WITH GRADIENTS
st.markdown("""
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    /* Main Background - Animated Gradient */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #1a1a2e 50%, #16213e 75%, #0f3460 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
    }
    
    .stApp { background: transparent; }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Header Styling */
    .title {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Montserrat', 'Segoe UI', sans-serif;
        text-align: center;
        padding: 3rem 0;
        font-size: 4rem;
        font-weight: 900;
        letter-spacing: -1px;
        animation: fadeInDown 0.8s ease;
        text-shadow: 0 4px 30px rgba(96, 165, 250, 0.3);
    }
    
    .subtitle {
        text-align: center;
        background: linear-gradient(90deg, #e0e7ff 0%, #fce7f3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.3rem;
        margin-bottom: 3rem;
        animation: fadeInUp 0.8s ease 0.2s both;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Upload Box - Premium */
    .upload-box {
        border: 2px solid;
        border-image: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6) 1;
        border-radius: 25px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.05), rgba(167, 139, 250, 0.05));
        backdrop-filter: blur(20px);
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 20px 60px rgba(96, 165, 250, 0.1), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .upload-box::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(96, 165, 250, 0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    .upload-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 80px rgba(96, 165, 250, 0.2), inset 0 1px 1px rgba(255, 255, 255, 0.2);
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.1), rgba(167, 139, 250, 0.1));
    }
    
    /* Answer Box */
    .answer-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8));
        padding: 2.5rem;
        border-radius: 20px;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(96, 165, 250, 0.3);
        box-shadow: 0 15px 50px rgba(96, 165, 250, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        animation: slideUp 0.5s ease;
        color: white;
    }
    
    .answer-box h3 {
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Source Card */
    .source-card {
        background: linear-gradient(135deg, rgba(60, 80, 120, 0.3), rgba(40, 60, 100, 0.3));
        border-left: 4px solid;
        border-image: linear-gradient(180deg, #60a5fa, #a78bfa) 1;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        transition: all 0.4s;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(96, 165, 250, 0.1);
    }
    
    .source-card:hover {
        transform: translateX(10px) translateY(-2px);
        background: linear-gradient(135deg, rgba(60, 80, 120, 0.5), rgba(40, 60, 100, 0.5));
        box-shadow: 0 15px 40px rgba(96, 165, 250, 0.2);
    }
    
    /* Buttons - Premium */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899) !important;
        background-size: 200% 200%;
        animation: gradientFlow 3s ease infinite;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 0.9rem 2.5rem !important;
        border-radius: 50px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.6) !important;
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Input Fields */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.1), rgba(167, 139, 250, 0.05)) !important;
        border: 1.5px solid rgba(96, 165, 250, 0.3) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.9rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s !important;
    }
    
    .stTextInput>div>div>input::placeholder,
    .stTextArea>div>div>textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: rgba(96, 165, 250, 0.8) !important;
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.3) !important;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.15), rgba(167, 139, 250, 0.1)) !important;
    }
    
    /* Modal Styles - Premium */
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.7), rgba(30, 41, 59, 0.7));
        backdrop-filter: blur(15px);
        z-index: 999;
        animation: fadeIn 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .modal-content {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1a1a2e 100%);
        border-radius: 30px;
        padding: 3rem;
        max-width: 650px;
        width: 95%;
        max-height: 90vh;
        overflow-y: auto;
        box-shadow: 0 25px 80px rgba(59, 130, 246, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        border: 1.5px solid rgba(96, 165, 250, 0.3);
        animation: slideInUp 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 2px solid rgba(96, 165, 250, 0.3);
    }
    
    .modal-title {
        font-size: 2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    
    .modal-close {
        background: rgba(96, 165, 250, 0.2);
        border: 1px solid rgba(96, 165, 250, 0.3);
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
        width: 45px;
        height: 45px;
        border-radius: 50%;
        transition: all 0.3s;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    
    .modal-close:hover {
        background: rgba(96, 165, 250, 0.4);
        transform: rotate(90deg);
        border-color: rgba(96, 165, 250, 0.6);
    }
    
    /* Contact Button */
    .contact-btn-awesome {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #ff7e5f 100%) !important;
        background-size: 200% 200%;
        animation: gradientFlow 3s ease infinite;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 1rem 2.5rem !important;
        border-radius: 50px !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4) !important;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        width: 100%;
    }
    
    .contact-btn-awesome:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 15px 45px rgba(245, 87, 108, 0.6) !important;
    }
    
    /* Success Badge */
    .success-badge {
        background: linear-gradient(135deg, #34d399, #10b981);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 1rem;
        animation: pulse 2s infinite;
        box-shadow: 0 10px 30px rgba(52, 211, 153, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, rgba(96, 165, 250, 0.05), rgba(167, 139, 250, 0.05));
        border-bottom: 2px solid rgba(96, 165, 250, 0.2);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: rgba(255, 255, 255, 0.7);
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(139, 92, 246, 0.3)) !important;
        color: white !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(180deg, transparent, rgba(96, 165, 250, 0.05));
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.95rem;
        margin-top: 5rem;
        border-top: 1px solid rgba(96, 165, 250, 0.2);
    }
    
    .footer p {
        margin: 0.5rem 0;
    }
    
    .footer a {
        color: #60a5fa;
        text-decoration: none;
        transition: all 0.3s;
        font-weight: 600;
    }
    
    .footer a:hover {
        color: #a78bfa;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        50% { box-shadow: 0 0 0 15px rgba(52, 211, 153, 0); }
    }
    
    /* Sidebar Gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(96, 165, 250, 0.2) !important;
    }
    
    /* Text Colors */
    .stMarkdown, label {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Loading spinner color */
    .stSpinner > div {
        border-color: rgba(96, 165, 250, 0.3);
        border-right-color: #60a5fa;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 class='title'>🧠 GENIUS PRO</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enterprise-Grade AI Document Intelligence Platform</p>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("---")
    st.image("https://img.icons8.com/fluency/100/000000/artificial-intelligence.png", width=80)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(96, 165, 250, 0.1), rgba(167, 139, 250, 0.1)); 
                padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(96, 165, 250, 0.2); margin: 1rem 0;'>
        <h3 style='background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; margin-bottom: 1rem;'>🚀 Technology Stack</h3>
        <p style='color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; line-height: 1.8;'>
            • <b>Gemini 2.5 Flash</b> (Advanced LLM)<br>
            • <b>LangChain</b> (RAG Framework)<br>
            • <b>FAISS</b> (Vector Database)<br>
            • <b>Embeddings</b> (Sentence Transformers)<br>
            • <b>N8N</b> (Workflow Automation)<br>
            • <b>Streamlit</b> (Frontend)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(96, 165, 250, 0.1), rgba(167, 139, 250, 0.1)); 
                padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(96, 165, 250, 0.2);'>
        <h3 style='background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; margin-bottom: 1rem;'>📖 Quick Start</h3>
        <p style='color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; line-height: 2;'>
            1️⃣ Upload your document<br>
            2️⃣ Wait for processing<br>
            3️⃣ Ask any question<br>
            4️⃣ Get AI-powered answers<br>
            5️⃣ Review cited sources
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem;'>
        <h3 style='background: linear-gradient(90deg, #f093fb, #f5576c); -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; margin-bottom: 1rem;'>👨‍💻 Creator</h3>
        <p style='color: rgba(255, 255, 255, 0.8); font-weight: 600;'>Anmol</p>
        <p style='color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;'>AI & Automation Expert</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
# --- LLM & EMBEDDINGS ---
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

@st.cache_resource
def load_embeddings():
    return SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

llm = load_llm()
embeddings = load_embeddings()

# --- AWESOME MODAL ---
st.markdown("""
<div id="contactModal" class="modal-overlay" style="display: none;">
    <div class="modal-content">
        <div class="modal-header">
            <h2 class="modal-title">💌 Connect With Me</h2>
            <button class="modal-close" onclick="document.getElementById('contactModal').style.display='none';">✕</button>
        </div>
        
        <div style="color: rgba(255, 255, 255, 0.9); margin-bottom: 2rem; font-size: 0.95rem; line-height: 1.8;">
            <p>Have feedback, questions, or want to collaborate? I'd love to hear from you! Fill out the form below and I'll respond promptly.</p>
        </div>
        
        <div id="formContainer"></div>
    </div>
</div>

<script>
    document.getElementById('contactModal')?.addEventListener('click', function(e) {
        if (e.target === this) {
            this.style.display = 'none';
        }
    });
    
    function loadN8NForm() {
        const container = document.getElementById('formContainer');
        if (container && !container.querySelector('iframe')) {
            const iframe = document.createElement('iframe');
            iframe.src = 'https://learnn8nmolky.app.n8n.cloud/form-test/11f029f3-d2b2-444b-904a-f6e1482b21f1';
            iframe.style.width = '100%';
            iframe.style.height = '500px';
            iframe.style.border = 'none';
            iframe.style.borderRadius = '15px';
            iframe.style.background = 'rgba(96, 165, 250, 0.05)';
            container.appendChild(iframe);
        }
    }
    
    const modal = document.getElementById('contactModal');
    const observer = new MutationObserver(() => {
        if (modal.style.display === 'flex') {
            setTimeout(loadN8NForm, 100);
        }
    });
    observer.observe(modal, { attributes: true });
</script>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 AI Engine", "⭐ Features", "🔗 Connect"])

with tab1:
    st.markdown("### 📤 Upload & Analyze")
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your document here",
        type=["pdf", "txt", "docx"],
        help="Supported: PDF, DOCX, TXT | Max 50MB"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        file_name = uploaded_file.name
        file_ext = Path(file_name).suffix.lower()
        file_size = uploaded_file.size / 1024 / 1024

        st.info(f"📁 **{file_name}** • {file_size:.2f} MB")

        temp_path = f"temp_{int(time.time())}_{file_name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔄 Processing your document..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("📖 Loading document...")
                progress_bar.progress(15)
                if file_ext == ".pdf":
                    loader = PyPDFLoader(temp_path)
                elif file_ext == ".docx":
                    loader = Docx2txtLoader(temp_path)
                else:
                    loader = TextLoader(temp_path, encoding="utf-8")
                docs = loader.load()

                status_text.text("✂️ Splitting into chunks...")
                progress_bar.progress(35)
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)

                status_text.text("🧠 Creating vector embeddings...")
                progress_bar.progress(65)
                vectorstore = FAISS.from_documents(chunks, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                status_text.text("⚡ Initializing AI model...")
                progress_bar.progress(90)
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                st.balloons()
                st.markdown("<div class='success-badge'>✅ Document ready! Ask your questions now.</div>", unsafe_allow_html=True)

                st.markdown("### ❓ Ask Questions")
                user_query = st.text_input(
                    "What do you want to know?",
                    placeholder="e.g., What are the key findings?"
                )

                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    ask_btn = st.button("🔍 Analyze", use_container_width=True)
                with col3:
                    clear_btn = st.button("🔄 Reset", use_container_width=True)

                if ask_btn and user_query.strip():
                    with st.spinner("🤔 AI is analyzing..."):
                        try:
                            response = qa(user_query)
                            answer = response["result"]
                            sources = response["source_documents"]

                            st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                            st.markdown("### 💡 Answer")
                            st.write(answer)
                            st.markdown("</div>", unsafe_allow_html=True)

                            if sources:
                                st.markdown("### 📚 Referenced Sources")
                                for i, doc in enumerate(sources, 1):
                                    page = doc.metadata.get('page', 'N/A')
                                    src = doc.metadata.get('source', 'Document')
                                    with st.expander(f"📄 Source {i} — Page {page}"):
                                        st.markdown(f"<div class='source-card'>", unsafe_allow_html=True)
                                        st.caption(f"**File:** {Path(src).name}")
                                        st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                        st.markdown("</div>", unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(96, 165, 250, 0.1), rgba(167, 139, 250, 0.1));
                    padding: 2rem; border-radius: 20px; border: 1px solid rgba(96, 165, 250, 0.2);'>
            <h3 style='background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent; margin-bottom: 1.5rem;'>⚡ Key Features</h3>
            <ul style='list-style: none; color: rgba(255, 255, 255, 0.8); font-size: 0.95rem; line-height: 2;'>
                <li>✅ <b>Lightning Fast</b> - Process docs in seconds</li>
                <li>✅ <b>AI-Powered</b> - Google Gemini 2.5 Flash</li>
                <li>✅ <b>Source Tracking</b> - Know where answers come from</li>
                <li>✅ <b>Secure Processing</b> - Files auto-deleted</li>
                <li>✅ <b>Premium UI</b> - Modern glassmorphism design</li>
                <li>✅ <b>Multi-Format</b> - PDF, DOCX, TXT support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(167, 139, 250, 0.1), rgba(244, 114, 182, 0.1));
                    padding: 2rem; border-radius: 20px; border: 1px solid rgba(167, 139, 250, 0.2);'>
            <h3 style='background: linear-gradient(90deg, #a78bfa, #f472b6); -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent; margin-bottom: 1.5rem;'>🛠️ Tech Stack</h3>
            <ul style='list-style: none; color: rgba(255, 255, 255, 0.8); font-size: 0.95rem; line-height: 2;'>
                <li>🧠 <b>LLM:</b> Google Gemini 2.5 Flash</li>
                <li>📚 <b>Embeddings:</b> Sentence Transformers</li>
                <li>🔍 <b>Vector DB:</b> FAISS</li>
                <li>⛓️ <b>Framework:</b> LangChain</li>
                <li>🎨 <b>Frontend:</b> Streamlit</li>
                <li>⚙️ <b>Automation:</b> N8N Workflows</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h3 style='font-size: 2.5rem; background: linear-gradient(90deg, #60a5fa, #a78bfa);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>100%</h3>
            <p style='color: rgba(255, 255, 255, 0.8); font-size: 0.95rem;'>Accurate Results</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h3 style='font-size: 2.5rem; background: linear-gradient(90deg, #a78bfa, #f472b6);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>< 5s</h3>
            <p style='color: rgba(255, 255, 255, 0.8); font-size: 0.95rem;'>Processing Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h3 style='font-size: 2.5rem; background: linear-gradient(90deg, #f472b6, #ff7e5f);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>∞</h3>
            <p style='color: rgba(255, 255, 255, 0.8); font-size: 0.95rem;'>Scalable</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 🌐 Connect & Follow")
    
    col3, col4 = st.columns(2)
    
    
    with col3:
        st.markdown("""
        <a href='https://learnn8nmolky.app.n8n.cloud/form/11f029f3-d2b2-444b-904a-f6e1482b21f1'
           style='display: inline-block; background: linear-gradient(135deg, rgba(244, 114, 182, 0.2), rgba(255, 126, 95, 0.2));
                  padding: 1.5rem; border-radius: 15px; text-decoration: none; text-align: center;
                  border: 1px solid rgba(244, 114, 182, 0.3); transition: all 0.3s; width: 100%;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>📧</div>
            <div style='color: white; font-weight: 600;'>Email</div>
        </a>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <a href='https://outreachify.io/' target='_blank'
           style='display: inline-block; background: linear-gradient(135deg, rgba(255, 126, 95, 0.2), rgba(96, 165, 250, 0.2));
                  padding: 1.5rem; border-radius: 15px; text-decoration: none; text-align: center;
                  border: 1px solid rgba(255, 126, 95, 0.3); transition: all 0.3s; width: 100%;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🌐</div>
            <div style='color: white; font-weight: 600;'>Portfolio</div>
        </a>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(96, 165, 250, 0.1), rgba(167, 139, 250, 0.1));
                padding: 2rem; border-radius: 20px; border: 1px solid rgba(96, 165, 250, 0.2); margin-top: 2rem;'>
        <h3 style='background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent; margin-bottom: 1rem;'>💬 Let's Work Together</h3>
        <p style='color: rgba(255, 255, 255, 0.8); line-height: 1.8;'>
            I'm passionate about AI, automation, and building innovative solutions. Whether you have a project idea, feedback, 
            or just want to connect, I'd love to hear from you! Use the contact button to reach out.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class='footer'>
    <p>🚀 <b>RAG GENIUS PRO v3.0</b> | Enterprise AI Document Intelligence</p>
    <p>Powered by Google Gemini • LangChain • FAISS • Streamlit • N8N</p>
    <p style='margin-top: 1.5rem; font-size: 0.9rem;'>
        Built by <a href='#' style='color: #60a5fa;'>Anmol</a> | 
    </p>
    <p style='margin-top: 1rem; font-size: 0.85rem; color: rgba(255, 255, 255, 0.5);'>
        © 2025 Anmol. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)