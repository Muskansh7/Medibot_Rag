import streamlit as st
import os
import time
import uuid

from dotenv import load_dotenv
from PIL import Image

import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="MediBot AI",
    page_icon="🩺",
    layout="wide"
)


# ---------------- LOAD ENV ---------------- #

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ---------------- GEMINI CONFIG ---------------- #

genai.configure(api_key=GEMINI_API_KEY)

vision_model = genai.GenerativeModel(
    "gemini-2.0-flash"
)


# ---------------- IMAGE STORAGE ---------------- #

UPLOAD_FOLDER = "uploaded_images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------- SIMPLE CSS ---------------- #

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.small-box{
    padding:15px;
    border-radius:12px;
    background-color:#f5f7fa;
    border:1px solid #e6e6e6;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🩺 MediBot AI")

    st.caption("Medical RAG Assistant")

    st.markdown("### Features")

    st.markdown("""
    ✅ Medical Chat  
    ✅ PDF Knowledge Base  
    ✅ Medical Image Analysis  
    ✅ Reference Pages  
    """)

    st.warning("⚠️ Educational Use Only")


# ---------------- HEADER ---------------- #

st.title("🩺 MediBot AI")

st.caption("AI Powered Medical Assistant")


# ---------------- DASHBOARD ---------------- #

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="small-box">
    📚<br>
    <b>Medical Encyclopedia</b>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="small-box">
    🤖<br>
    <b>Groq + Gemini AI</b>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="small-box">
    🖼️<br>
    <b>Image Analysis</b>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ---------------- VECTOR DATABASE ---------------- #

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# ---------------- LOAD LLM ---------------- #

def load_llm():

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.5
    )

    return llm


# ---------------- PROMPT ---------------- #

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.

If you don't know the answer, just say that you dont know.

Dont try to make up an answer.

Context: {context}
Question: {question}

Start the answer directly.
"""


def set_custom_prompt():

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return prompt


# ---------------- LOAD DATABASE ---------------- #

db = get_vectorstore()


# ---------------- QA CHAIN ---------------- #

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={
        'prompt': set_custom_prompt()
    }
)


# ---------------- SESSION STATE ---------------- #

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------- DISPLAY HISTORY ---------------- #

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------- IMAGE UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "📸 Upload Medical Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    unique_name = f"{uuid.uuid4()}.png"

    image_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    with open(image_path, "wb") as f:

        f.write(uploaded_file.getbuffer())

    image = Image.open(image_path)

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    with st.spinner("🧠 Analyzing image..."):

        image_response = vision_model.generate_content([
            "Analyze this medical image and explain possible condition.",
            image
        ])

        st.subheader("🧠 Image Analysis")

        st.write(image_response.text)


# ---------------- CHAT INPUT ---------------- #

prompt = st.chat_input(
    "Ask your medical question..."
)


# ---------------- RESPONSE ---------------- #

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)

    lower_prompt = prompt.lower()

    # ---------------- GREETINGS ---------------- #

    if lower_prompt in ["hi", "hello", "hey"]:

        result = """
👋 Hello! I am MediBot AI.

I can help you with:
- Medical questions
- Disease information
- Image analysis
- Reference pages
"""

        source_documents = []

    else:

        with st.spinner("🧠 Thinking..."):

            time.sleep(1)

            response = qa_chain.invoke(
                {'query': prompt}
            )

            result = response["result"]

            source_documents = response["source_documents"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": result
    })

    with st.chat_message("assistant"):

        st.markdown(result)

        # ---------------- SOURCE REFERENCES ---------------- #

        if source_documents:

            with st.expander("📚 Source References"):

                shown_pages = set()

                for doc in source_documents:

                    page_number = doc.metadata.get("page")

                    source_text = doc.page_content[:400]

                    st.markdown(
                        f"### 📄 Page {page_number + 1}"
                    )

                    st.write(source_text)

                    image_path = (
                        f"page_images/page_{page_number + 1}.png"
                    )

                    if os.path.exists(image_path):

                        if image_path not in shown_pages:

                            st.image(
                                image_path,
                                caption=f"Reference Page {page_number + 1}",
                                width=500
                            )

                            shown_pages.add(image_path)


# ---------------- FOOTER ---------------- #

st.divider()

st.caption(
    "⚠️ MediBot AI is for educational purposes only."
)