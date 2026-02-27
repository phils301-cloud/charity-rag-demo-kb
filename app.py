import streamlit as st
import os

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Use langchain_classic for legacy chains (required in LangChain 1.0+)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

INDEX_FOLDER = "charity_faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_REPO = "mistralai/Mistral-7B-Instruct-v0.3"
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 400
RETRIEVER_K = 4

# ────────────────────────────────────────────────
# VECTOR STORE LOADING (cached)
# ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading FAISS index…")
def load_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if not os.path.exists(INDEX_FOLDER):
            st.error(f"Index folder not found: {INDEX_FOLDER}")
            st.stop()

        files = os.listdir(INDEX_FOLDER)
        st.info(f"Index folder found. Files: {files}")

        vectorstore = FAISS.load_local(
            INDEX_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.success("FAISS index loaded successfully")
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load FAISS index\n\n{str(e)}")
        st.stop()

# ────────────────────────────────────────────────
# LLM & CHAIN SETUP (cached)
# ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing LLM and chain…")
def create_rag_chain():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        st.error("HF_TOKEN secret is missing. Please add it in Streamlit Cloud → Secrets.")
        st.stop()

    try:
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=LLM_REPO,
            huggingfacehub_api_token=hf_token,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        llm = ChatHuggingFace(llm=llm_endpoint)

        prompt = ChatPromptTemplate.from_template(
            """You are a Singapore Charity Expert. Answer the question using only the provided context. 
Be accurate, concise and professional.

Context:
{context}

Question: {input}

Answer:"""
        )

        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(
            load_vectorstore().as_retriever(search_kwargs={"k": RETRIEVER_K}),
            combine_docs_chain
        )

        return retrieval_chain
    except Exception as e:
        st.error(f"Failed to initialize LLM or chain\n\n{str(e)}")
        st.stop()

# ────────────────────────────────────────────────
# STREAMLIT UI
# ────────────────────────────────────────────────

st.set_page_config(
    page_title="Charity Grant Assistant",
    page_icon="🇸🇬",
    layout="wide"
)

st.title("🇸🇬 Singapore Charity Grant Assistant")
st.markdown("Ask questions about charity rules, grants, IPC status, governance, compliance, etc.")

# Load everything on first run
with st.spinner("Starting system..."):
    rag_chain = create_rag_chain()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about charity grants, governance, eligibility..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)

                # Optional: show sources (uncomment if desired)
                # st.markdown("**Sources used:**")
                # for doc in response.get("context", []):
                #     st.markdown(f"- {doc.page_content[:200]}...")

            except Exception as e:
                st.error(f"Error during generation:\n\n{str(e)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})