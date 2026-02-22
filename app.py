import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint

@st.cache_resource
def load_rag():
    loader = PyPDFDirectoryLoader("kb/")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Free HF Inference LLM (works reliably on Spaces)
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.2,
        max_new_tokens=512,
        huggingfacehub_api_token=st.secrets.get("HF_TOKEN")
    )
    
    prompt = PromptTemplate.from_template(
        """Use the following pieces of context to answer the question. 
        If you don't know, just say "I don't know".
        
        Context: {context}
        
        Question: {input}
        
        Helpful Answer:"""
    )
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    
    return rag_chain

rag = load_rag()

st.title("🏥 Charity Funding RAG Assistant")
st.caption("Ask anything about Singapore charity grants, eligibility, budgets, compliance, etc.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = rag.invoke({"input": prompt})
            answer = result["answer"]
            sources = len(result.get("context", []))
            response = f"{answer}\n\n**Sources used:** {sources} document chunks"
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
