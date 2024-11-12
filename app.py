import streamlit as st
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# Function to extract text from PDFs
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
def split_text(text, chunk_size=1000):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to create FAISS index for the documents
def create_faiss_index(text_chunks):
    embeddings = OpenAIEmbeddings()
    faiss_index = FAISS.from_texts(text_chunks, embeddings)
    return faiss_index

# Function to compare multiple PDFs
def compare_pdfs(file_paths):
    # Extract text from each PDF and split them into chunks
    all_chunks = []
    for file_path in file_paths:
        text = extract_text_from_pdf(file_path)
        chunks = split_text(text)
        all_chunks.extend(chunks)
    
    # Create FAISS index for all documents
    faiss_index = create_faiss_index(all_chunks)
    
    # Create retrieval chain for querying the index
    retriever = faiss_index.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(OpenAI(), retriever)
    
    return qa_chain

# Function to process user queries
def process_query(query, qa_chain):
    return qa_chain.run(query)

# Streamlit interface
st.title("PDF Document Comparison and Chatbot Interface")

# File upload for multiple PDFs
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Save uploaded files temporarily
    file_paths = []
    for idx, uploaded_file in enumerate(uploaded_files):
        temp_file_path = f"temp_file_{idx}.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        file_paths.append(temp_file_path)
    
    # Compare PDFs and create a retrieval chain
    qa_chain = compare_pdfs(file_paths)

    # Chatbot interaction
    st.subheader("Ask questions about the uploaded documents:")
    user_query = st.text_input("Enter your query")

    if user_query:
        response = process_query(user_query, qa_chain)
        st.write("Answer:", response)

    # Display uploaded PDF names
    st.write("Uploaded PDFs:")
    for uploaded_file in uploaded_files:
        st.write(uploaded_file.name)
