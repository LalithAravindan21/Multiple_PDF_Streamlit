import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# Load environment variables if needed
from dotenv import load_dotenv
load_dotenv()

# Streamlit app setup
st.title("PDF Comparison and Analysis Assistant")
st.write("Upload multiple PDF documents to compare their content and ask questions.")

# File upload for PDFs
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
docs = []

# Check for API key in environment
if not os.getenv("OPENAI_API_KEY"):
    st.error("Please set your OpenAI API key in a .env file as 'OPENAI_API_KEY'")

if uploaded_files and os.getenv("OPENAI_API_KEY"):
    # Load PDFs and extract text
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        docs.append(text)

    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    faiss_index = FAISS.from_texts(docs, embeddings)

    # RAG setup with OpenAI’s model
    chat_model = ChatOpenAI(temperature=0.5)
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="map_reduce",
        retriever=faiss_index.as_retriever()
    )

    # Function to compare PDFs and highlight differences
    def compare_pdfs(docs):
        comparison_results = []
        for i in range(len(docs) - 1):
            comparison_query = f"Compare the content differences between Document {i+1} and Document {i+2}."
            result = retrieval_chain.run(input_documents=[docs[i], docs[i+1]], question=comparison_query)
            comparison_results.append(result)
        return comparison_results

    # Display comparison results
    if len(docs) > 1:
        comparison_results = compare_pdfs(docs)
        for i, result in enumerate(comparison_results):
            st.write(f"Comparison between Document {i+1} and Document {i+2}:")
            st.write(result)
    else:
        st.info("Please upload at least two PDF files to compare.")

    # Assistant capability
    user_question = st.text_input("Ask a question about the documents:")
    if user_question:
        answer = retrieval_chain.run(input_documents=docs, question=user_question)
        st.write("Answer:", answer)

else:
    if not uploaded_files:
        st.info("Please upload PDF files to start comparing and analyzing.")
    else:
        st.error("OpenAI API key is missing. Please add it to a .env file as 'OPENAI_API_KEY'.")
