import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import os

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to compare documents based on cosine similarity
def compare_documents(doc1, doc2):
    # Convert documents to a list of text
    documents = [doc1, doc2]

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Convert documents into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Return similarity score between the two documents
    return similarity_matrix[0, 1]

# Function to generate insights based on document content (using transformers)
def generate_insights(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Function to answer questions from document content using QA pipeline
def answer_question(question, context):
    qa_pipeline = pipeline("question-answering")
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Streamlit app
def main():
    st.title("PDF Document Comparison & Assistant")
    st.write("This app compares PDF documents, generates insights, and answers questions about the documents.")
    
    # File upload
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if len(uploaded_files) == 2:
        # Extract text from uploaded PDFs
        pdf_texts = []
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            pdf_texts.append(text)

        # Compare documents
        similarity_score = compare_documents(pdf_texts[0], pdf_texts[1])
        st.write(f"Similarity score between the two documents: {similarity_score:.2f}")

        # Generate insights
        st.subheader("Generated Insights")
        document_1_summary = generate_insights(pdf_texts[0])
        document_2_summary = generate_insights(pdf_texts[1])

        st.write(f"Document 1 Summary: {document_1_summary}")
        st.write(f"Document 2 Summary: {document_2_summary}")

        # QA Section
        st.subheader("Ask a Question About the Documents")
        question = st.text_input("Enter your question:")
        
        if question:
            # Combine both documents into one context for QA
            combined_text = pdf_texts[0] + "\n\n" + pdf_texts[1]
            answer = answer_question(question, combined_text)
            st.write(f"Answer: {answer}")

    elif len(uploaded_files) > 2:
        st.warning("Please upload exactly two PDF files for comparison.")
    elif len(uploaded_files) == 0:
        st.warning("Please upload two PDF files to compare.")

if __name__ == "__main__":
    main()
