import os
import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def load_vector_store():
    """Load the FAISS vector store and chunks"""
    try:
        # Load FAISS index
        index = faiss.read_index(os.path.join(DB_FAISS_PATH, "index.faiss"))
        
        # Load chunks
        with open(os.path.join(DB_FAISS_PATH, "chunks.pkl"), 'rb') as f:
            chunks = pickle.load(f)
        
        # Load embedding model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        return index, chunks, model
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None, None, None

def search_similar_chunks(query, model, index, chunks, k=3):
    """Search for similar chunks"""
    # Embed the query
    query_embedding = model.encode([query])
    
    # Search in FAISS
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    # Get the relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    
    return relevant_chunks, distances[0]

def format_response(query, relevant_chunks):
    """Format a simple response based on retrieved chunks"""
    if not relevant_chunks:
        return "I couldn't find relevant information in the medical encyclopedia to answer your question."
    
    # Create a simple response
    response = f"Based on the medical encyclopedia, here's what I found about your question:\n\n"
    
    for i, chunk in enumerate(relevant_chunks[:2], 1):  # Show top 2 results
        response += f"**Source {i}:**\n{chunk['page_content'][:500]}...\n\n"
    
    response += "\n*Note: This is for informational purposes only. Please consult a healthcare professional for medical advice.*"
    
    return response

def main():
    st.title("🩺 Medical Chatbot")
    st.markdown("Ask questions about medical topics based on the GALE Encyclopedia of Medicine")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Ask a medical question...")

    if prompt:
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        try:
            with st.spinner("Searching medical knowledge base..."):
                # Load vector store
                index, chunks, model = load_vector_store()
                
                if index is None:
                    st.error("Failed to load the medical knowledge base")
                    return
                
                # Search for relevant chunks
                relevant_chunks, distances = search_similar_chunks(prompt, model, index, chunks)
                
                # Generate response
                response = format_response(prompt, relevant_chunks)
                
                with st.chat_message('assistant'):
                    st.markdown(response)
                
                st.session_state.messages.append({'role': 'assistant', 'content': response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
