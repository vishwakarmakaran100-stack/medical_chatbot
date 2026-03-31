import os
import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")

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

def get_groq_response(query, context):
    """Get response from Groq API"""
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None
            
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        prompt = f"""You are a helpful medical assistant. Based on the following medical information from an encyclopedia, please answer the user's question. Provide accurate, helpful information while noting that this is for informational purposes only.

Context:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context provided:"""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 512
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return None
            
    except Exception as e:
        st.error(f"Error calling Groq API: {str(e)}")
        return None

def format_context(relevant_chunks):
    """Format context chunks into a single string"""
    if not relevant_chunks:
        return "No relevant information found."
    
    context = ""
    for i, chunk in enumerate(relevant_chunks, 1):
        context += f"Source {i}: {chunk['page_content'][:800]}\n\n"
    
    return context

def main():
    st.title("🩺 Enhanced Medical Chatbot")
    st.markdown("Ask questions about medical topics with AI-powered responses")
    
    # Check for API keys
    groq_key = os.environ.get("GROQ_API_KEY")
    hf_key = os.environ.get("HF_TOKEN")
    
    if groq_key and groq_key.startswith('gsk_'):
        st.success("✅ Groq API key detected - Enhanced responses available")
    else:
        st.warning("⚠️ Groq API key not found - Basic responses only")

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
                
                # Format context
                context = format_context(relevant_chunks)
                
                # Try to get enhanced response from Groq
                if groq_key and groq_key.startswith('gsk_'):
                    with st.spinner("Generating AI response..."):
                        ai_response = get_groq_response(prompt, context)
                        
                        if ai_response:
                            response = ai_response
                        else:
                            # Fallback to basic response
                            response = f"**AI Response Unavailable - Showing Basic Results:**\n\n{format_response(prompt, relevant_chunks)}"
                else:
                    # Basic response without AI
                    response = format_response(prompt, relevant_chunks)
                
                with st.chat_message('assistant'):
                    st.markdown(response)
                
                st.session_state.messages.append({'role': 'assistant', 'content': response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

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

if __name__ == "__main__":
    main()
