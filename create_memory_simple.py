import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def load_pdf_files(data_path):
    """Load PDF files from the data directory"""
    documents = []
    pdf_file = os.path.join(data_path, "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
    
    if os.path.exists(pdf_file):
        reader = PdfReader(pdf_file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                documents.append({
                    'page_content': text,
                    'metadata': {'source': pdf_file, 'page': i}
                })
    return documents

def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    """Create text chunks from documents"""
    text_chunks = []
    
    for doc in documents:
        text = doc['page_content']
        metadata = doc['metadata']
        
        # Simple chunking
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                text_chunks.append({
                    'page_content': chunk_text,
                    'metadata': metadata
                })
    
    return text_chunks

def create_vector_store():
    """Create and save FAISS vector store"""
    print("Loading PDF files...")
    documents = load_pdf_files("data/")
    print(f"Loaded {len(documents)} pages")
    
    print("Creating chunks...")
    text_chunks = create_chunks(documents)
    print(f"Created {len(text_chunks)} chunks")
    
    print("Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("Creating embeddings...")
    texts = [chunk['page_content'] for chunk in text_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    print("Saving vector store...")
    os.makedirs("vectorstore/db_faiss", exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, "vectorstore/db_faiss/index.faiss")
    
    # Save chunks and metadata
    with open("vectorstore/db_faiss/chunks.pkl", 'wb') as f:
        pickle.dump(text_chunks, f)
    
    print("Vector store created successfully!")
    print(f"Index contains {index.ntotal} vectors")

if __name__ == "__main__":
    create_vector_store()
