

%%writefile src/indexer.py
import os
import json
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import re

class TextIndexer:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print("Loading embedding model: all-MiniLM-L6-v2...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        self.chunks: List[Dict] = []
        self.index = None
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, url: str) -> List[Dict]:
      text = self.clean_text(text)
      sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
           
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'url': url,
                    'sentences': current_sentences.copy()
                })
                
                
                overlap_text = " ".join(current_sentences[-2:]) if len(current_sentences) >= 2 else current_sentences[-1] if current_sentences else ""
                current_chunk = overlap_text + " " + sentence
                current_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences[-1:] if current_sentences else []
                current_sentences.append(sentence)
            else:
                current_chunk += " " + sentence
                current_sentences.append(sentence)
        
       
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'url': url,
                'sentences': current_sentences
            })
        
        return chunks
    
    def index_documents(self, documents_path: str = 'data/crawled_documents.json'):
      with open(documents_path, 'r') as f:
            documents = json.load(f)
        
        print(f"Processing {len(documents)} documents...")
        
       
        for doc in documents:
            doc_chunks = self.chunk_text(doc['text'], doc['url'])
            self.chunks.extend(doc_chunks)
        
        print(f"Created {len(self.chunks)} chunks (avg {np.mean([len(c['text']) for c in self.chunks]):.0f} chars)")
        
       
        print("Generating embeddings...")
        chunk_texts = [c['text'] for c in self.chunks]
        embeddings = self.embedding_model.encode(
            chunk_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
      
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        print("Building vector index...")
        self.index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self.index.init_index(
            max_elements=len(embeddings),
            ef_construction=200,  
            M=16  
        )
        self.index.add_items(embeddings, np.arange(len(embeddings)))
        self.index.set_ef(50) 
        
       
        print("Saving index and chunks...")
        self.index.save_index('data/vector_index.bin')
        
        with open('data/chunks.json', 'w') as f:
            json.dump(self.chunks, f, indent=2)
        
        # Save embeddings for later use
        np.save('data/embeddings.npy', embeddings)
        
        return {
            'vector_count': len(self.chunks),
            'embedding_dim': self.embedding_dim,
            'avg_chunk_size': np.mean([len(c['text']) for c in self.chunks]),
            'errors': []
        }
