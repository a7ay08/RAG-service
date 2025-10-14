
%%writefile src/ask.py
import os
import json
import time
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict, Tuple
import torch

class QuestionAnswerer:
    def __init__(self, top_k: int = 10, similarity_threshold: float = 0.40):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        print("Loading generator model: flan-t5-large...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-large",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print("Loading vector index...")
        self.index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self.index.load_index('data/vector_index.bin')
        self.index.set_ef(50)
        
        with open('data/chunks.json', 'r') as f:
            self.chunks = json.load(f)
        
        print(f"✅ Loaded {len(self.chunks)} chunks")
    
    def retrieve(self, question: str) -> Tuple[List[Dict], float]:
        start_time = time.time()
        
        question_embedding = self.embedding_model.encode([question], convert_to_numpy=True)
        question_embedding = question_embedding / np.linalg.norm(question_embedding, axis=1, keepdims=True)
        
        labels, distances = self.index.knn_query(question_embedding, k=min(self.top_k, len(self.chunks)))
        similarities = 1 - distances[0]
        
        retrieved = []
        for idx, sim in zip(labels[0], similarities):
            if sim >= self.similarity_threshold:
                chunk = self.chunks[idx].copy()
                chunk['similarity'] = float(sim)
                retrieved.append(chunk)
        
        retrieved = sorted(retrieved, key=lambda x: x['similarity'], reverse=True)
        retrieval_time = time.time() - start_time
        
        return retrieved, retrieval_time
    
    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> Tuple[str, float]:
        start_time = time.time()
        
        if not retrieved_chunks:
            return "I could not find relevant information in the crawled content to answer this question.", time.time() - start_time
        
        # Build context from chunks
        context_parts = [chunk['text'] for chunk in retrieved_chunks]
        context = "\n\n".join(context_parts)
        
        # CRITICAL FIX: Use instruction format that FLAN-T5 understands better
        # Remove cautious language that makes model refuse
        prompt = f"""Using the information below, answer the question directly and concisely.

Information:
{context}

Question: {question}
Answer:"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1536,
            truncation=True
        ).to(self.generator.device)
        
        # CRITICAL FIX: Use greedy decoding for more confident answers
        outputs = self.generator.generate(
            **inputs,
            max_new_tokens=150,
            min_new_tokens=3,
            num_beams=1,  
            temperature=1.0,  
            do_sample=False,  
            early_stopping=True
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time = time.time() - start_time
        
        return answer.strip(), generation_time
    
    def ask(self, question: str) -> Dict:
        total_start = time.time()
        
        retrieved_chunks, retrieval_time = self.retrieve(question)
        
        print(f"\n📊 Retrieved {len(retrieved_chunks)} chunks (threshold: {self.similarity_threshold})")
        if retrieved_chunks:
            print(f"   Top similarity: {retrieved_chunks[0]['similarity']:.3f}")
            if len(retrieved_chunks) > 1:
                print(f"   Lowest similarity: {retrieved_chunks[-1]['similarity']:.3f}")
        
        if not retrieved_chunks:
            return {
                'question': question,
                'answer': "I could not find relevant information in the crawled content to answer this question.",
                'sources': [],
                'timings': {
                    'retrieval_ms': int(retrieval_time * 1000),
                    'generation_ms': 0,
                    'total_ms': int((time.time() - total_start) * 1000)
                },
                'confidence': 'none',
                'num_chunks_used': 0
            }
        
        answer, generation_time = self.generate_answer(question, retrieved_chunks)
        
        
        refusal_indicators = [
            "could not find",
            "cannot answer based on",
            "information is not available"
        ]
        is_refusal = any(indicator in answer.lower() for indicator in refusal_indicators)
        
        sources = []
        seen_urls = set()
        for chunk in retrieved_chunks:
            if chunk['url'] not in seen_urls:
                sources.append({
                    'url': chunk['url'],
                    'snippet': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    'similarity': chunk['similarity']
                })
                seen_urls.add(chunk['url'])
        
        total_time = time.time() - total_start
        
        if is_refusal:
            confidence = 'refused'
        elif retrieved_chunks[0]['similarity'] > 0.7:
            confidence = 'high'
        elif retrieved_chunks[0]['similarity'] > 0.55:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'timings': {
                'retrieval_ms': int(retrieval_time * 1000),
                'generation_ms': int(generation_time * 1000),
                'total_ms': int(total_time * 1000)
            },
            'confidence': confidence,
            'num_chunks_used': len(retrieved_chunks)
        }
