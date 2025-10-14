
from src.indexer import TextIndexer


indexer = TextIndexer(
    chunk_size=400,      
    chunk_overlap=80     
)


index_result = indexer.index_documents()

print(f"\n✅ Indexing complete!")
print(f"Total chunks: {index_result['vector_count']}")
print(f"Average chunk size: {index_result['avg_chunk_size']:.0f} chars")
print(f"Embedding dimension: {index_result['embedding_dim']}")
