# RAG Project (Retrieval-Augmented Generation)

This project implements a Retrieval-Augmented Generation (RAG) pipeline to crawl websites, index textual content, and answer queries using a QA system.

## Features & Instructions

- Developed and tested in **Google Colab**, allowing GPU acceleration for embeddings and answer generation.
- Clone the repository and navigate: `git clone <repo_url> && cd <repo_folder>`
- Install dependencies: `pip install -r requirements.txt`
- Project folders (`src/`, `data/`, `logs/`) are auto-created on setup.
- **Crawl**: Use `WebCrawler(start_url, max_pages, max_depth)` to crawl websites; saves results in `data/crawled_documents.json`.
- **Index**: Use `TextIndexer(chunk_size, chunk_overlap)` to chunk documents, generate embeddings, and build HNSW vector index (`data/vector_index.bin` and `data/chunks.json`).
- **Ask**: Use `QuestionAnswerer(top_k, similarity_threshold)` to retrieve chunks and generate answers with sources and confidence.
- Supports evaluation of multiple questions; outputs saved in `data/qa_results.json`.
- Generates performance charts (retrieval vs generation time, confidence distribution) in `data/qa_analysis.png`.
- BFS-style crawler with robots.txt compliance and domain-specific link restriction.
- Smart chunking (400 chars, 20% overlap) improves retrieval precision and context continuity.
- Embeddings generated with `sentence-transformers/all-MiniLM-L6-v2` and normalized for cosine similarity.
- Vector search with HNSW (`hnswlib`) for fast approximate nearest neighbor retrieval.
- Answer generation with `flan-t5-large`; refusal detection ensures out-of-scope questions are handled safely.
- Tradeoffs: smaller chunks improve precision but increase storage; overlap preserves context; greedy decoding reduces hallucination but limits diversity; CPU inference is slow, GPU recommended; crawling respects robots.txt but may skip pages; HNSW parameters tuned for moderate recall; re-crawl/re-index needed for new content.


