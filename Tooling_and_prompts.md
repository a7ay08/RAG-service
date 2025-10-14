## Tooling & Prompts

- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` for dense vector representation.
- **LLM:** `google/flan-t5-large` for answer generation.
- **Libraries:** `trafilatura`, `beautifulsoup4`, `hnswlib`, `torch`, `transformers`, `requests`, `tqdm`, `pandas`, `matplotlib`, `robotexclusionrulesparser`, `accelerate`, `pytest`.
- **Prompt Template:**  
  ```text
  Using the information below, answer the question directly and concisely.

  Information:
  <retrieved chunks>

  Question: <user question>
  Answer:
