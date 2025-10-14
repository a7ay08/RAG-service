
from src.ask import QuestionAnswerer
import json

# Initialize QA system
qa = QuestionAnswerer(
    top_k=10,
    similarity_threshold=0.40  
)

# Test questions
questions = [
    "Who is the Prime Minister of India?", 
    "What is Python?",
    "When was Python 3.0 released?",
    "What is the weather forecast for tomorrow in Tokyo?"  
]


results = []
for i, question in enumerate(questions, 1):
    print("\n" + "="*80)
    print(f"\n📝 Question {i}: {question}")
    
   
    result = qa.ask(question)
    results.append(result)
    
   
    print(f"\n💬 Answer: {result['answer']}")
    print(f"\n📊 Metrics:")
    print(f"   Confidence: {result['confidence']}")
    if result['confidence'] != 'none':
        print(f"   Chunks used: {result['num_chunks_used']}")
    print(f"   Retrieval time: {result['timings']['retrieval_ms']} ms")
    print(f"   Generation time: {result['timings']['generation_ms']} ms")
    print(f"   Total time: {result['timings']['total_ms']} ms")
    
   
    if result['sources']:
        print(f"\n🔗 Sources ({len(result['sources'])} URLs):")
        for j, source in enumerate(result['sources'][:3], 1):
            print(f"   {j}. [{source['similarity']:.3f}] {source['url']}")
            print(f"      → {source['snippet'][:150]}...")
    else:
        print("\n⚠️ No sources found (below similarity threshold)")


with open('data/qa_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("✅ All questions processed! Results saved to data/qa_results.json")
