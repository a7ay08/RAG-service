
import pandas as pd
import matplotlib.pyplot as plt


df = pd.DataFrame([
    {
        'question': r['question'][:50] + '...' if len(r['question']) > 50 else r['question'],
        'confidence': r['confidence'],
        'num_sources': len(r['sources']),
        'retrieval_ms': r['timings']['retrieval_ms'],
        'generation_ms': r['timings']['generation_ms'],
        'total_ms': r['timings']['total_ms']
    }
    for r in results
])

print("\n📈 Performance Summary:")
print(df.to_string(index=False))

# Plot timing breakdown
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Timing comparison
df[['retrieval_ms', 'generation_ms']].plot(kind='bar', ax=axes[0], stacked=True)
axes[0].set_title('Timing Breakdown by Question')
axes[0].set_xlabel('Question Index')
axes[0].set_ylabel('Time (ms)')
axes[0].legend(['Retrieval', 'Generation'])

# Confidence distribution
confidence_counts = df['confidence'].value_counts()
axes[1].bar(confidence_counts.index, confidence_counts.values, color=['green', 'orange', 'red', 'gray'])
axes[1].set_title('Answer Confidence Distribution')
axes[1].set_xlabel('Confidence Level')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('data/qa_analysis.png', dpi=150, bbox_inches='tight')
print("\n📊 Analysis chart saved to data/qa_analysis.png")
