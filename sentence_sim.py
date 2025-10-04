from sentence_transformers import SentenceTransformer, util

# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Sentences from your AI model's output
sentences = [
    "This is an example sentence.",
    "This sentence is an example.",
    "The example of this sentence."
]

# Encode the sentences to get their embeddings
embeddings = model.encode(sentences)

# Calculate cosine similarity between the first sentence and all other sentences
cosine_scores = util.cos_sim(embeddings[0], embeddings)

# Print the results
for i in range(len(sentences)):
    print(f"Similarity between '{sentences[0]}' and '{sentences[i]}': {cosine_scores[0][i]:.4f}")