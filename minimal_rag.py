import faiss
import numpy as np
from openai import OpenAI
import os

dim = 5  # dimension of the vectors
nlist = 1  # how many cells to create
quantizer = faiss.IndexFlatL2(dim) 
index = faiss.IndexIVFFlat(quantizer, dim, nlist)

docs = [
    'Vectors are used in NLP',
    'Word Embeddings are used in NLP',
    'Word2Vec is a model used in NLP',
]

emb = np.array([
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.2, 0.3, 0.4, 0.5, 0.6],
    [0.3, 0.4, 0.5, 0.6, 0.7]
], dtype=np.float32)

xq = np.array([[0.1, 0.2, 0.3, 0.3, 0.5]], dtype=np.float32) # Mock vec representation of "How are vectors used?"

faiss.normalize_L2(emb)
faiss.normalize_L2(xq)

index.train(emb)
index.add(emb)

D, indices = index.search(xq.reshape(1, -1), k=4)  # Reshape xq to make it a 2D array
context_i = indices[0][0]
context = docs[context_i]

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": f"Based on the provided context, answer the following question: Where are vectors used?\n\nContext:\n{context}. Be concise",
        },
        {
            "role": "user",
            "content": "Where are vectors used?",
        },
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion.choices[0].message.content.strip()) # Output: "Vectors are used in NLP (Natural Language Processing)."
