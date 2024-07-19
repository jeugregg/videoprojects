'''
Test on Langchain : Simple Embedding with Langchain + Ollama using query embedding
Compare to result from creators of mxbai-embed-large-v1
https://www.mixedbread.ai/blog/mxbai-embed-large-v1
'''

# import
from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers.util import cos_sim
import numpy as np
from numpy.testing import assert_almost_equal
# definitions
ollama_emb = OllamaEmbeddings(model="mxbai-embed-large", query_instruction="")

# test on ollama
query = "Represent this sentence for searching relevant passages: A man is eating a piece of bread"

docs = [
    query,
    "A man is eating food.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
]

#r_1 = ollama_emb.embed_documents(docs)
r_1 = [ollama_emb.embed_query(doc) for doc in docs]

# Calculate cosine similarity
similarities = cos_sim(r_1[0], r_1[1:])
print(similarities.numpy()[0])
print("to be compared to :\n [0.7920, 0.6369, 0.1651, 0.3621]")
try :
    assert_almost_equal(similarities.numpy()[0], np.array([0.7920, 0.6369, 0.1651, 0.3621]),decimal=2)
    print("TEST 1 : OLLAMA PASSED.")
except AssertionError:
    print("TEST 1 : OLLAMA FAILED.")