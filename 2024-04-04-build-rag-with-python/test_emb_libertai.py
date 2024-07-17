'''
Test on Langchain : libertai llama-cpp online embedding free access


'''
# import
from libs.libertai import LibertaiEmbeddings
from sentence_transformers.util import cos_sim
import numpy as np
# definitions

 # TEST with 
embedder = LibertaiEmbeddings()

query = 'Represent this sentence for searching relevant passages: A man is eating a piece of bread'

docs = [
    query,
    "A man is eating food.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
]
print("\n\n TEST 1 : embbding docs : \n")
r_1 = embedder.embed_documents(docs)

# Calculate cosine similarity
similarities = cos_sim(r_1[0], r_1[1:])
print('TEST 1 : LIBERTAI... ')
r1_array = similarities.numpy()[0]
print(r1_array)
print("to be compared to :\n [0.7920, 0.6369, 0.1651, 0.3621]")
ref_array = np.array([0.7920, 0.6369, 0.1651, 0.3621])

assert np.all(np.argsort(r1_array) == np.argsort(ref_array)), "TEST 1 : LIBERTAI FAILED: similarities are not same order with ref embeddings"
print("TEST 1 : LIBERTAI PASSED.")

print("END")
