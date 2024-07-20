'''
Test on Langchain : Simple Embedding with Ollama and Llamafile
Compare to result from creators of mxbai-embed-large-v1
https://www.mixedbread.ai/blog/mxbai-embed-large-v1
'''

# import
import os
import ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import LlamafileEmbeddings
from sentence_transformers.util import cos_sim
import numpy as np
from numpy.testing import assert_almost_equal
from utilities import getconfig
from libs.tools_llamafile import launch_llamafile
# definitions
embedmodel = getconfig()["embedmodel"]
print("Tested model embedded: ",  embedmodel)
ollama.pull(embedmodel)
ollama_emb = OllamaEmbeddings(model=embedmodel, embed_instruction="")

llamafilename = getconfig("emb-llamafile")["embedmodel"]
pathdata = getconfig("emb-llamafile")["pathdata"]
llamafilename = os.path.join(pathdata, llamafilename)

query = 'Represent this sentence for searching relevant passages: A man is eating a piece of bread'

docs = [
    query,
    "A man is eating food.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
]

r_1 = ollama_emb.embed_documents(docs)

# Calculate cosine similarity
similarities = cos_sim(r_1[0], r_1[1:])
print('TEST 1 : OLLAMA... ')
print(similarities.numpy()[0])
print("to be compared to :\n [0.7920, 0.6369, 0.1651, 0.3621]")
try :
    assert_almost_equal( similarities.numpy()[0], np.array([0.7920, 0.6369, 0.1651, 0.3621]),decimal=2)
    print("TEST 1 : OLLAMA PASSED.")
except AssertionError:
    print("TEST 1 : OLLAMA FAILED.")

print("TEST 2 : llamafile...")
launch_llamafile()
embedder = LlamafileEmbeddings()
r_2 = embedder.embed_documents(docs)
similarities_2 = cos_sim(r_2[0], r_2[1:])
print(list(similarities_2.numpy()[0]))
print("to be compared to :\n [0.7920, 0.6369, 0.1651, 0.3621]")
assert_almost_equal( similarities_2.numpy()[0], np.array([0.7920, 0.6369, 0.1651, 0.3621]),decimal=2)
print("TEST 2 : llamafile PASSED.")