'''
Test on Langchain : RAG + Query with :
    - DB: ChromaDB
    - embbedding: Llamafile 
    - LLM: Ollama
'''
# import
import chromadb
from langchain_community.embeddings import LlamafileEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from utilities import getconfig

# definitions
embedmodel = getconfig()["embedmodel"]
mainmodel = getconfig()["mainmodel"]
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection("buildragwithpython")
embedder = LlamafileEmbeddings(model=embedmodel)
langchain_chroma = Chroma(
    client=chroma,
    collection_name="buildragwithpython",
    embedding_function=embedder,
)
print("There are", langchain_chroma._collection.count(), "in the collection")


query = "What did happen in Taiwan ?"

results = langchain_chroma.similarity_search_with_score(query, k=10)


relevantdocs = [doc[0].page_content for doc in results]
context = "\n\n".join(relevantdocs)
print("\n\n")
print("LENGTH of DOCS : ", len(context))
print("\n\n")
modelquery = f'{query} - Answer that question using the following text as a resource:\n"""\n{context}\n"""'
print("\n\n model query:")
print(modelquery)
print("\n\n")
print("\n\n")
sum_len = 0
for k, elem in enumerate(results):
    length_curr = len(elem[0].page_content)
    distance = elem[1]
    sum_len+=length_curr
    print(f'k:{k} DISTANCE:{distance} length:{length_curr}/{sum_len} ID:{elem[0].metadata["source"]}')


llm = Ollama(model=mainmodel)
stream = llm.stream(modelquery)

print("RESPONSE : ")
print("\n\n")
answer_0 =""
for chunk in stream:
    if chunk:
        print(chunk, end='', flush=True)
        answer_0 += chunk

print("\nTEST 0 ")
# what did happen in taiwan ?
# answer speak about an earthquake.
assert answer_0.find("earthquake") != -1
print("\nTEST 0 PASSED")

print('\nTEST 1 ... ')
query_1 = "Will iPhone 16 bring notable changes to the iPhone lineup?"
print("Question: ", query_1)
results_1 = langchain_chroma.similarity_search_with_score(query_1, k=10)
relevantdocs_1 = [doc[0].page_content for doc in results_1]
context_1 = "\n\n".join(relevantdocs_1)
modelquery_1 = f'{query_1} - Answer that question using the following text as a resource:\n"""\n{context_1}\n"""'
stream_1 = llm.stream(modelquery_1)
print("RESPONSE : ")
print("\n\n")
answer_1 =""
for chunk in stream_1:
    if chunk:
        print(chunk, end='', flush=True)
        answer_1 += chunk


assert answer_1.find("Larger") != -1
assert answer_1.find("Capture") != -1
assert answer_1.find("Faster") != -1
assert answer_1.find("Camera") != -1
assert answer_1.find("Action") != -1

print("\nTEST 1 PASSED")
#query = "What did happen to the TSMC chip production lines?"
#query = "What did happen in Taiwan about Apple Chip Production Lines?"