'''
Test on Langchain : RAG + Query with Embedding ChromaDB + Ollama 
TODO : update ollama because it's not working : issue https://github.com/ollama/ollama/issues/4207 
with update availabe on ollama
'''

# import
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from libs.tools_ollama import launch_server_ollama
from utilities import getconfig

# definitions
config = "main"
embedmodel = getconfig(config)["embedmodel"]
#mainmodel = getconfig(config)["mainmodel"]
collectionname = "buildragwithpython"
relative_path_db = getconfig(config)["dbpath"]

# start ChromaDB
chroma = chromadb.PersistentClient(path=relative_path_db)
collection = chroma.get_or_create_collection(collectionname)
# load embedding mdl
#embedder = OllamaEmbeddings(model=embedmodel)
embedder = OllamaEmbeddings(
    model=embedmodel,
    embed_instruction="",
    query_instruction="Represent this sentence for searching relevant passages: ",
)
# load llm mdl
mainmodel = launch_server_ollama(config=config)
llm = Ollama(model=mainmodel, num_ctx=8000)
# create retriever
langchain_chroma = Chroma(
    client=chroma,
    collection_name="buildragwithpython",
    embedding_function=embedder,
)
print("There are", langchain_chroma._collection.count(), "in the collection")

query = "What did happen in Taiwan ?"

results = langchain_chroma.similarity_search_with_score(query, k=5)
#queryembed = ollama.embeddings(model=embedmodel, prompt=query)['embedding']
#results = collection.query(query_embeddings=[queryembed], n_results=100)
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
    sum_len += length_curr
    print(f'k:{k} DISTANCE:{distance} length:{length_curr}/{sum_len} ID:{elem[0].metadata["source"]}')

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
results_1 = langchain_chroma.similarity_search_with_score(query_1, k=6)
relevantdocs_1 = [doc[0].page_content for doc in results_1]
context_1 = "\n\n".join(relevantdocs_1)
modelquery_1 = f'{query_1} - Answer that question in minimum of 5 bullets  and using the following text as a resource:\n"""\n{context_1}\n"""'
print("\n\nMODEL QUERY 1: ")
print(modelquery_1)
print("\nmodel query size: ", len(modelquery_1))
print("\n")
sum_len = 0
for k, elem in enumerate(results_1):
    length_curr = len(elem[0].page_content)
    distance = elem[1]
    sum_len+=length_curr
    print(f'k:{k} DISTANCE:{distance} length:{length_curr}/{sum_len} ID:{elem[0].metadata["source"]}')

stream_1 = llm.stream(modelquery_1)
print("RESPONSE 1: ")
print("\n\n")
answer_1 =""
for chunk in stream_1:
    if chunk:
        print(chunk, end='', flush=True)
        answer_1 += chunk


assert answer_1.lower().find("larger") != -1
assert (answer_1.lower().find("faster") != -1) | (answer_1.lower().find("performance") != -1)
assert answer_1.lower().find("camera") != -1
assert answer_1.lower().find("capture") != -1
assert (answer_1.lower().find("action") != -1) | (answer_1.lower().find("button") != -1)

print("\nTEST 1 PASSED")