'''
Test on Langchain : RAG + Query on PDF DB with :
    - DB: ChromaDB
    - embbedding: Llamafile 
    - LLM: Ollama

1)  Link to DB with embeddings with ChromaDB + Llamafile on Langchain
2)  Search on DB about a query
3)  Stuff documents as context and query Ollama LLM on Langchain
'''
# import
import chromadb
from langchain_community.embeddings import LlamafileEmbeddings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from utilities import getconfig

# definitions
collectionname = "buildragwithpythonPDF"
embedmodel = getconfig()["embedmodel"]
mainmodel = getconfig()["mainmodel"]
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection(collectionname)
embedder = LlamafileEmbeddings(model=embedmodel)
langchain_chroma = Chroma(
    client=chroma,
    collection_name=collectionname,
    embedding_function=embedder,
)

query = "Quels sont les besoins dans le domaine des pratiques RH que la blockchain adresse ou pourrait adresser ?" 

results = langchain_chroma.similarity_search_with_score(query, k=10)

relevantdocs = [doc[0].page_content for doc in results]
context = "\n\n".join(relevantdocs)
print("\n\n")
print("LENGTH of DOCS : ", len(context))
print("\n\n")
modelquery = f'{query} - Repond à la question en utilisant les ressources suivantes:\n"""\n{context}\n"""'
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
