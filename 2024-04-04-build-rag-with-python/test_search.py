import ollama, sys, chromadb
from utilities import getconfig

embedmodel = getconfig()["embedmodel"]
mainmodel = getconfig()["mainmodel"]
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection("buildragwithpython")

#query = "Will iPhone 16 bring notable changes to the iPhone lineup?"
#query = "What did happen to the TSMC chip production lines?"
query = "What did happen in Taiwan about Apple Chip Production Lines?"
#query = "What did happen in Taiwan ?"
queryembed = ollama.embeddings(model=embedmodel, prompt=query)['embedding']
results = collection.query(query_embeddings=[queryembed], n_results=100)
relevantdocs = results["documents"][0][0:12]
docs = "\n\n".join(relevantdocs)
print("\n\n")
print("LENGTH of DOCS : ", len(docs))
print("\n\n")
modelquery = f"{query} - Answer that question using the following text as a resource: {docs}"
print("\n\n model query:")
print(modelquery)
print("\n\n")
print("\n\n")
for id, distance in zip(results["ids"][0], results["distances"][0]):
  print(f"ID: {id} DISTANCE: {distance}")
stream = ollama.generate(model=mainmodel, prompt=modelquery, stream=True)
print("RESPONSE : ")
print("\n\n")
for chunk in stream:
  if chunk["response"]:
    print(chunk['response'], end='', flush=True)

