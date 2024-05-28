'''
Import DB with embeddings with ChromaDB + Llamafile on Langchain

'''

import ollama, chromadb, time
from utilities import readtext, getconfig
from mattsollamatools import chunker, chunk_text_by_sentences

collectionname="buildragwithpython"

chroma = chromadb.HttpClient(host="localhost", port=8000)
print(chroma.list_collections())
if any(collection.name == collectionname for collection in chroma.list_collections()):
  print('deleting collection')
  chroma.delete_collection("buildragwithpython")
collection = chroma.get_or_create_collection(name="buildragwithpython", metadata={"hnsw:space": "cosine"})

embedmodel = getconfig()["embedmodel"]
starttime = time.time()
with open('sourcedocs.txt', 'r', encoding='utf-8') as f:
  lines = f.readlines()
  for filename in lines:
      text = readtext(filename.strip(), filter="maincontent")
      chunks = chunk_text_by_sentences(source_text=text, sentences_per_chunk=2, overlap=0)
      print(f"with {len(chunks)} chunks")
      for index, chunk in enumerate(chunks):
          embed = ollama.embeddings(model=embedmodel, prompt=chunk)['embedding']
          print(".", end="", flush=True)
          filename = filename.replace("\n","")
          curr_id = f"{filename} ({index})"
          collection.add(
            ids=[curr_id],
            embeddings=[embed],
            documents=[chunk],
            metadatas={"source": filename}
          )
    
print("--- %s seconds ---" % (time.time() - starttime))

