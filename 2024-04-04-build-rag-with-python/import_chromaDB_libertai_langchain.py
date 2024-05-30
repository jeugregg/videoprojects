'''
Import DB with embeddings with ChromaDB + Libertai on Langchain

'''
# imports
import chromadb, time
from langchain_chroma import Chroma
from libs.libertai import LibertaiEmbeddings
from utilities import readtext
from mattsollamatools import chunk_text_by_sentences

# definitions
collectionname = "buildragwithpython_libertai"


embedder = LibertaiEmbeddings()

chroma = chromadb.HttpClient(host="localhost", port=8000)
print(chroma.list_collections())
if any(collection.name == collectionname for collection in chroma.list_collections()):
    print('deleting collection')
    chroma.delete_collection(collectionname)

collection = chroma.get_or_create_collection(name=collectionname, metadata={"hnsw:space": "cosine"})

langchain_chroma = Chroma(
    client=chroma,
    collection_name=collectionname,
    embedding_function=embedder,
)

starttime = time.time()
with open('sourcedocs.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for filename in lines:
        text = readtext(filename.strip(), filter="maincontent")
        chunks = chunk_text_by_sentences(source_text=text, sentences_per_chunk=2, overlap=0)
        print(f"with {len(chunks)} chunks")
        for index, chunk in enumerate(chunks):
            embed = embedder.embed_query(chunk)

            print(".", end="", flush=True)
            filename = filename.replace("\n","")
            curr_id = f"{filename} ({index})"
            langchain_chroma._collection.add(
                ids=[curr_id],
                embeddings=[embed],
                documents=[chunk],
                metadatas={"source": filename}
            )

print("--- %s seconds ---" % (time.time() - starttime))
