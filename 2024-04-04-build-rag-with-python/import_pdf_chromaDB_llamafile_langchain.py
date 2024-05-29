'''
Import DB with embeddings with ChromaDB + Llamafile on Langchain

'''

import chromadb, time
from langchain_chroma import Chroma
from langchain_community.embeddings import LlamafileEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from utilities import readtext
from mattsollamatools import chunk_text_by_sentences

collectionname="buildragwithpythonPDF"
llamafilename = "/Users/gregory/code/llamafile/mxbai-embed-large-v1-f16.llamafile"
file_pdf = "./pdf/rapport_blockchain_rh_01042024.pdf"
embedder = LlamafileEmbeddings()

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

loader = PyPDFLoader(file_pdf)
pages = loader.load_and_split()
print(pages[0])

print(pages[1])




for k, page in enumerate(pages):
    text = page.page_content
    chunks = chunk_text_by_sentences(source_text=text, sentences_per_chunk=5, overlap=0)
    print(f"with {len(chunks)} chunks")
    for index, chunk in enumerate(chunks):
        embed = embedder.embed_query(chunk)
        print(".", end="", flush=True)
        curr_id = f"page {k} / {index}"
        langchain_chroma._collection.add(
          ids=[curr_id],
          embeddings=[embed],
          documents=[chunk],
          metadatas={"source": file_pdf,
                     "id": curr_id,
                    }
        )

print("--- %s seconds ---" % (time.time() - starttime))
