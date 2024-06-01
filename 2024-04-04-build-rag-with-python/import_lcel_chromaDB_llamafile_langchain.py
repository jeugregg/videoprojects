'''
Import DB with embeddings with ChromaDB + Llamafile on Langchain
- pre requisites : 
  - Llamafile server running locally
  - chromaDB server running locally
'''
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import LlamafileEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from libs.documentloaders import CustomDocumentLoader

# definitions
collectionname = "buildragwithpython"
llamafilename = "/Users/gregory/code/llamafile/mxbai-embed-large-v1-f16.llamafile"
path_file_list = 'sourcedocs.txt'
# connect to Llamafile embedding
embedder = LlamafileEmbeddings()

# connect to chroma
chroma = chromadb.HttpClient(host="localhost", port=8000)
print(chroma.list_collections())
# prepare collection : delete if exist to rebluid it
if any(collection.name == collectionname for collection in chroma.list_collections()):
    print('deleting collection')
    chroma.delete_collection(collectionname)
collection = chroma.get_or_create_collection(name=collectionname, metadata={"hnsw:space": "cosine"})

# langchain chroma connection
langchain_chroma = Chroma(
    client=chroma,
    collection_name=collectionname,
    embedding_function=embedder,
)

# load docs
loader = CustomDocumentLoader(path_file_list)
docs = loader.load()

# split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0)
all_splits = text_splitter.split_documents(docs)

# load it into Chroma
langchain_chroma.add_documents(all_splits)

# Test check
print("\n\nTEST count docs in collection...")
print("There are", langchain_chroma._collection.count(), "in the collection")
assert langchain_chroma._collection.count() == len(all_splits)
print("\nTEST count docs in collection PASSED.")
print("\n\nTEST import... ")
req_docs = langchain_chroma.similarity_search("What did happen in Taiwan ?")
# print results
print(req_docs[0].page_content[:100])
assert req_docs[0].page_content.find("earthquake") != -1
print("\nTEST import PASSED.")
