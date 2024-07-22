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
from langchain_text_splitters import Language
from libs.documentloaders import CustomDocumentLoader
from utilities import getconfig
from libs.tools_llamafile import launch_llamafile

# definitions
config = "emb-llamafile"
collectionname = "buildragwithpython"
pathdata = getconfig(config)["pathdata"]
relative_path_db = getconfig(config)["dbpath"]

path_file_list = 'sourcedocs.txt'
# connect to Llamafile embedding
# Start embedding LLamafile model
launch_llamafile(config=config)
embedder = LlamafileEmbeddings(
    embed_instruction="",
    query_instruction="Represent this sentence for searching relevant passages: ",
)
# connect to chroma
chroma = chromadb.PersistentClient(path=relative_path_db)
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
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML, chunk_size=1000, chunk_overlap=0
)
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0)
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
