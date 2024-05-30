'''
Test on Langchain : RAG + Query with :
    - DB: ChromaDB
    - embbedding: Llamafile 
    - LLM: Ollama
'''
# import
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import LlamafileEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utilities import getconfig

# definitions
embedmodel = getconfig()["embedmodel"]
mainmodel = getconfig()["mainmodel"]
collectionname = "buildragwithpython"

# Embedding model
embedder = LlamafileEmbeddings(model=embedmodel)
# Vector DB
chroma = chromadb.HttpClient(host="localhost", port=8000)
vectorstore = Chroma(
    client=chroma,
    collection_name=collectionname,
    embedding_function=embedder,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) # only 10 docs
# LLM model
llm = Ollama(model=mainmodel)


# TEST  : LCEL Chain : LangChain Expression Language
query = "What did happen in Taiwan ?"
# Utilisation des structures de LCEL Chain : LangChain Expression Language
print("\nTEST LCEL MODE : \n")
# Prompt with context definition
prompt = ChatPromptTemplate.from_template(
    """
    Répondez en Français à la question suivante en vous basant uniquement sur le contexte fourni:
    <context>
    {context}
    </context>
    Question: {input}
    """
)
# Stuff all relevant docs (10 in this case)
document_chain = create_stuff_documents_chain(llm, prompt)
# Add to the retriever to have the complet chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# REPONSE TEST :
results_2 = retrieval_chain.invoke({"input": query})
print(results_2["answer"])

print("\nTEST 0 ")
# what did happen in taiwan ?
# answer speak about an earthquake.
assert results_2["answer"].find("séisme") != -1
print("\nTEST 0 PASSED")
