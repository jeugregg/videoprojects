'''
Test on Langchain : RAG + Query on PDF DB with :
    - DB: ChromaDB
    - embbedding: Llamafile 
    - LLM: Ollama

1)  Link to DB with embeddings with ChromaDB + Llamafile on Langchain
2)  Search on DB about a query
3)  Stuff documents as context and query Ollama LLM on Langchain

TEST 1 : QA Chain (simple) 
TEST 2 : LCEL Chain : LangChain Expression Language (chaine globale)
TEST 3 : LLM only

https://python.langchain.com/v0.1/docs/get_started/quickstart/#retrieval-chain
'''
# import
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import LlamafileEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from utilities import getconfig

# tools
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# definitions
collectionname = "buildragwithpythonPDF"
embedmodel = getconfig()["embedmodel"]
mainmodel = getconfig()["mainmodel"]
chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection(collectionname)

embedder = LlamafileEmbeddings(model=embedmodel)

vectorstore = Chroma(
    client=chroma,
    collection_name=collectionname,
    embedding_function=embedder,
)

llm = Ollama(model=mainmodel)


# TEST 1 : QA CHAIN
# Prompt
rag_prompt = PromptTemplate.from_template(
    '''
    [INST]<<SYS>> Tu es un assistant pour les tâches de réponses aux questions. 
    Utilisez les éléments de contexte récupérés suivants pour répondre à la question. 
    Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.
    <</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]"
    '''
)

# Chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)


# REPONSE TEST 1: 

question = "Quels sont les besoins dans le domaine des pratiques RH que la blockchain adresse ou pourrait adresser ?"
results_1 = qa_chain.invoke(question)

print(results_1)

# TEST 2 : LCEL Chain : LangChain Expression Language

# Utilisation des structures de LCEL Chain : LangChain Expression Language

print("\nTEST LCEL MODE : \n")


prompt = ChatPromptTemplate.from_template(
    """
    Répondez en Français à la question suivante en vous basant uniquement sur le contexte fourni:
    <context>
    {context}
    </context>
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

# REPONSE TEST 2: 
results_2 = retrieval_chain.invoke({"input": question})
print(results_2["answer"])

# REPONSE TEST 3: 
print("\n\n REPONSE LLM only : \n")

results_3 = llm.invoke("Quels sont les besoins dans le domaine des pratiques RH que la blockchain adresse ou pourrait adresser ?")

print(results_3)

print("END")
