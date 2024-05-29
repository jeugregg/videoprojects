'''
Test summarize long PDf on Langchain: 
- DB : ChromaDB
- LLM : Ollama
- Method : Map-Reduce : https://python.langchain.com/v0.1/docs/use_cases/summarization/#option-2-map-reduce
    ATTENTION : it is very slow.
Pre-requisites : 
- PDF file already parsed with ChromaDB : ./import_pdf_chromaDB_llamafile_langchain.py
- PDF file : ./pdf/rapport_blockchain_rh_01042024.pdf
'''


# import
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_community.llms import Ollama
from utilities import getconfig

# definitions
collectionname="buildragwithpythonPDF"
mainmodel = getconfig()["mainmodel"]

chroma = chromadb.HttpClient(host="localhost", port=8000)
langchain_chroma = Chroma(
    client=chroma,
    collection_name=collectionname,
)

llm = Ollama(model=mainmodel)

# get all docs from collection (no query here)
docs = langchain_chroma._collection.get()["documents"]
docs = [Document(doc) for doc in docs]

# 1- Map : mapping each document to an individual summary
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# 2- Reduce 
reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

# reduce chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=4000,
)

# 3- Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

# 4- Run all
response = map_reduce_chain.run(docs)

print(response)
