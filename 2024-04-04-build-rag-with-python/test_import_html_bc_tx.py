"""
Global example of parsing HTML to extract information.
here we use ChromaDB + Llamafile + Langchain
- We have a report from polygonscan about a blockchain transaction 
   - multi wallet from / to 
   - multi tokens are distributed
- The task is to find out what happened in the transaction : 
    - to or from a given wallet address
    - which tokens were transfered

- DONE : 
    - What are the tokens exchanged with the wallet ?

- Used different methods to import an HTML file

"""
import os, time
from langchain_community.document_loaders import WebBaseLoader
from utilities import getconfig
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
#from selenium.webdriver.common.by import By
#from selenium.webdriver.support.wait import WebDriverWait
#from selenium.webdriver.support import expected_conditions as EC
#from dotenv import load_dotenv, find_dotenv
#from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import LlamafileEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List
from libs.tools_ollama import launch_server_ollama
from libs.tools_llamafile import launch_llamafile

# definitions
config = "emb-llamafile"
pathdata = getconfig(config)["pathdata"]
relative_path_db = getconfig(config)["dbpath"]
collectionname = "tx_test"
url_test = "https://polygonscan.com/tx/0x99e3c197172b967eb4215249be50034a1696423a9ae805438ae217a501d86aa9"
file_path_test = "content/file_test_polygonscan.html" # local download of remote  HTML file
address_test = "0x8da02d597a2616e9ec0c82b2b8366b00d69da29a" # address of the wallet to scan
# tokens to find
dict_tokens_to_find = {
    "FOMO":  "0x44a6e0be76e1d9620a7f76588e4509fe4fa8e8c8",
    "FUD": "0x403e967b044d4be25170310157cb1a4bf10bdd0f", 
    "KEK": "0x42e5e06ef5b90fe15f853f59299fc96259209c5c",
    "ALPHA": "0x6a3E7C3c6EF65Ee26975b12293cA1AAD7e1dAeD2",
}

# Start Ollama model : actually, check if it exists or pull it and prepare it 
# cf. config.ini
mainmodel = launch_server_ollama(config=config)
# Start embedding LLamafile model
launch_llamafile(config=config)

# Web text directly : NOK : html tag are missing so understanding is difficult (bs get_text is used)
'''loader = WebBaseLoader(url_test)
data = loader.load()
data_split = data[0].page_content.lower().split(address_test[:10])
for k, d in enumerate(data_split):
    print(f"\nsplit n°{k} : {d[:100]}")'''
# test with unstructured : NOK : header missing ?
'''from langchain_community.document_loaders import UnstructuredURLLoader
loader = UnstructuredURLLoader(urls=[url_test])
data = loader.load()
print(data[0].page_content)
print("END")'''
# test with unstructured : NOK : header missing ?  HTML ERROR 403
'''from unstructured.partition.html import partition_html
elements = partition_html(url=url_test)'''
# test with utilies download_file classical : NOK :  header missing ? HTML ERROR 403
'''download_file(url_test)'''
# test selenium : NOK : load url and download it with langchain selenium document loader
'''from langchain_community.document_loaders import SeleniumURLLoader
loader = SeleniumURLLoader(urls=[url_test])
docs = loader.load()'''


# Download with BeautyfulSoup
if not os.path.isdir(pathdata):
    os.mkdir(pathdata)

if not os.path.isfile(file_path_test):
    print("Loading url : ", url_test)
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    # get html content from chrome driver
    driver.get(url_test)
    # parse final content
    content = driver.page_source
    soup = BeautifulSoup(content, "html.parser")
    # save html
    with open(file_path_test, "w", encoding="utf-8") as file:
        file.write(str(soup))
    # close browser
    driver.close()
    driver.quit()
else:
    # reload file
    print("Loading file : ", file_path_test)

# load text from HTML file on disk (without tags) : NOK
"""from langchain_community.document_loaders import UnstructuredFileLoader
loader = UnstructuredFileLoader(file_path_test)
docs = loader.load()"""


# load text from HTML file on disk (with tags) : 2500 chars to have one tx
# https://python.langchain.com/v0.2/docs/how_to/code_splitter/
loader = TextLoader(file_path_test)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML, chunk_size=2500, chunk_overlap=0
)
all_splits = text_splitter.split_documents(docs)


# load it into Chroma if not already done
# connect to Llamafile embedding
embedder = LlamafileEmbeddings()

# Connect to VectorStore
#client = chromadb.HttpClient(host="localhost", port=8000) # TO SERVER (TEST 04 : 85s)
client = chromadb.PersistentClient(path=relative_path_db) # TO LOCAL PERSISTENT DB (TEST 04 : 85s same result)

if any(collection.name == collectionname for collection in client.list_collections()):
    print("docs already in collection")
    mode_add = False
else:
    mode_add = True

# langchain chroma connection
vectorstore = Chroma(
    client=client,
    collection_name=collectionname,
    embedding_function=embedder,
)

if mode_add:
    vectorstore.add_documents(all_splits) # VERY LONG ? 
# query it
#query = "Which tokens were transfered in this transaction?"
#query = "Which tokens are transfered from or to this address '0x8da02d597a2616e9ec0c82b2b8366b00d69da29a'?"
#query = "Find all tokens transfered to this address '0x8da02D59...0d69da29A'. Outputs only the name and address of these tokens and nothing else."
#query = "Find tokens, their names and adresses, that were exchanged with this wallet '0x8da02D59...0d69da29A' without the receiver or destination wallet addresses."
#query = "Find tokens, their names and adresses, that were exchanged with this wallet '0x8da02d597a2616e9ec0c82b2b8366b00d69da29a' without the receiver or destination wallet addresses."
query = f"Find tokens, their symbols and adresses, that were exchanged with this wallet '{address_test}' without the receiver or destination wallet addresses."

search_kwargs = {
    "k": 10,
    "where_document": {"$contains": address_test[:10]},
}
retriever = vectorstore.as_retriever(search_kwargs=search_kwargs) # only 10 docs
# LLM model
llm = Ollama(model=mainmodel, num_ctx=8000)


# TEST 0 :  try to extract FUD, FOMO, KEK, ALPHA from the context
#  but without specific output format
print('\nTEST 0 : with chat model : \n')
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the context provided:
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
results = retrieval_chain.invoke({"input": query})
print(results["answer"])
print("\nTEST 0 DONE.\n")

print("\nTEST 1 : CHECK CONTEXT :\n")
# check context
for k, doc in enumerate(results["context"]):
    if doc.page_content.find("FUD") != -1:
        print("FOUND FUD in context   : ", k)
    if doc.page_content.find("FOMO") != -1:
        print("FOUND FOMO in context  : ", k)
    if doc.page_content.find("KEK") != -1:
        print("FOUND KEK in context   : ", k)
    if doc.page_content.find("ALPHA") != -1:
        print("FOUND ALPHA in context : ", k)

# TEST 3 : With output format + 1 example and the last 2 context found : json
print("\nTEST 3 : With output format on 2 context found and an example : \n")
# prepare an example :
results_old = results
context_ref = results_old["context"][-5].page_content

for token, address_token in dict_tokens_to_find.items():
    if context_ref.find(token) != -1:
        print("FOUND TOKEN : ", token )
        if context_ref.find(address_token) != -1:
            print("FOUND ADDRESS : ", address_token )
            example = f"``` {context_ref} ```"
            output_example = """
            ```json
            {
            "symbol": [""" + '"' + token + '"' + """],
            "address": [""" + '"' + address_token + '"' + """]
            }
            ```
            """
            break

# try to extract for all tx found in all context

# Define your desired data structure.
class TokenData(BaseModel):
    symbol: str = Field(description="symbol of the token found")
    address: str = Field(description="address of the token found")
    # You can add custom validation logic easily with Pydantic.
    @validator("address")
    def address_start_with_0x(cls, field):
        """Check address format"""
        if field[:2] != "0x":
            raise ValueError("Badly formed address!")
        return field

class ListTokenData(BaseModel):
    tokens: List[TokenData] = Field(description="list of tokens found")

# Set up a parser + inject instructions into the prompt template.
# prompt without example
query_2 = f"Find tokens ticker symbol and address that were exchanged with this wallet '{address_test}' ?"
parser = PydanticOutputParser(pydantic_object=ListTokenData)
prompt_2 = PromptTemplate(
    template="""Answer the following question only based on the context (several pieces of a HTML file) and instructions provided.
        <context>
        {context}
        </context>
        <instructions>
        {format_instructions}
        </instructions>
        <question>
         {query}
         Don't take wallet address of sender (From) or of receiver (To), but only token address.
         A ticker symbol of a token is composed of 3 to 5 characters and use exclusively capital letters of the Latin alphabet (A-Z).
         An address of a token is composed of 42 characters and start with 0x
         A token have only one address, so output only one address per token.
         As answer, for each token found, can you give his ticker symbol and his address. 
         Do not explain how you have done.
        </question>
        """,
    input_variables=["query", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
dict_param_prompt_2 = {
    "context": results_old["context"][-2].page_content + results_old["context"][-1].page_content,
    "query": query_2,
}

# prompt with example
'''prompt_2 = PromptTemplate(
    template="""Answer the following question only based on the context (parts of a HTML file) and instructions provided.
        To help you, use the given example but not extract any data from it as output.
        <context>
        {context}
        </context>
        <example>
        This example of html file part:
        {example}
        gives the output: 
        {output_example}
        </example>
        <instructions>
        {format_instructions}
        </instructions>
        <question>
         {query}
         
         As answer I need only short symbols of token like (and not the long names) and addresses of token found. Do not explain how you have done.
        </question>
        """,
    input_variables=["query", "example","output_example","context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
dict_param_prompt_2 = {
    "context": results_old["context"][-2].page_content + results_old["context"][-1].page_content,
    "example": example,
    "output_example": output_example,
    "query": query,
}'''
# And a query intended to prompt a language model to populate the data structure.
#prompt_and_model = prompt_ex | llm
#output = prompt_and_model.invoke(dict_param_prompt_ex)

prompt_and_model = prompt_2 | llm
output = prompt_and_model.invoke(dict_param_prompt_2)
print("\noutput to parser:\n", output)
res = parser.invoke(output)
print("\nAnswer parsed : \n")
print(res)

#res = llm.invoke(prompt.invoke({"context": results_old["context"][-1].page_content, "input": query}))
#print(res)
# try several docs in context :
#print(llm.invoke(prompt.invoke({"context": results_old["context"][-3].page_content+results_old["context"][-2].page_content+results_old["context"][-1].page_content , "input": query})))
print("\nTEST 3 END")

# TEST 4 : With output format on all context found : json
print("\nTEST 4 : With output format on all context found and an example : \n")
starttime = time.time()
'''class DictToken(dict):
    """
    A dict that can be updated without mutating the original dict.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
    def UpdateToken(self, my_res):
        for symbol, address in zip(my_res.symbol, my_res.address):
            if symbol not in self:
                self[symbol] = address'''

# declaration
dict_token_found = {}
# loop over contexts
for k, doc in enumerate(results["context"]):
    print('\nContext n# ', k)
    # find token symbol and address
    dict_param_prompt_2["context"] = doc.page_content
    output = prompt_and_model.invoke(dict_param_prompt_2)
    print("1-Output to parser:\n", output)
    try:
        res = parser.invoke(output)
        print("2-Answer parsed:\n")
        print(res)
        for token in res.tokens:
            if token.symbol not in dict_token_found:
                dict_token_found[token.symbol] = token.address
    except:
        print("2-NO MORE token found ?")
print("--- %s seconds ---" % (time.time() - starttime))
print(dict_token_found)

print("\nTEST 4 END")

# test with Unstructured
#TODO
'''print("Loading UnstructuredHTMLLoader")
loader = UnstructuredHTMLLoader(
    file_path_test,
    mode="elements",
) '''