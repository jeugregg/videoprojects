"""
Run as a seperate process a Llamafile LLM server in local machine
- steps:
    - download Llamafile : https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile/resolve/main/mxbai-embed-large-v1-f16.llamafile?download=true
chmod +x "/Users/gregory/code/llamafile/mxbai-embed-large-v1-f16.llamafile”
/Users/gregory/code/llamafile/mxbai-embed-large-v1-f16.llamafile --server --nobrowser --embedding
"""

import os
from libs.tools_llamafile import launch_llamafile
from libs.tools_llamafile import kill_llamafile
from libs.tools_llamafile import PATH_PID
from langchain_community.embeddings import LlamafileEmbeddings

print("TEST: launch llamafile...")

launch_llamafile(config="emb-llamafile")

assert os.path.isfile(PATH_PID), "launch llamafile failed"
print("TEST: launch llamafile done.")

print("TEST EMBEDDING...")

embedder = LlamafileEmbeddings()
text = "This is a test document."
query_result = embedder.embed_query(text)
print(query_result[:5])

print("TEST EMBEDDING PASSED.")

print("TEST: kill llamafile...")

kill_llamafile()

assert os.path.isfile(PATH_PID) != True, "kill llamafile failed"
print("TEST: kill llamafile PASSED.")


# EXTERNAL process not blocking : Popen + shell True
'''process = subprocess.Popen(
    "/Users/gregory/code/llamafile/mxbai-embed-large-v1-f16.llamafile --server --nobrowser --embedding", 
    shell=True,
)
print(process.pid)
time.sleep(5)'''


# INTERNAL process : process blocked until Llamafile process is finished : call + shell True
'''retcode = subprocess.call(
    "/Users/gregory/code/llamafile/mxbai-embed-large-v1-f16.llamafile --server --nobrowser --embedding",
    shell=True,
)
print(retcode)'''

