# Build RAG with Python

[![Watch the video](https://img.youtube.com/vi/GxLoMquHynY/maxresdefault.jpg)](https://youtu.be/GxLoMquHynY)

1. Get started by installing the requirements: `pip install -r requirements.txt`
2. Make sure you have the models listed in config.ini. so for llama3 8B, run `ollama pull llama3`
    - Update the config.ini to show whatever models you want to use.
3. Create an ollama llama3 coherent for LLM with a modelfile : `ollama create llama3-coherent -f ./ModelFileLLM`
4. Then run Vector DB ChromaDB in a separate terminal:
    - `chroma run --host localhost --port 8000 --path ../vectordb-stores/chromadb`
5. Edit the list of docs in `sourcedocs.txt`
6. For embedding docs, Ollama is buggy for embeddings in v0.1.8 so a workaround : 
    - you can use LlamFile model:
        - Download llamafile for embedding: 
            - https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile/resolve/main/mxbai-embed-large-v1-f16.llamafile?download=true
        - Make executable: `chmod +x "/somewhere/mxbai-embed-large-v1-f16.llamafile”`
        - Run: `/somewhere/mxbai-embed-large-v1-f16.llamafile --server --nobrowser --embedding`
        - Update the config.ini to show whatever models you want to use.
        - import docs into DB : run `python3 import_chromaDB_llamafile_langchain.py`
    - OR use libertai model online:
        - just use libertai libs (no need to download anything else)
        - import docs into DB : run `python3 import_chromaDB_libertai_langchain.py`

10. Test search with augmented context with relevant docs : 
    - with Llamafile : `python3 test_search_chroma_llamafile_langchain.py`
    - with Libertai  : `python3 test_search_chroma_libertai_langchain.py`
11. Test search with augmented context with relevant docs in LCEL mode (cleaner):
    - with Llamafile : `python3 test_search_lcel_llamafile_langchain.py`
    - with Libertai  : `python3 test_search_lcel_libertai_langchain.py`



Some folks on Windows seem to have issues with installing this. I guess the install package names are different there. Try `pip install python-magic-bin` to get the magic stuff working.

# Test Embedding with Ollama & Llamafile
1. Test embedding with model mxbai-embed-large-v1 : `python3 test_langchain_emb.py`

# Test Embedding with Libertai
1. Test embedding with nomic model on Libertai : `python3 test_emb_libertai.py`

