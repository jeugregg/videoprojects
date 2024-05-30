# Build RAG with Python

[![Watch the video](https://img.youtube.com/vi/GxLoMquHynY/maxresdefault.jpg)](https://youtu.be/GxLoMquHynY)

1. Get started by installing the requirements: `pip install -r requirements.txt`
2. Make sure you have the models listed in config.ini. so for mxbai-embed-large, run `ollama pull mxbai-embed-large`. Update the config to show whatever models you want to use.
3. Create an ollama llama3 coherent for LLM with a modelfile : `ollama create llama3-coherent -f ./ModelFileLLM`
4. Download llamafile for embedding: 
    -https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile/resolve/main/mxbai-embed-large-v1-f16.llamafile?download=true
5. Make executable: `chmod +x "/somewhere/mxbai-embed-large-v1-f16.llamafile”`
6. Run: `/somewhere/mxbai-embed-large-v1-f16.llamafile --server --nobrowser --embedding`
7. Then run ChromaDB in a separate terminal: `chroma run --host localhost --port 8000 --path ../vectordb-stores/chromadb`
8. Edit the list of docs in `sourcedocs.txt`
9. Import the docs: `python3 import_chromaDB_llamafile_langchain.py`
10. Test with `python3 test_search_chroma_llamafile_langchain.py`
11. Perform a search: `python3 search.py <yoursearch>` TO BE UPDATED


Some folks on Windows seem to have issues with installing this. I guess the install package names are different there. Try `pip install python-magic-bin` to get the magic stuff working.


# Test Embedding with Libertai
After installed all steps above except maybe LlamaFile part (steps 4-6),
1. Test embedding with nomic model on Libertai : `test_emb_libertai.py`
