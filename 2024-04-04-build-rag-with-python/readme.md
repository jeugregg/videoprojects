# Build RAG with Python

[![Watch the video](https://img.youtube.com/vi/GxLoMquHynY/maxresdefault.jpg)](https://youtu.be/GxLoMquHynY)

1. Get started by installing the requirements: `pip install -r requirements.txt`
2. Make sure you have the models listed in config.ini. so for mxbai-embed-large, run `ollama pull mxbai-embed-large`. Update the config to show whatever models you want to use.
3. Create an ollama llama3 coherent for testing with a modelfile : `ollama create llama3-coherent -f ./ModelFileLLM`
4. Then run ChromaDB in a separate terminal: `chroma run --host localhost --port 8000 --path ../vectordb-stores/chromadb`
5. Edit the list of docs in `sourcedocs.txt`
6. Import the docs: `python3 import.py`
7. Test with `python3 test_search.py`
8. Perform a search: `python3 search.py <yoursearch>`


Some folks on Windows seem to have issues with installing this. I guess the install package names are different there. Try `pip install python-magic-bin` to get the magic stuff working.