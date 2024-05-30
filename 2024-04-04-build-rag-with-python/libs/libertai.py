"""
Libertai LLM model adaptation for LangChain

"""


from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests

# NeuralBeagle 7B:neuralbeagle14-7b.Q5_K_M.gguf : 
API_BASE = "https://curated.aleph.cloud/vm/a8b6d895cfe757d4bc5db9ba30675b5031fe3189a99a14f13d5210c473220caf"
# LLAMA 3 8B: Hermes-2-Pro-Llama-3-8B-Q5_K_M.gguf : 
# API_BASE = "https://curated.aleph.cloud/vm/84df52ac4466d121ef3bb409bb14f315de7be4ce600e8948d71df6485aa5bcc3" 
class Libertai(LLM):
    api_base: str = API_BASE
    temperature: float = 0.9
    top_p: float = 1
    top_k: int = 40
    cache_prompt: bool = True
    max_length: int = 100
    slot_id: int = -1
    session: requests.Session = None
    stop: list = ["<|"]

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM."""
        if self.session is None:
            self.session = requests.Session()

        params = {
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "n": 1,
            "n_predict": self.max_length,
            "stop": stop or self.stop,
            "slot_id": self.slot_id,
            "cache_prompt": True
        }
        print(params)
        response = requests.post(f"{self.api_base}/completion", json=params)
        if response.status_code == 200:
            output = response.json()
            print(output)
            #self.slot_id = output['slot_id']
            return output['content']
        else:
            return None

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "api_base": self.api_base,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "cache_prompt": self.cache_prompt,
            "max_length": self.max_length,
            }

#openhermes = Libertai(api_base=API_BASE)

import logging
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel

logger = logging.getLogger(__name__)


class LibertaiEmbeddings(BaseModel, Embeddings):
    """
    Libertai Embeddings from : 
    - default model : nomic-embed-text-v1.5.f16.gguf
    - and Llamafile class from langchain : 
        - https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/embeddings/llamafile.py

    Llamafile lets you distribute and run large language models with a
    single file.

    To get started, see: https://github.com/Mozilla-Ocho/llamafile

    To use this class, you will need to first:

    1. Download a llamafile.
    2. Make the downloaded file executable: `chmod +x path/to/model.llamafile`
    3. Start the llamafile in server mode with embeddings enabled:

        `./path/to/model.llamafile --server --nobrowser --embedding`

    Example:
        .. code-block:: python

            from langchain_community.embeddings import LlamafileEmbeddings
            embedder = LlamafileEmbeddings()
            doc_embeddings = embedder.embed_documents(
                [
                    "Alpha is the first letter of the Greek alphabet",
                    "Beta is the second letter of the Greek alphabet",
                ]
            )
            query_embedding = embedder.embed_query(
                "What is the second letter of the Greek alphabet"
            )

    """

    # default model : nomic-embed-text-v1.5.f16.gguf
    base_url: str = "https://curated.aleph.cloud/vm/ee1b2a8e5bd645447739d8b234ef495c9a2b4d0b98317d510a3ccf822808ebe5"
    """Base url where the llamafile server is listening."""

    request_timeout: Optional[int] = None
    """Timeout for server requests"""

    def _embed(self, text: str) -> List[float]:
        try:
            response = requests.post(
                url=f"{self.base_url}/embedding",
                headers={
                    "Content-Type": "application/json",
                },
                json={
                    "content": text,
                },
                timeout=self.request_timeout,
            )
        except requests.exceptions.ConnectionError:
            raise requests.exceptions.ConnectionError(
                f"Could not connect to Llamafile server. Please make sure "
                f"that a server is running at {self.base_url}."
            )

        # Raise exception if we got a bad (non-200) response status code
        response.raise_for_status()

        contents = response.json()
        if "embedding" not in contents:
            raise KeyError(
                "Unexpected output from /embedding endpoint, output dict "
                "missing 'embedding' key."
            )

        embedding = contents["embedding"]

        # Sanity check the embedding vector:
        # Prior to llamafile v0.6.2, if the server was not started with the
        # `--embedding` option, the embedding endpoint would always return a
        # 0-vector. See issue:
        # https://github.com/Mozilla-Ocho/llamafile/issues/243
        # So here we raise an exception if the vector sums to exactly 0.
        if sum(embedding) == 0.0:
            raise ValueError(
                "Embedding sums to 0, did you start the llamafile server with "
                "the `--embedding` option enabled?"
            )

        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a llamafile server running at `self.base_url`.
        llamafile server should be started in a separate process before invoking
        this method.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        doc_embeddings = []
        for text in texts:
            doc_embeddings.append(self._embed(text))
        return doc_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a llamafile server running at `self.base_url`.
        llamafile server should be started in a separate process before invoking
        this method.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._embed(text)