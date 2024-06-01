from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from utilities import list_docs_parser, readtext

class CustomDocumentLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.

        Read all content of each file of the list in one doc per files

        """
        list_docs = list_docs_parser(self.file_path)

        for file_path in list_docs:
            yield Document(
                page_content=readtext(file_path, filter_id="maincontent"),
                metadata={"source": file_path},
            )

    # alazy_load is OPTIONAL.
