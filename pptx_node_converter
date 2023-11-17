from haystack.nodes.base import BaseComponent
from haystack.schema import Document
from langchain.document_loaders import UnstructuredPowerPointLoader
from typing import Tuple, List, Optional, Any, Dict
from pathlib import Path

class PptxConverter(BaseComponent):
  outgoing_edges = 1

  def __init__(self):
    pass

  def run(self, file_paths: Path, meta: dict) -> tuple[dict[str, lst[Document]], str]:
    loader = UnstructuredPowerPointLoader(file_paths)
    text = loader.load()
    document = Document(content=text[0].page_content, meta=meta)
    output = {
      "documents": document
    }
    return output, "output_1"  
