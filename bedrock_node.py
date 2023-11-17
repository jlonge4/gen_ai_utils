from haystack.nodes.base import BaseComponent
from haystack.schema import Document
import numpy as np
import boto3
from langchain.embeddings import BedrockEmbeddings
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union, Tuple

# Custom pipeline node logic since Haystack doesnt currently support Bedrock as a retriever
class BedrockEmbeddingRetriever(BaseComponent):
    outgoing_edges = 1

    def __init__(self, document_store, bedrock_client):
        self.client = bedrock_client
        self.embedding_model = BedrockEmbeddings(client=self.client)
        self.document_store = document_store

    def run(self, documents: List[Document]) -> tuple[dict[str, list[Document]], str]:
        content = [d.content for d in documents]
        metadata = [d.meta for d in documents]
        
        embeddings = self.embedding_model.embed_documents(content)
        np_embedded = np.array(embeddings, dtype=np.float32)

        docs = []

        for i in range(0, len(embeddings)):
            docs.append(Document(content=content[i], meta=metadata[i], embedding=np_embedded[i]))

        output = {
            "documents": docs,
        }
        return output, "output_1"

    def run_batch(self, documents: List[Document]) -> tuple[dict[str, list[Document]], str]:
        pass    
    

class BedrockContextRetriever(BaseComponent):
    outgoing_edges = 1

    def __init__(self, document_store, top_k, filters, bedrock_client):
        self.client = bedrock_client
        self.embedding_model = BedrockEmbeddings(client=self.client)
        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k

    def run(self, query) -> tuple[dict[str, list[Document]], str]:
        document_store = self.document_store
        filters = self.filters
        embedding_model = self.embedding_model
        embedded_q = np.array([embedding_model.embed_query(query)], dtype=np.float32)

        if filters is not None:
            num_docs_to_check = 30
            res = document_store.query_by_embedding(embedded_q, top_k=num_docs_to_check)
        else:
            res = document_store.query_by_embedding(embedded_q, top_k=self.top_k)

        if filters is not None:
            cont = []
            for r in res:
                if r.meta['document_name'] == filters:
                    cont.append(r)
            if len(cont) == 0:
                cont = [Document(content="No Context Found", meta={"document_name": "N/A", "score": 0, "page": "N/A", "sha1": "N/A"})]
                return cont
            top_k_cont = cont[:self.top_k]
            answers = top_k_cont
        else:    
            answers = res

        output = {
            "answers": answers,
        }
        return output, "output_1"
