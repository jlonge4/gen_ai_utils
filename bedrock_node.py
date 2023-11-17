from haystack.nodes.base import BaseComponent
from haystack.schema import Document
import numpy as np
import boto3
from langchain.embeddings import BedrockEmbeddings
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union, Tuple

bedrock = boto3.client('bedrock-runtime')
embedding_model = BedrockEmbeddings(client=bedrock)

# Custom pipeline node logic since Haystack doesnt currently support Bedrock as a retriever
class BedrockRetriever(BaseComponent):
    outgoing_edges = 1

    def __init__(self, document_store):
        self.client = bedrock
        self.embedding_model = embedding_model
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
    

class BedrockRetrieverNode(BaseComponent):
    outgoing_edges = 1

    def __init__(self, document_store, top_k):
        self.client = bedrock
        self.embedding_model = embedding_model
        self.document_store = document_store
        # self.filters = filters
        self.top_k = top_k

    def run(self, query) -> tuple[dict[str, list[Document]], str]:
        document_store = self.document_store
        embedding_model = self.embedding_model
        embedded_q = np.array([embedding_model.embed_query(query)], dtype=np.float32)
        # if filters is not None:
        #     num_docs_to_check = 50
        #     res = document_store.query_by_embedding(embedded_q, top_k=num_docs_to_check)
        # else:
        answers = self.document_store.query_by_embedding(embedded_q, top_k=3)

        output = {
            "answers": answers,
        }
        return output, "output_1"
    #         # TODO Implement filters
    #         doc_filter = {
    #             "document_name": {
    #                 "$eq": f"{filters}"
    #             }
    #         }

    #         if filters is not None:
    #             cont = []
    #             st.write(f'Using {filters} filter...')
    #             for r in res:
    #                 if r.meta['document_name'] == filters:
    #                     cont.append(r)
    #             if len(cont) == 0:
    #                 cont = [Document(content="No Context Found", meta={"document_name": "N/A", "score": 0, "page": "N/A", "sha1": "N/A"})]
    #                 return cont
    #             top_k_cont = cont[:3]
    #             return top_k_cont
    #         else:    
    #             return res

    def run_batch(self, query) -> tuple[dict[str, list[Document]], str]:
        document_store = self.document_store
        embedding_model = self.embedding_model
        embedded_q = np.array([embedding_model.embed_query(query)], dtype=np.float32)
        # if filters is not None:
        #     num_docs_to_check = 50
        #     res = document_store.query_by_embedding(embedded_q, top_k=num_docs_to_check)
        # else:
        res = self.document_store.query_by_embedding(embedded_q, top_k=3)
        answers = []
        for i in range(0, len(res)):
            answers.append(Document(content=res[i]))

        output = {
            "answers": answers,
        }
        return output, "output_1"