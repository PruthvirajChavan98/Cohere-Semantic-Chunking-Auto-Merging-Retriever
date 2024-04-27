from llama_index.core.node_parser import (
    HierarchicalNodeParser, get_leaf_nodes, SemanticSplitterNodeParser
    )
from llama_index.core import (
    load_index_from_storage, StorageContext, SimpleDirectoryReader, 
    Document, VectorStoreIndex
    )
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

import os

class SemanticAutoMergingIndexManager:
    def __init__(self, rerank, embed_model):
        # self.rerank = CohereRerank(
        #     top_n=rerank_top_n, api_key=cohere_api_key
        # )
        self.rerank = rerank
        self.embed_model = embed_model

    def load_or_build_semantic_automerging_index(self, input_dir, persist_dir, chunk_sizes=None):
        chunk_sizes = chunk_sizes or [2048, 512, 128]
        splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model
        )
        h_node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)

        if os.path.exists(persist_dir):
                print(f"Loading existing index from {persist_dir}")
                try:
                    return load_index_from_storage(
                        StorageContext.from_defaults(persist_dir=persist_dir),
                    )
                except Exception as e:
                    print(f"Error loading existing index from {persist_dir}: {e}")

        print(f"Building new index at {persist_dir}")
        documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
        doc_text = "\n\n".join([d.get_content() for d in documents])
        docs = [Document(text=doc_text)]

        nodes = splitter.get_nodes_from_documents(docs, show_progress=True)
        h_nodes = h_node_parser.get_nodes_from_documents(nodes, show_progress=True)

        leaf_nodes = get_leaf_nodes(h_nodes)

        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(h_nodes)

        try:
            automerging_index = VectorStoreIndex(
                leaf_nodes, storage_context=storage_context, show_progress=True,
            )
            automerging_index.storage_context.persist(persist_dir=persist_dir)
            return automerging_index
        except Exception as e:
            print(f"Failed to build or persist new index at {persist_dir}: {e}")
            raise


    def get_retriever_and_query_engine(self, automerging_index: VectorStoreIndex, similarity_top_k=12):
        base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
        retriever = AutoMergingRetriever(
            vector_retriever=base_retriever, storage_context=automerging_index.storage_context, verbose=True
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever, node_postprocessors=[self.rerank]
        )
        return retriever, query_engine