"""Hybrid retrieval tutorial: BM25 + dense embeddings + cross-encoder reranking.

Indexes a subset of the PubMed abstracts dataset into an
InMemoryDocumentStore with embeddings, then builds a hybrid retrieval
pipeline that runs BM25 and embedding-based retrieval in parallel and
merges/reranks the results with a cross-encoder similarity ranker.

Fix vs. the original tutorial: the tutorial hardcodes
`device=ComponentDevice.from_str("cuda:0")`, which only works on machines
with an NVIDIA GPU. This script lets Haystack auto-detect the best
available device (CUDA, Apple Silicon MPS, or CPU) instead, so it runs on
any machine. Set HYBRID_DEVICE to force a specific device (e.g. "cpu",
"mps", "cuda:0").
"""

import os

from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import ComponentDevice
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.rankers.sentence_transformers import SentenceTransformersSimilarityRanker

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RANKER_MODEL = os.environ.get("RANKER_MODEL", "BAAI/bge-reranker-base")

# Full dataset is ~60k rows; default to a small slice so the tutorial runs
# quickly out of the box. Set SAMPLE_SIZE=0 to index the full dataset.
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "500"))


def resolve_device() -> ComponentDevice:
    forced = os.environ.get("HYBRID_DEVICE")
    if forced:
        return ComponentDevice.from_str(forced)
    return ComponentDevice.resolve_device(None)


def load_documents() -> list[Document]:
    split = "train" if SAMPLE_SIZE <= 0 else f"train[:{SAMPLE_SIZE}]"
    dataset = load_dataset("anakin87/medrag-pubmed-chunk", split=split)

    docs = []
    for doc in dataset:
        docs.append(
            Document(
                content=doc["contents"],
                meta={"title": doc["title"], "abstract": doc["content"], "pmid": doc["id"]},
            )
        )
    return docs


def build_indexing_pipeline(document_store: InMemoryDocumentStore, device: ComponentDevice) -> Pipeline:
    document_splitter = DocumentSplitter(split_by="word", split_length=512, split_overlap=32)
    document_embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL, device=device)
    document_writer = DocumentWriter(document_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("document_splitter", document_splitter)
    indexing_pipeline.add_component("document_embedder", document_embedder)
    indexing_pipeline.add_component("document_writer", document_writer)

    indexing_pipeline.connect("document_splitter", "document_embedder")
    indexing_pipeline.connect("document_embedder", "document_writer")

    return indexing_pipeline


def build_hybrid_retrieval_pipeline(document_store: InMemoryDocumentStore, device: ComponentDevice) -> Pipeline:
    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL, device=device)
    embedding_retriever = InMemoryEmbeddingRetriever(document_store)
    bm25_retriever = InMemoryBM25Retriever(document_store)
    ranker = SentenceTransformersSimilarityRanker(model=RANKER_MODEL, device=device)

    hybrid_retrieval = Pipeline()
    hybrid_retrieval.add_component("text_embedder", text_embedder)
    hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
    hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
    hybrid_retrieval.add_component("ranker", ranker)

    hybrid_retrieval.connect("text_embedder", "embedding_retriever")
    hybrid_retrieval.connect("bm25_retriever", "ranker")
    hybrid_retrieval.connect("embedding_retriever", "ranker")

    return hybrid_retrieval


def pretty_print_results(prediction: dict) -> None:
    for doc in prediction["documents"]:
        print(doc.meta["title"], "\t", doc.score)
        print(doc.meta["abstract"])
        print("\n")


def main() -> None:
    device = resolve_device()
    print(f"Using device: {device}")

    document_store = InMemoryDocumentStore()

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    indexing_pipeline = build_indexing_pipeline(document_store, device)
    indexing_pipeline.run({"document_splitter": {"documents": docs}})
    print(f"Indexed {document_store.count_documents()} chunks")

    hybrid_retrieval = build_hybrid_retrieval_pipeline(document_store, device)

    query = os.environ.get("QUERY", "apnea in infants")
    result = hybrid_retrieval.run(
        {"text_embedder": {"text": query}, "bm25_retriever": {"query": query}, "ranker": {"query": query}}
    )
    pretty_print_results(result["ranker"])


if __name__ == "__main__":
    main()
