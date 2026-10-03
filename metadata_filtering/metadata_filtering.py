"""Metadata filtering tutorial.

Builds a small InMemoryDocumentStore with versioned Haystack docs, then
runs BM25 search combined with metadata filters (simple and AND-combined).
"""

from datetime import datetime

from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


def build_document_store() -> InMemoryDocumentStore:
    documents = [
        Document(
            content=(
                "Use pip to install a basic version of Haystack's latest release: "
                "pip install farm-haystack. All the core Haystack components live in the "
                "haystack repo. But there's also the haystack-extras repo which contains "
                "components that are not as widely used, and you need to install them "
                "separately."
            ),
            meta={"version": 1.15, "date": datetime(2023, 3, 30)},
        ),
        Document(
            content=(
                "Use pip to install a basic version of Haystack's latest release: "
                "pip install farm-haystack[inference]. All the core Haystack components "
                "live in the haystack repo. But there's also the haystack-extras repo "
                "which contains components that are not as widely used, and you need to "
                "install them separately."
            ),
            meta={"version": 1.22, "date": datetime(2023, 11, 7)},
        ),
        Document(
            content=(
                "Use pip to install only the Haystack 2.0 code: pip install haystack-ai. "
                "The haystack-ai package is built on the main branch which is an unstable "
                "beta version, but it's useful if you want to try the new features as "
                "soon as they are merged."
            ),
            meta={"version": 2.0, "date": datetime(2023, 12, 4)},
        ),
    ]

    document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
    document_store.write_documents(documents=documents)
    return document_store


def build_pipeline(document_store: InMemoryDocumentStore) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component(
        instance=InMemoryBM25Retriever(document_store=document_store), name="retriever"
    )
    return pipeline


def print_results(label: str, result: dict) -> None:
    print(f"\n=== {label} ===")
    documents = result["retriever"]["documents"]
    if not documents:
        print("(no documents matched the filter)")
    for doc in documents:
        print(f"- version={doc.meta['version']} date={doc.meta['date'].date()} score={doc.score:.4f}")
        print(f"  {doc.content}")


def main() -> None:
    document_store = build_document_store()
    pipeline = build_pipeline(document_store)

    query = "Haystack installation"

    # Simple comparison filter: version > 1.21
    result = pipeline.run(
        data={
            "retriever": {
                "query": query,
                "filters": {"field": "meta.version", "operator": ">", "value": 1.21},
            }
        }
    )
    print_results("version > 1.21", result)

    # Logical AND of two conditions
    result = pipeline.run(
        data={
            "retriever": {
                "query": query,
                "filters": {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.version", "operator": ">", "value": 1.21},
                        {"field": "meta.date", "operator": ">", "value": datetime(2023, 11, 7)},
                    ],
                },
            }
        }
    )
    print_results("version > 1.21 AND date > 2023-11-07", result)


if __name__ == "__main__":
    main()
