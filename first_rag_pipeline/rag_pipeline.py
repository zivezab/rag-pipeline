"""First RAG pipeline tutorial: Seven Wonders of the Ancient World Q&A.

Indexes the "Seven Wonders" dataset into an InMemoryDocumentStore and
answers questions about it using a retrieval-augmented generation (RAG)
pipeline built from Haystack components.

The chat generator (the LLM that produces the final answer) is
configurable via the LLM_PROVIDER environment variable so the script can
run against a hosted API or a fully local, no-API-key model:

    LLM_PROVIDER=openai    -> OpenAIChatGenerator   (needs OPENAI_API_KEY)
    LLM_PROVIDER=mistral   -> MistralChatGenerator   (needs MISTRAL_API_KEY)
    LLM_PROVIDER=local     -> TransformersChatGenerator (default, no key needed)

See the README in this folder for details and model overrides.
"""

import os

from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TEMPLATE = [
    ChatMessage.from_user(
        """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
    )
]


def index_documents() -> InMemoryDocumentStore:
    document_store = InMemoryDocumentStore()

    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

    doc_embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL)
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(docs_with_embeddings["documents"])

    return document_store


def build_chat_generator():
    """Pick a ChatGenerator based on LLM_PROVIDER (default: local, no API key)."""
    provider = os.environ.get("LLM_PROVIDER", "local").lower()

    if provider == "openai":
        from haystack.components.generators.chat import OpenAIChatGenerator

        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("LLM_PROVIDER=openai requires the OPENAI_API_KEY environment variable to be set.")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIChatGenerator(model=model)

    if provider == "mistral":
        from haystack_integrations.components.generators.mistral import MistralChatGenerator

        if "MISTRAL_API_KEY" not in os.environ:
            raise RuntimeError("LLM_PROVIDER=mistral requires the MISTRAL_API_KEY environment variable to be set.")
        model = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
        return MistralChatGenerator(model=model)

    if provider == "local":
        from haystack_integrations.components.generators.transformers import TransformersChatGenerator

        # Default to a small, CPU-friendly model so the tutorial runs out of the box.
        # Override with LOCAL_MODEL=Qwen/Qwen3-4B-Instruct-2507 for the model used in
        # the original tutorial (needs more RAM/time to download and run).
        model = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        return TransformersChatGenerator(model=model)

    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Use one of: openai, mistral, local.")


def build_pipeline(document_store: InMemoryDocumentStore) -> Pipeline:
    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)
    retriever = InMemoryEmbeddingRetriever(document_store)
    prompt_builder = ChatPromptBuilder(template=TEMPLATE, required_variables="*")
    chat_generator = build_chat_generator()

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", chat_generator)

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder")
    rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

    return rag_pipeline


def ask(pipeline: Pipeline, question: str) -> str:
    response = pipeline.run(
        {"text_embedder": {"text": question}, "prompt_builder": {"question": question}}
    )
    return response["llm"]["replies"][0].text


def main() -> None:
    document_store = index_documents()
    pipeline = build_pipeline(document_store)

    questions = [
        "What does Rhodes Statue look like?",
    ]
    if os.environ.get("RUN_ALL_EXAMPLES"):
        questions += [
            "Where is Gardens of Babylon?",
            "Why did people build Great Pyramid of Giza?",
            "Why did people visit the Temple of Artemis?",
            "What is the importance of Colossus of Rhodes?",
            "What happened to the Tomb of Mausolus?",
            "How did Colossus of Rhodes collapse?",
        ]

    for question in questions:
        print(f"\nQ: {question}")
        print(f"A: {ask(pipeline, question)}")


if __name__ == "__main__":
    main()
