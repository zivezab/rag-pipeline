"""File type preprocessing + indexing tutorial.

Builds an indexing pipeline that converts mixed file types (.txt, .md, .pdf)
into Documents, cleans and splits them into chunks, embeds them, and writes
them to an InMemoryDocumentStore. Optionally builds a RAG query pipeline on
top of the indexed documents using the Hugging Face Inference API.

Fixes vs. the original tutorial:
- Replaces the Google Drive / gdown download (fragile in automated or
  offline environments) with locally generated sample files — run
  `make_sample_files.py` first, or point RECIPE_FILES_DIR at your own
  folder of .txt/.md/.pdf files.
- Explicitly registers the "text/markdown" mimetype for the .md extension.
  Python's `mimetypes` module doesn't reliably map .md to text/markdown on
  every platform, which would otherwise cause FileTypeRouter to silently
  drop markdown files instead of routing them to MarkdownToDocument.
- The optional query section no longer blocks on an interactive
  `getpass()` prompt; it reads HF_API_TOKEN from the environment and is
  skipped (with a clear message) if it isn't set.
"""

import mimetypes
import os
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

# Some platforms don't map ".md" to "text/markdown" in the stdlib mimetypes
# registry, which would make FileTypeRouter drop markdown files silently.
mimetypes.add_type("text/markdown", ".md")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RECIPE_FILES_DIR = Path(os.environ.get("RECIPE_FILES_DIR", "recipe_files"))


def build_indexing_pipeline(document_store: InMemoryDocumentStore) -> Pipeline:
    file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
    text_file_converter = TextFileToDocument()
    markdown_converter = MarkdownToDocument()
    pdf_converter = PyPDFToDocument()

    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)

    document_embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL)
    document_writer = DocumentWriter(document_store)

    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
    preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
    preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
    preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

    preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
    preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    preprocessing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
    preprocessing_pipeline.connect("text_file_converter", "document_cleaner")
    preprocessing_pipeline.connect("pypdf_converter", "document_cleaner")
    preprocessing_pipeline.connect("markdown_converter", "document_cleaner")
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    preprocessing_pipeline.connect("document_splitter", "document_embedder")
    preprocessing_pipeline.connect("document_embedder", "document_writer")

    return preprocessing_pipeline


def run_indexing(document_store: InMemoryDocumentStore) -> None:
    if not RECIPE_FILES_DIR.exists() or not any(RECIPE_FILES_DIR.iterdir()):
        raise FileNotFoundError(
            f"No files found in {RECIPE_FILES_DIR}. Run `python make_sample_files.py` first, "
            "or set RECIPE_FILES_DIR to a folder of your own .txt/.md/.pdf files."
        )

    preprocessing_pipeline = build_indexing_pipeline(document_store)
    sources = [p for p in RECIPE_FILES_DIR.glob("**/*") if p.is_file()]
    preprocessing_pipeline.run({"file_type_router": {"sources": sources}})


def build_query_pipeline(document_store: InMemoryDocumentStore) -> Pipeline:
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.huggingface_api import HuggingFaceAPIChatGenerator

    template = [
        ChatMessage.from_user(
            """
Answer the questions based on the given context.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
"""
        )
    ]

    pipe = Pipeline()
    pipe.add_component("embedder", SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL))
    pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
    pipe.add_component("chat_prompt_builder", ChatPromptBuilder(template=template))
    pipe.add_component(
        "llm",
        HuggingFaceAPIChatGenerator(
            api_type="serverless_inference_api",
            api_params={
                "model": os.environ.get("HF_MODEL", "Qwen/Qwen3.5-9B"),
                "provider": os.environ.get("HF_PROVIDER", "together"),
            },
            generation_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ),
    )

    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever", "chat_prompt_builder.documents")
    pipe.connect("chat_prompt_builder.prompt", "llm.messages")

    return pipe


def main() -> None:
    document_store = InMemoryDocumentStore()
    run_indexing(document_store)
    print(f"Indexed {document_store.count_documents()} chunks from {RECIPE_FILES_DIR}")

    if "HF_API_TOKEN" not in os.environ:
        print(
            "\nHF_API_TOKEN is not set, skipping the optional query pipeline. "
            "Set HF_API_TOKEN to try question answering over the indexed documents."
        )
        return

    query_pipeline = build_query_pipeline(document_store)
    question = os.environ.get(
        "QUESTION",
        "What ingredients would I need to make vegan keto eggplant lasagna, "
        "vegan persimmon flan, and vegan hemp cheese?",
    )
    result = query_pipeline.run({"embedder": {"text": question}, "chat_prompt_builder": {"question": question}})
    print(f"\nQ: {question}")
    print(f"A: {result['llm']['replies'][0].text}")


if __name__ == "__main__":
    main()
