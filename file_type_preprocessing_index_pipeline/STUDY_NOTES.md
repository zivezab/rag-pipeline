# Study Notes: File Type Preprocessing + Indexing Pipeline

## The big idea

Real-world document collections are never one clean file type. An indexing
pipeline needs to: (1) figure out what kind of file it's looking at, (2)
convert each type into Haystack's common `Document` representation, (3) clean
and chunk the text, (4) embed it, and (5) write it to storage. This tutorial
builds exactly that pipeline for `.txt`, `.md`, and `.pdf` files.

## Why a router, not an if/else

```python
file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
```

`FileTypeRouter` inspects each file's MIME type and sends it down a different
named output branch (`file_type_router.text/plain`,
`file_type_router.application/pdf`, `file_type_router.text/markdown`). This
is a **branching** pipeline rather than a straight line — three converters
run in parallel on different subsets of the input files, and their outputs
all merge back into one `document_cleaner`:

```
sources -> file_type_router --text/plain-----> text_file_converter --\
                            --application/pdf-> pypdf_converter -------> document_cleaner -> document_splitter -> document_embedder -> document_writer
                            --text/markdown---> markdown_converter ---/
```

Each converter (`TextFileToDocument`, `PyPDFToDocument`, `MarkdownToDocument`)
knows how to turn its specific file format into a `Document` — stripping PDF
layout artifacts, parsing markdown syntax, etc. — so by the time documents
reach `document_cleaner`, the pipeline no longer needs to know what format
they came from.

## Cleaning and splitting are separate concerns

- **`DocumentCleaner`** — normalizes whitespace, removes artifacts. Doesn't
  change document *count*, just content quality.
- **`DocumentSplitter(split_by="word", split_length=150, split_overlap=50)`**
  — breaks long documents into fixed-size chunks *with overlap*. This is one
  of the most consequential decisions in a RAG system:
  - Too large a chunk → embeddings get "diluted" (a chunk about three
    different topics embeds to an average that matches none of them well).
  - Too small a chunk → you lose context the LLM needs to answer well, and
    you multiply your document count (more storage, slower search).
  - **Overlap** (50 of 150 words here) exists so that a sentence/idea
    straddling a chunk boundary isn't completely severed from its context in
    either resulting chunk.

## The payoff: format stops mattering after conversion

Once everything is a `Document`, the rest of the pipeline — cleaning,
splitting, embedding, writing — is identical to the single-file-type case
from the [`first_rag_pipeline`](../first_rag_pipeline/STUDY_NOTES.md) notes.
This is the general pattern for *any* heterogeneous ingestion pipeline:
normalize early, then reuse the same processing pipeline downstream.

## MIME type detection is a real gotcha

`FileTypeRouter` relies on Python's `mimetypes` module, which maps file
extensions to MIME types from a registry that **varies by OS and Python
version**. `.md` is not guaranteed to map to `text/markdown` everywhere —
when it doesn't, the router silently has no matching output branch for that
file, and the file is dropped with no error. This is the kind of bug that
passes code review and then randomly fails on a teammate's machine or in CI.
The fix applied in `preprocessing_pipeline.py`:

```python
mimetypes.add_type("text/markdown", ".md")
```

registered before building the pipeline, so the mapping is guaranteed
regardless of platform. General lesson: anything that silently drops data
instead of raising is worth an explicit test with a real file of that type.

## Why the Google Drive step was replaced

The original tutorial pulls a shared folder from Google Drive via `gdown`.
That's a reasonable shortcut in a notebook with a human watching, but it's a
fragile dependency for a reproducible script: Drive folder links can need
auth, get rate-limited, or simply be taken down — and then the entire
tutorial fails for reasons unrelated to Haystack. `make_sample_files.py`
generates equivalent local files instead, which is the more general lesson:
**prefer self-contained, generated or vendored test fixtures over live
external dependencies** whenever a tutorial/test doesn't specifically need
to demonstrate the external integration.

## Querying after indexing

The optional second half of the tutorial is just the
[`first_rag_pipeline`](../first_rag_pipeline/STUDY_NOTES.md) RAG pattern
again, applied to the documents you just indexed — proof that once
documents are in a `DocumentStore`, the query side doesn't care how they got
there.

## See also

- [`README.md`](README.md) — how to set up and run this folder.
- [`readme`](readme) — the original tutorial text.
