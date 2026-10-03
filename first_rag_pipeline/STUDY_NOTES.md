# Study Notes: First RAG Pipeline

## The big idea

Retrieval-Augmented Generation (RAG) answers a question in two stages instead
of relying on an LLM's memorized knowledge alone:

1. **Retrieve** — find the documents most relevant to the question from a
   corpus you control.
2. **Generate** — hand those documents to an LLM as context and ask it to
   answer *using that context*.

This buys you two things a bare LLM call doesn't have: answers grounded in
your own data (so you can cite/update/correct it), and the ability to answer
questions about things the model was never trained on.

## The four components, and why each exists

| Component | Role | Why it's needed |
|---|---|---|
| `SentenceTransformersDocumentEmbedder` / `...TextEmbedder` | Turn text into vectors | Semantic search needs documents and queries in the same vector space |
| `InMemoryDocumentStore` | Hold documents + their embeddings | Something has to store what you can retrieve from |
| `InMemoryEmbeddingRetriever` | Nearest-neighbor search over embeddings | Finds the *k* most semantically similar documents to the query |
| `ChatPromptBuilder` | Fill a template with retrieved docs + the question | Turns retrieval output into something an LLM can consume |
| `ChatGenerator` (OpenAI / Mistral / Transformers / ...) | Produce the final answer | The actual "generation" in RAG |

**Document embedder vs. text embedder** — these are two different classes
(`SentenceTransformersDocumentEmbedder` vs.
`SentenceTransformersTextEmbedder`) even though they wrap the same underlying
model. The document embedder batches over a list of `Document` objects at
indexing time; the text embedder embeds a single query string at query time.
**They must use the same model** — if you embed documents with one model and
queries with another, the vectors live in different spaces and similarity
scores become meaningless. This is a common real-world bug source.

## How data flows through the pipeline

```
indexing (done once, offline):
  docs -> doc_embedder -> document_store

querying (done per question):
  question -> text_embedder -> embedding
                                   |
                                   v
           retriever (uses embedding + document_store) -> top-k documents
                                   |
                                   v
           prompt_builder (documents + question -> ChatMessage) -> llm -> answer
```

Two separate pipelines exist implicitly: an *indexing* step (not wrapped in a
`Pipeline` object in this tutorial — it's just sequential Python calls) and
the *query* `Pipeline` (the four connected components). In production
systems these are usually two distinct Haystack `Pipeline`s, since indexing
runs on a schedule/batch and querying runs on-demand.

## Connections are explicit about which output feeds which input

```python
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
```

Haystack components can have multiple named outputs/inputs, so
`"text_embedder.embedding"` says *specifically* use the embedding output (as
opposed to, say, metadata). `"retriever"` with no `.field` works because
`InMemoryEmbeddingRetriever` has one primary output (`documents`) that
happens to match `prompt_builder`'s only documents-shaped input.

## The prompt template

```jinja2
Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
```

This is Jinja2, not f-strings — `ChatPromptBuilder` renders it per-call with
whatever `documents` and `question` values flow in from the pipeline.
`required_variables="*"` makes both variables mandatory; without it, a typo
in a variable name fails silently instead of raising an error at prompt-build
time.

## Why the generator is pluggable

The tutorial (and this repo's `rag_pipeline.py`) deliberately decouples
*retrieval quality* from *generation choice*. Swapping `OpenAIChatGenerator`
for `MistralChatGenerator` or `TransformersChatGenerator` doesn't touch any
other component — they all implement the same `run(messages) -> replies`
contract. This is the main practical benefit of Haystack's component model:
you can benchmark different LLMs against the same retrieval pipeline by
changing one line.

- **Hosted APIs** (OpenAI, Mistral) — fast, no local compute, costs money per
  call, requires sending your data to a third party.
- **Local models** (`TransformersChatGenerator`) — free per-call, private,
  but needs RAM/GPU and is slower, especially on CPU.

## Gotchas worth remembering

- `InMemoryDocumentStore` does not persist to disk and doesn't scale — it's
  for prototyping. Swap in a real vector database (OpenSearch, Qdrant,
  Pinecone, etc.) for production, without changing the rest of the pipeline.
- Embedding model choice determines both retrieval quality and the token
  budget on the generation side indirectly — smaller/faster embedding models
  trade off semantic nuance for speed.
- `getpass()`-style interactive key prompts (from the original tutorial)
  only make sense in a notebook; any script meant to run non-interactively
  should read credentials from the environment and fail loudly if missing
  (see `build_chat_generator()` in `rag_pipeline.py`).

## See also

- [`README.md`](README.md) — how to set up and run this folder.
- [`readme`](readme) — the original tutorial text.
