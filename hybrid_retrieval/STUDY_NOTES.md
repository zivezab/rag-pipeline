# Study Notes: Hybrid Retrieval

## The big idea

Two fundamentally different retrieval strategies exist, and each has blind
spots the other doesn't:

- **Keyword / lexical retrieval (BM25)** — scores documents by term overlap
  with the query (with some TF-IDF-like weighting). Excellent at matching
  exact terms: drug names, acronyms, IDs, rare domain vocabulary. Bad at
  synonyms/paraphrase — if the query says "heart attack" and the document
  says "myocardial infarction," BM25 sees no overlap.
- **Dense / embedding retrieval** — encodes query and documents into the
  same vector space and ranks by similarity. Excellent at semantic/
  paraphrase matching. Bad when the embedding model wasn't trained on your
  domain's vocabulary (e.g. a general-purpose embedder on highly technical
  medical text may not separate "apnea" from other respiratory terms as
  sharply as a domain-tuned one would).

**Hybrid retrieval** runs both and combines the results, so you get lexical
precision *and* semantic recall. This matters most in specialized domains
(exactly why this tutorial uses PubMed abstracts) where both exact
terminology and conceptual similarity are in play.

## Why reranking, not just merging

```
bm25_retriever   --\
                    --> ranker --> final results
embedding_retriever-/
```

BM25 scores and embedding similarity scores are **not on the same scale** —
you can't just sort the union of both result sets by their raw scores. A
**cross-encoder reranker** (`SentenceTransformersSimilarityRanker`) solves
this by independently scoring every (query, document) pair pulled from
*either* retriever, using a single consistent model. This produces one
unified ranking you can actually sort by.

Cross-encoders are more accurate than the bi-encoders used for retrieval
(they attend over the query and document jointly, rather than compressing
each into a fixed vector independently) but much slower per pair — which is
exactly why they're used to rerank a shortlist (tens of candidates) instead
of scanning an entire corpus.

## Pipeline shape

```python
hybrid_retrieval.connect("text_embedder", "embedding_retriever")
hybrid_retrieval.connect("bm25_retriever", "ranker")
hybrid_retrieval.connect("embedding_retriever", "ranker")
```

Both retrievers feed the *same* ranker input — Haystack pipelines support
multiple components connecting into one downstream component's input,
which is how "merge these two result sets" is expressed. Note the query
itself is passed to three different components at run time
(`text_embedder`, `bm25_retriever`, `ranker`) because each needs it in a
different form (raw string for BM25 and the ranker, embedded vector for
dense retrieval).

## Chunking at a larger size than the other tutorials

```python
document_splitter = DocumentSplitter(split_by="word", split_length=512, split_overlap=32)
```

512 words here vs. 150 in the preprocessing tutorial — chunk size is a
property of the *embedding model's* effective context window and the
*domain's* document structure, not a universal constant. PubMed abstracts
are already fairly dense, self-contained units of meaning; splitting much
smaller would cut abstracts mid-argument. Always size chunks relative to
what the embedding model handles well and what a "complete thought" looks
like in your corpus.

## The CUDA device bug (fixed in this repo)

The original tutorial hard-codes:

```python
device=ComponentDevice.from_str("cuda:0")
```

This is a trap worth recognizing: it silently assumes an NVIDIA GPU exists.
On a CPU-only machine or Apple Silicon, this raises an error. The general
pattern for portable code is to **auto-detect** rather than hard-code:

```python
device = ComponentDevice.resolve_device(None)
```

`resolve_device(None)` checks what's actually available (CUDA → MPS → CPU,
in order of preference) and picks accordingly — confirmed in this repo to
resolve to MPS on Apple Silicon. The broader lesson: anything that encodes
"my machine" into code (a device, a file path, a port number) is a bug
waiting for a different machine to run it.

## See also

- [`README.md`](README.md) — how to set up and run this folder.
- [`readme`](readme) — the original tutorial text.
