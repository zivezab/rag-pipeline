# Study Notes: Metadata Filtering

## The big idea

Retrieval isn't always "find the most semantically/lexically similar
documents out of everything." Often you know in advance that only a
*subset* of the corpus is even eligible — documents belonging to a specific
user, published after a date, tagged with a specific version, in a specific
language, etc. **Metadata filtering** narrows the search space *before or
alongside* scoring, rather than relying on the retrieval algorithm to figure
out relevance on its own.

This is complementary to, not a replacement for, retrieval quality — you
still rank within the filtered set by BM25/embeddings/etc.

## Metadata lives beside content, not in it

```python
Document(
    content="Use pip to install only the Haystack 2.0 code...",
    meta={"version": 2.0, "date": datetime(2023, 12, 4)},
)
```

`meta` is a plain dict attached to each `Document`. It's never embedded or
tokenized for search — it exists purely for filtering (and for enriching
results after retrieval, e.g. showing a source date in the UI). This
separation matters: you can add/change metadata without touching embeddings
or re-indexing content.

## Filter syntax: two shapes

**Simple comparison** — one field, one operator, one value:

```python
{"field": "meta.version", "operator": ">", "value": 1.21}
```

**Logical combination** — combine multiple conditions:

```python
{
    "operator": "AND",
    "conditions": [
        {"field": "meta.version", "operator": ">", "value": 1.21},
        {"field": "meta.date", "operator": ">", "value": datetime(2023, 11, 7)},
    ],
}
```

Haystack's filter DSL supports the usual comparison operators (`==`, `!=`,
`>`, `>=`, `<`, `<=`, `in`, `not in`) and logical operators (`AND`, `OR`,
`NOT`), and these compose/nest arbitrarily — a `conditions` list can itself
contain another `{"operator": "OR", "conditions": [...]}` block. This is a
small, serializable query language, which is what lets filters come from
user input (e.g. UI checkboxes) rather than being hard-coded.

## Strict comparisons bite in predictable ways

In this tutorial's second example, the `1.22`-versioned document has
`date == datetime(2023, 11, 7)` exactly, and the filter asks for
`date > datetime(2023, 11, 7)`. Strictly-greater-than excludes it — only the
`2.0` document (dated `2023-12-04`) survives both conditions. This is a good
concrete reminder to double check `>` vs `>=` against real boundary data,
since off-by-one-day filter bugs are easy to miss in testing with small
datasets and easy to hit in production with real ones.

## `bm25_algorithm="BM25Plus"`

```python
document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
```

Haystack's `InMemoryDocumentStore` supports multiple BM25 variants. `BM25Plus`
adds a lower-bound term-frequency normalization that helps with very short
documents (it prevents a document from scoring zero just because it doesn't
repeat a query term, which vanilla BM25 is prone to at short lengths). Worth
remembering as a knob to reach for if your corpus is full of short snippets
rather than long articles.

## Filters flow through the retriever, not a separate step

```python
pipeline.run(data={"retriever": {"query": query, "filters": {...}}})
```

Filtering isn't a separate pipeline component — it's a parameter on the
retriever call itself. This means the document store applies the filter
*before* scoring (so BM25 scores are computed only over the eligible
subset), not as a post-hoc filter over already-ranked results. That
distinction matters for correctness: post-hoc filtering could return fewer
than `top_k` results even when more eligible documents exist, if the
top-scored unfiltered results happen to get filtered out.

## See also

- [`README.md`](README.md) — how to set up and run this folder.
- [`readme`](readme) — the original tutorial text.
