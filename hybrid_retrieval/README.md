# Hybrid Retrieval

Based on the Haystack "Creating a Hybrid Retrieval Pipeline" tutorial (see
[`readme`](readme) for the original tutorial text).

Indexes a slice of the `anakin87/medrag-pubmed-chunk` PubMed abstracts
dataset with dense embeddings, then runs a hybrid retrieval pipeline that
combines BM25 keyword search and embedding-based dense retrieval, merging
and reranking both result sets with a cross-encoder similarity ranker.

> New to this? See [`STUDY_NOTES.md`](STUDY_NOTES.md) for an explanation of
> the retrieval/reranking concepts and Haystack components used here.

## Setup

One-off setup script (creates `.venv` and installs `requirements.txt`,
auto-picking a Python 3.10+ interpreter):

```bash
./setup.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Needs Python 3.10+ (tested with Python 3.13 via Homebrew:
> `/opt/homebrew/bin/python3.13 -m venv .venv`, or
> `PYTHON_BIN=/path/to/python3.x ./setup.sh`).

After setup, activate the venv before running scripts:

```bash
source .venv/bin/activate
```

## Run

```bash
python hybrid_retrieval.py
```

This indexes 500 documents by default (quick to run) and answers the
tutorial's example query, "apnea in infants", printing each result's title,
reranked score, and abstract.

Useful environment variables:

- `SAMPLE_SIZE` — number of dataset rows to index (default `500`; set to
  `0` to index the full ~15k-row dataset, which takes longer).
- `QUERY` — the search query (default `"apnea in infants"`).
- `EMBEDDING_MODEL` / `RANKER_MODEL` — override the embedding or
  cross-encoder reranker models.
- `HYBRID_DEVICE` — force a specific device (e.g. `cpu`, `mps`, `cuda:0`);
  by default the best available device is auto-detected.

## Notes / fixes vs. the original tutorial

- **Fixed a hard-coded CUDA device.** The tutorial hard-codes
  `device=ComponentDevice.from_str("cuda:0")` for both the document and
  text embedders, which raises an error on any machine without an NVIDIA
  GPU (e.g. Apple Silicon or a CPU-only machine). This script instead uses
  `ComponentDevice.resolve_device(None)` to auto-detect the best available
  device — on Apple Silicon this correctly picks up MPS. Verified by
  running on an Apple M4 Pro: `ComponentDevice.resolve_device(None)`
  resolves to `DeviceType.MPS`, and the full pipeline runs successfully on
  it.
- Defaults to a 500-document slice of the dataset (via `SAMPLE_SIZE`)
  instead of the full ~15k rows, so the tutorial runs in a reasonable time
  out of the box; the full dataset is one environment variable away.
- Verified end-to-end: indexing (200-document run) and the hybrid
  retrieval + reranking pipeline both complete and return ranked results.
- Added `setup.sh` for one-off venv creation + dependency install.
