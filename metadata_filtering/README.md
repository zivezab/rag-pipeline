# Metadata Filtering

Based on the Haystack "Metadata Filtering" tutorial (see [`readme`](readme) for
the original tutorial text).

Builds a small `InMemoryDocumentStore` with three documents describing how to
install different versions of Haystack, each tagged with a `version` and
`date` in its metadata. A `InMemoryBM25Retriever` then answers a keyword
query ("Haystack installation") while filtering results by metadata — first
with a simple comparison (`version > 1.21`), then with a combined `AND`
filter (`version > 1.21` AND `date > 2023-11-07`).

No API keys or external downloads are required — everything runs locally.

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

> This tutorial needs Python 3.10+ (tested with Python 3.13). If your default
> `python3` is older, point the venv at a newer interpreter, e.g.
> `/opt/homebrew/bin/python3.13 -m venv .venv` on macOS with Homebrew Python,
> or `PYTHON_BIN=/path/to/python3.x ./setup.sh`.

After setup, activate the venv before running scripts:

```bash
source .venv/bin/activate
```

## Run

```bash
python metadata_filtering.py
```

## Expected output

The script prints two filtered result sets:

1. `version > 1.21` — matches the two documents tagged `1.22` and `2.0`.
2. `version > 1.21 AND date > 2023-11-07` — narrows that down to just the
   `2.0` document (the `1.22` document's date is exactly `2023-11-07`, which
   fails the strict `>` comparison).

## Notes / fixes vs. the original tutorial

- The tutorial text is a plain walkthrough; this folder turns it into a
  single, runnable `metadata_filtering.py` script with a `main()` entry
  point and explicit result printing, rather than loose notebook cells.
- No functional issues were found in the original steps — `pip install
  haystack-ai` plus the documented components run as-is.
- Added `setup.sh` for one-off venv creation + dependency install.
