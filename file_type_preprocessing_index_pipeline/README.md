# File Type Preprocessing + Indexing Pipeline

Based on the Haystack "Preprocessing Different File Types" tutorial (see
[`readme`](readme) for the original tutorial text).

Builds an indexing pipeline that converts mixed file types (`.txt`, `.md`,
`.pdf`) into Documents via a `FileTypeRouter`, cleans and splits them into
chunks, embeds them, and writes them to an `InMemoryDocumentStore`.
Optionally builds a RAG query pipeline on top using the Hugging Face
Inference API.

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
python make_sample_files.py   # generates recipe_files/*.txt, *.md, *.pdf
python preprocessing_pipeline.py
```

This indexes the sample files and prints how many chunks were written. The
optional query step (answering a question using the indexed documents via
`HuggingFaceAPIChatGenerator`) runs only if `HF_API_TOKEN` is set:

```bash
HF_API_TOKEN=hf_... python preprocessing_pipeline.py
```

To point at your own folder of files instead of the generated samples:

```bash
RECIPE_FILES_DIR=/path/to/your/files python preprocessing_pipeline.py
```

## Notes / fixes vs. the original tutorial

- **Replaced the Google Drive / `gdown` download** with
  `make_sample_files.py`, which generates small local `.txt`, `.md`, and
  `.pdf` recipe files. The original tutorial's shared Google Drive folder
  is a fragile dependency for an automated or offline setup (it can
  require auth, get rate-limited, or simply go away), so this version is
  self-contained. `gdown` was dropped from `requirements.txt`; if you'd
  like to use the original Drive folder, download it manually and pass its
  path via `RECIPE_FILES_DIR`.
- **Fixed silent markdown drop.** `FileTypeRouter` relies on Python's
  `mimetypes` module to classify files by extension, and `.md` isn't
  reliably registered as `text/markdown` on every platform — when it
  isn't, markdown files get silently skipped instead of routed to
  `MarkdownToDocument`. `preprocessing_pipeline.py` now explicitly
  registers it with `mimetypes.add_type("text/markdown", ".md")` before
  building the pipeline.
- **Removed the blocking `getpass()` prompt** in the optional query
  section. The script now reads `HF_API_TOKEN` from the environment and
  skips the query step with a clear message if it isn't set, rather than
  hanging on interactive input.
- Verified end-to-end: `make_sample_files.py` generates all three file
  types, and `preprocessing_pipeline.py` routes, converts, cleans, splits,
  embeds, and indexes all of them (3 chunks from 3 sample files).
- Added `setup.sh` for one-off venv creation + dependency install.
