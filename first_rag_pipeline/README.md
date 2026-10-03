# First RAG Pipeline

Based on the Haystack "Creating Your First QA Pipeline with RAG" tutorial
(see [`readme`](readme) for the original tutorial text).

Indexes the Wikipedia-derived "Seven Wonders of the Ancient World" dataset
into an `InMemoryDocumentStore` with sentence-transformer embeddings, then
answers questions about it with a retrieval-augmented generation (RAG)
pipeline: `SentenceTransformersTextEmbedder` → `InMemoryEmbeddingRetriever`
→ `ChatPromptBuilder` → a `ChatGenerator`.

> New to this? See [`STUDY_NOTES.md`](STUDY_NOTES.md) for an explanation of
> the RAG concepts and Haystack components used here.

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
python rag_pipeline.py
```

By default this runs fully locally — no API key required — and answers
"What does Rhodes Statue look like?" using a small local model.

To ask all of the tutorial's example questions:

```bash
RUN_ALL_EXAMPLES=1 python rag_pipeline.py
```

## Choosing a ChatGenerator

The tutorial shows three options for the LLM that generates the final
answer. This script picks between them with the `LLM_PROVIDER` environment
variable:

| `LLM_PROVIDER` | Generator                    | Requirement                          |
|----------------|-------------------------------|---------------------------------------|
| `local` (default) | `TransformersChatGenerator` | none — runs on CPU/MPS, no API key    |
| `openai`       | `OpenAIChatGenerator`          | `OPENAI_API_KEY` env var              |
| `mistral`      | `MistralChatGenerator`         | `MISTRAL_API_KEY` env var             |

```bash
# Hosted provider example:
OPENAI_API_KEY=sk-... LLM_PROVIDER=openai OPENAI_MODEL=gpt-4o-mini python rag_pipeline.py
```

For `local`, the default model is `Qwen/Qwen2.5-0.5B-Instruct` (small and
fast, good for verifying the pipeline works). To match the original
tutorial's model instead:

```bash
LOCAL_MODEL=Qwen/Qwen3-4B-Instruct-2507 python rag_pipeline.py
```

(the 4B model needs substantially more RAM/time to download and run).

## Notes / fixes vs. the original tutorial

- The tutorial's notebook cells use `getpass()` to prompt for API keys
  interactively. That blocks in a non-interactive script, so this version
  reads `OPENAI_API_KEY` / `MISTRAL_API_KEY` from the environment and
  raises a clear error if the selected provider's key is missing, instead
  of hanging.
- Defaults to the local, no-API-key `TransformersChatGenerator` path so the
  tutorial is runnable out of the box; the tutorial's default (`gpt-4o-mini`
  via OpenAI) is one `LLM_PROVIDER=openai` away.
- Verified end-to-end: indexing, retrieval, prompt building, and generation
  all run and produce a correct answer for "What does Rhodes Statue look
  like?".
- Added `setup.sh` for one-off venv creation + dependency install.
