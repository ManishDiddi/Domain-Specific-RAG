# Domain-Specific RAG — HuggingFace / Gradio Documentation

An end-to-end Retrieval-Augmented Generation pipeline over 2,647 documentation files from
the HuggingFace and Gradio GitHub repositories, with a hand-labelled evaluation harness
that separates retrieval failures from generation failures.

Everything lives in [`rag.ipynb`](rag.ipynb), which runs top-to-bottom from a clean kernel.

```
RAG_BCS_Dataset.csv  ->  chunks.jsonl  ->  embeddings.npy  ->  Qdrant collection
   2,647 documents        16,071 chunks     16,071 x 768          HNSW, inner product
                                                                        |
                              question  ->  retrieve top-7  ->  grounded answer
                                                                        |
                                            retrieval metrics + LLM-judge scores
```

## Results

Measured against a 14-query control set — 10 answerable, 4 out-of-scope — whose gold
chunks were labelled by hand from depth-20 candidate pools.

| | value | notes |
|---|---|---|
| `recall@7` | **0.913** | macro-averaged over the 10 in-scope queries |
| `precision@7` | **0.343** | against a theoretical ceiling of 0.400; 7 of 10 queries sit *at* their ceiling |
| `Hallucination` | **0.060** | Opik + `gpt-4o`; threshold < 0.5, no query fails |
| `AnswerRelevance` | **0.946** | threshold >= 0.7, no query fails |
| out-of-scope abstention | **4 / 4** | the guardrail fires on every unanswerable query |
| in-scope false refusals | **0 / 10** | and never fires when it shouldn't |

**Precision has a hard ceiling.** A query with `g` gold chunks caps `precision@k` at
`min(g, k) / k`. Three of the ten queries have a single gold chunk, capping their
`precision@7` at 0.143. The measured 0.343 is 86% of the 0.400 that is actually
attainable, not 34% of a meaningless 1.0.

### Choosing `top_k`

Swept, not assumed:

| k | precision@k | recall@k | Δrecall |
|---|---|---|---|
| 1 | 0.700 | 0.281 | — |
| 3 | 0.600 | 0.710 | +0.429 |
| 5 | 0.400 | 0.749 | +0.039 |
| **7** | **0.343** | **0.913** | **+0.164** |
| 10 | 0.260 | 0.961 | +0.048 |

Going 7 → 10 buys +0.048 recall for −0.083 precision. Recall gates answer quality — a gold
chunk that never reaches the prompt cannot be used — but at k=10 roughly three quarters of
the context window is material the model must read past.

### Query-side instruction

BGE v1.5 is trained on asymmetric (short query, long passage) pairs with a retrieval
instruction on the query side and none on the passage side. Measured rather than assumed,
because BAAI report v1.5 also works without it:

| `use_instruction` | recall@7 | precision@7 |
|---|---|---|
| True | **0.913** | 0.343 |
| False | 0.788 | 0.314 |

One string concatenation, no re-embedding, no index rebuild.

## Architecture

| stage | choice | why |
|---|---|---|
| chunking | sliding window, 500 tokens / 64 overlap | `bge-base` has a 512-position limit and spends 2 on `[CLS]`/`[SEP]`; 500 leaves headroom so nothing is silently truncated. Overlap ≈13% so a fact straddling a boundary survives whole in one window. |
| lineage | `source`, `doc_id`, `chunk_index`, `char_start`, `char_end`, `n_tokens` on every chunk | lets a retrieved chunk be cited back to a file and character range; validated as an exact substring of its parent |
| embeddings | `BAAI/bge-base-en-v1.5`, 768-dim, `normalize_embeddings=True` | unit length makes inner product **identical to cosine**, so the cheaper metric costs nothing |
| vector store | Qdrant, `Distance.DOT`, HNSW, `chunk_id` as point ID | externally-assigned IDs stay stable across re-ingestion, so two evaluation runs stay comparable |
| generator | `openai/gpt-oss-20b` via Groq, `temperature=0.0` | deterministic, so metrics measure the pipeline rather than sampling noise |
| judge | `gpt-4o` via Opik | deliberately a different model family — scoring a model with itself is circular |

### Latency

| stage | min | median | max |
|---|---|---|---|
| retrieval | 74 ms | **166 ms** | 3,924 ms |
| generation | 851 ms | **26,735 ms** | 35,629 ms |

Generation is roughly **160× the cost of retrieval**, which decides where tuning effort is
worth spending: raising `top_k` costs almost nothing in retrieval time, and its real price
is prompt tokens and distraction. A ~27 s median answer suits an offline harness and would
not suit an interactive support desk — the first thing to revisit for that use case.

## Design substitutions

The brief specifies **Milvus**; this build uses **Qdrant in local (embedded) mode**. The
requirement Milvus was carrying — an explicit schema, an inner-product metric, an ANN
index and a robust ingestion path — is met and verified in Stage 2.2, and local mode needs
no server process, so the notebook reproduces from a clean kernel on one machine. What is
given up: partitions, tunable consistency levels, and any evidence the design scales past
a single process.

This and three other substitutions — the embedding model, the hand-labelled control set
over an LLM-labelled one, and the judge's context format — are documented with their
trade-offs in the notebook's closing section.

## Setup

Requires Python 3.10+ (the type annotations use `X | Y` syntax).

**1. Dataset.** `RAG_BCS_Dataset.csv` is 21 MB and is not committed. Place it in the
project root. It should have 2,647 rows and the columns `text` and `source`.

**2. Credentials.** Create a `.env` file in the project root — it is gitignored, and no
key is ever written into the notebook or printed by it:

```
GROQ_API_KEY=...      # required — the generator LLM
OPENAI_API_KEY=...    # required — the gpt-4o judge used by Opik in 5.4
OPIK_API_KEY=...      # optional — tracing and the Opik dashboard
```

All three are checked in Setup, so a missing key fails in seconds with the exact line to
add, rather than 30 minutes in after chunking and encoding have already run.

**3. Dependencies** are installed by the first cell:

```
sentence-transformers  qdrant-client  groq  opik  python-dotenv
```

## Running

Open `rag.ipynb` and run all cells. On a first run every artifact is built in order —
chunking, encoding 16,071 vectors, Qdrant ingestion, pool construction, answer generation
and judging. Re-runs reload the persisted artifacts instead and take about a minute; the
`FORCE_RECHUNK`, `FORCE_REEMBED` and `FORCE_REINGEST` flags in the Configuration cell
force a rebuild when one is wanted.

Every stage ends with a checkpoint block reporting what it produced, so a run can be
verified without re-executing it:

```
--------------------------------------------------------------------
Stage 1 — data processing & chunking
--------------------------------------------------------------------
  documents in           2,647
  documents represented  2,647
  chunks out             16,071
  window                 500 tokens, 64 overlap (13%)
  tokens per chunk       min 9, mean 461, max 500
  over budget            0  (must be 0)
```

`PROJECT_DIR` resolves from the working directory, so the notebook runs unchanged on
another machine. Set `RAG_PROJECT_DIR` if the kernel starts elsewhere.

## Repository layout

| path | |
|---|---|
| `rag.ipynb` | the full pipeline and evaluation, with outputs |
| `processed/control_set_v1.jsonl` | 14 queries with hand-labelled gold chunks, pinned to a corpus hash |
| `processed/candidate_pools_v1.jsonl` | depth-20 in-scope pools, every candidate judged |
| `processed/candidate_pools_oos_v1.jsonl` | out-of-scope pools with lexical verification |
| `processed/generated_answers_v1.jsonl` | the evaluated answers, with the exact context each one saw |
| `STAGES.md` | build notes |
| `PROBLEM_STATEMENT.md` | the original brief |

`chunks.jsonl`, `embeddings.npy` and the Qdrant store are regenerable and excluded from
version control.

## Limits

Stated in full in the notebook's §5.5.9, and briefly here:

- **n = 10 in-scope queries.** One query moves any macro-average by up to 0.10, which is
  why the analysis rests on a per-query failure table rather than on the means.
- **Gold came from a depth-20 pool**, so recall at high k is partly tautological; `recall@1`,
  `@3` and `@5` are the informative columns.
- **One annotator, who is also the analyst.** No inter-annotator agreement figure exists.
- **The judge is validated on 4 hand-checked claims, not an agreement study.** It proved
  directionally reliable but uncalibrated — score magnitude does not track error severity —
  so it is used to locate answers worth reading, never to rank them.
- **`bge-base` vs `bge-small` was never measured.** The 2× embedding cost is asserted to be
  worthwhile, not demonstrated.
