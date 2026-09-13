# Assignment stages

The submission portal reveals the work in **6 stages**. This file records each
stage's tasks verbatim as they are unlocked, re-ordered into build order, with
the rubric row each task feeds.

Source of truth for grading: `PROBLEM_STATEMENT.md`.
Source of truth for design decisions: `CLAUDE.md`.

---

## Stage 1 of 6 — Chunking the Knowledge Base

**Status:** 1.1–1.3 done, 1.4 outstanding (target was Mon 31 Aug — one day over)
**Rubric row it feeds:** #1 Data Processing & Chunking Logic
**Notebook:** `rag.ipynb`

Tasks, in the order they should actually be done (the portal lists them shuffled):

### Task 1.1 — Load & inspect
> Load the HuggingFace doc dataset and inspect its structure (fields, document
> lengths, count).
>
> *Hint: think about what fields and scale you're working with before you design
> your chunking logic.*

Done means: field names, row count, and the **distribution** of document lengths
— not just a mean. The 371,057-char outlier has to be visible in the output, and
it has to be visible *before* the chunker is written, because it is what the
chunker gets tested against.

### Task 1.2 — Sliding-window chunker
> Implement a sliding-window chunking function that takes `chunk_size` and
> `chunk_overlap` as parameters and walks each document, producing overlapping
> text chunks.
>
> *Hint: think about what happens if your window never advances — what stopping
> condition guarantees the loop terminates once you reach the end of the text?*

Done means: `chunk_size` and `chunk_overlap` are **parameters**, not constants
baked into the body. The step is `chunk_size - chunk_overlap`; if that is ever
`<= 0` the loop never advances — decide whether you guard it or let it raise,
and be able to say which and why. Last window must not silently drop the tail.

### Task 1.3 — Lineage metadata
> Attach lineage metadata (a unique `chunk_id` and the parent `source`) to every
> chunk so it can be traced back to its original document.
>
> *Hint: think about what information a downstream retrieval step would need to
> answer "where did this text come from?"*

Done means: every chunk carries at minimum `chunk_id`, `source`, and its index
within the parent document. This is the field set that has to survive all the
way into the Milvus schema (rubric #3) and back out of retrieval into the cited
answer (rubric #5) — so decide it here, once.

### Task 1.4 — Run it and report
> Run the chunker across the full dataset and report chunk counts and average
> chunks-per-document.
>
> *Hint: think about how you'd summarize the output of Task 1.2/1.3 to
> sanity-check it before moving on.*

Done means: total chunks, mean chunks/doc, and the **max** chunks for a single
document. Expect roughly **~12k chunks** at 512/64 — if the number is wildly off
from that, the chunker is wrong, not the estimate. Keep the run cheap to repeat;
you will re-run it when you tune size/overlap for rubric row #8.

### Decisions locked in Stage 1

| Decision | Value | Reason |
|---|---|---|
| Chunk unit | **tokens**, not characters | A fixed char window overflows the model's 512-token limit on dense text (code, CJK) and `bge-small` truncates **silently** — the stored vector would cover only part of the chunk. A token window makes that structurally impossible. |
| Boundary mechanism | `return_offsets_mapping=True`, slice the original string | Cuts on token boundaries but never calls `decode()`, so chunk text is a byte-exact substring of the source and `char_start`/`char_end` stay valid for citation. |
| `chunk_size` | **510** | 512 model positions − `[CLS]` − `[SEP]`. |
| `chunk_overlap` | **64** | **Provisional.** No evidence yet; must be validated against retrieval in Stage 4 or it is an asserted choice, not a measured one. |
| `chunk_id` | self-assigned global int, not Milvus `auto_id` | Stable across re-ingestion, so two eval runs stay comparable. |
| Record shape | one list of dicts, text + metadata together | Parallel lists desynced once already (`(15726, 2647)`); one structure makes it unrepresentable. |
| Min-chunk floor | **deferred to Stage 4** | No evidence for a threshold value yet. Sliver count to be measured in 1.4. |

Full corpus run at 510/64 yields **15,726 chunks** from 2,647 documents (~6/doc) —
consistent with ~2,300 tokens/doc over a stride of 446.

### Open debts carried out of Stage 1

- Task 1.4 report + invariant checks (notebook cells 20 and 22) — **still TODO**
- Design-justification markdown (notebook cells 4, 9, 13) — **still TODO**, rubric rows 1 and 8
- Max chunk length in *characters* — needed to set Milvus `VARCHAR max_length` in Stage 3
- Non-English share of the corpus — known limitation of the English-only model, needs quantifying
- 19 exact-duplicate document texts in the corpus — note before Stage 7, or they read as a retrieval bug

**Carry into Stage 2:** the chunk records produced here are the input to the
batched embedding pipeline. Persist them rather than regenerating.

---

## Stage 2 of 6 — Vectorization & Vector Store

**Status:** 2.1–2.4 implemented in Colab as of 4 Sep — under review, not signed off
**Rubric rows it feeds:** #2 Vectorization & Memory Management, #3 Vector DB Architecture & Persistence
**Resolved:** vector store is **Qdrant** (`qdrant-client` local mode, HNSW index).
The reason for moving off Milvus is still undocumented — see the note at the end of this stage.

### Task 2.1 — Model + batched encoding
> Choose and load an embedding model, then batch-encode all chunks (rather than one at a
> time) to keep memory and throughput manageable.
>
> *Hint: think about what happens to memory and runtime if you encode the entire corpus in
> a single call versus in smaller groups.*

Done means: encoding runs in batches with a tuned `batch_size`, and the resulting vectors
are **persisted to disk**. Rubric row 2 says vectors are never recomputed — if restarting
the kernel costs you the embeddings, this task is not done. 15,726 chunks x 384 dims x
float32 = ~24 MB, so persistence is cheap and there is no excuse.

For row 8 you also owe a *measured* comparison: bge-small vs bge-base, and the latency
cost of each. Numbers, not adjectives.

### Task 2.2 — Normalization
> Apply vector normalization during encoding so similarity search behaves consistently.
>
> *Hint: think about which similarity metric works correctly only when vectors are
> unit-length.*

Done means: `normalize_embeddings=True`, plus the explanation of **why** — on unit-length
vectors, inner product is mathematically identical to cosine similarity. This is the thing
the rubric is quietly testing. He should be able to state it without hesitating, and
verify it numerically (all norms == 1.0).

**Trap:** BGE models take a query-side instruction prefix and **no prefix on passages**.
Chunks encoded here are passages: no prefix. Getting this backwards silently degrades
retrieval with no error anywhere.

### Task 2.3 — Collection setup
> Set up a vector database collection with the correct dimensionality and similarity metric
> for your normalized vectors, handling the case where the collection already exists.
>
> *Hint: think about what happens if you re-run this cell twice without handling an
> existing collection.*

Done means: schema carries all nine metadata fields, `dim=384`, IP metric, an index, and
**idempotent creation** — re-running the cell must not raise, and must not silently append
a second copy of the corpus. Text field length needs headroom above the measured 6,821-char
maximum.

### Task 2.4 — Batched insert + verification
> Insert all chunk vectors and metadata into the collection in batches, and confirm every
> record was inserted.
>
> *Hint: think about how you would verify, after insertion, that no records were silently
> dropped.*

Done means: batched ingestion, and a post-insert count that is **asserted against 15,726**,
not eyeballed. Note that entity counts are not necessarily accurate until data is flushed —
verifying too early is the classic way to "confirm" a number that is still settling.

### Vector store change — Milvus to Qdrant (4 Sep)

Implemented on Qdrant local mode with an HNSW index. `PROBLEM_STATEMENT.md` evaluation
criteria row 3 names **Milvus** explicitly ("technical accuracy in configuring the Milvus
collection schema, metric types (IP)"), so this costs a graded row **unless the trade-off is
written into the notebook**. The choice is made and is not being reopened; the debt is the
justification, and it needs the concrete blocker that caused the switch.

Note the metric drifted with the database: the collection is created with
`Distance.COSINE`, not dot product. Mathematically identical on unit vectors, but it
discards the IP-equals-cosine argument the assignment was testing, and makes Qdrant
re-normalize vectors that were already normalized in Task 2.2.

---

## Stage 3 of 6 — Retrieving Relevant Context

**Status:** implemented (notebook cells 70–74). Portal task text not captured.
**Rubric row:** #4 Retrieval Precision

`search_vector_db(query, qdrant_client, top_k=5)` — embeds the query, searches Qdrant,
returns `chunk_id` / `score` / `text` / `source`.

**Open debts:**
- **The BGE query prefix is missing.** Queries are encoded bare; BGE expects a query-side
  instruction prefix and no prefix on passages. Silently degrades retrieval. Fixing it and
  measuring both ways is free row 8 evidence.
- `with_payload` returns only `text` and `source` — `doc_id` and `chunk_index` are dropped,
  so citations can't say "chunk 4 of 17" and doc-level metrics need a re-join.
- `top_k=5` is a default, not yet a justified choice. Row 4 wants evidence.

---

## Stage 4 of 6 — Grounded Answer Generation

**Status:** implemented (notebook cells 75–98). Portal task text not captured.
**Rubric row:** #5 Grounded Synthesis & Hallucination Control

Groq client, `openai/gpt-oss-20b`. `format_retrieved_context` + `generate_answer`,
composed by `complete_rag_pipeline(query, top_k=5)`. Guardrail demonstrated on two
out-of-scope queries (pgvector, Kubernetes ingress) in cells 97–98.

**Open debt:** the LLM moved from the planned Gemini/Claude to Groq without a written
reason — same debt as the Milvus→Qdrant swap. One sentence each, in the notebook.

---

## Stage 5 of 6 — Automated Evaluation & Diagnosis

**Status:** in progress (started 7 Sep)
**Rubric rows:** #4 Retrieval Precision, #7 Opik Metric Interpretation, #8 Technical Rigor

> **Rows 7 and 8 are where this submission is won.** Most submissions write two sentences
> of metric interpretation. Task 5.4 is the two-page version. Do not under-invest here to
> spend more time tuning the pipeline.

### Task 5.1 — Labeled test set
> Build a labeled test set of queries with known-relevant chunks.
>
> *Hint: think about how you'd decide, for a given query, which chunks count as 'relevant'
> without manually labeling the entire corpus.*

The hint is pointing at **pooling**: you cannot label 15,726 chunks, so retrieve a generous
candidate pool per query across a few configurations, label only the pool, and report recall
as *recall over the pool* — a documented approximation, not ground truth.

**Relevance rule (state it in the notebook):** a chunk is relevant if it contains information
*necessary* to answer the query, whether or not it is sufficient alone. "Sufficient alone"
is the wrong bar — with 500-token chunks many answers span neighbours.

Done means:
- [ ] Stratified query sampling — across `huggingface`/`gradio`, document lengths, and
      deliberately including low `chars_per_token` chunks so the CJK weakness surfaces
- [ ] Queries generated from seed chunks; seed chunk is the initial gold
- [ ] Gold set expanded by judging a pooled candidate set against the relevance rule
- [ ] **The judge validated against 40–50 hand-labelled pairs, with an agreement number.**
      Without this the metrics rest on an oracle nobody checked. Budget ~90 minutes.
- [ ] Hand-written queries (cells 92–98) kept as a separate control — the gap between their
      scores and the generated ones measures leakage bias in the synthetic set
- [ ] Persisted as a versioned artifact with a manifest, like every other stage

### Task 5.2 — Retrieval metrics
> Implement and compute retrieval metrics (precision@k and recall@k) across the test set.
>
> *Hint: think about what precision and recall each measure differently when judging a
> ranked list of results.*

Done means: both metrics across **several values of k** — that sweep is the evidence rubric
row 4 wants for the chosen `top_k`, and a single k proves nothing.

**Report precision@k, but interpret it.** With ~2 relevant chunks per query, precision@10
is mathematically capped at 0.2. It will look terrible while retrieval works fine. Add
**hit rate@k** and **MRR** — MRR is what actually tells you whether to raise or lower
`top_k`. Explaining the ceiling effect *is* the answer to the portal's hint.

### Task 5.3 — Answer quality
> Score generated answers for relevance and hallucination using an automated evaluation
> approach.
>
> *Hint: think about what signals would tell you an answer is unsupported by its retrieved
> context, versus simply low quality.*

Opik: Hallucination + Answer Relevance. **Both are reference-free** — they judge the answer
against the retrieved context and the question, not against a gold answer. No reference
answer set is needed, and building one would be days the rubric does not pay for.

The hint distinguishes the two axes: unsupported-by-context is hallucination, doesn't-address-
the-question is relevance. Store query + retrieved chunks + answer + both scores together,
because Task 5.4 needs to join them.

### Task 5.4 — Interpretation and diagnosis
> Interpret the results against explicit thresholds and analyze any low-scoring cases to
> diagnose likely causes.
>
> *Hint: think about what you'd want to inspect for a query that scored poorly — is the
> failure in retrieval or in generation?*

**"Explicit thresholds" means stated and justified *before* reading the results.** A
threshold chosen after seeing the scores is a rationalisation, and picking it in advance is
the difference between evaluation and storytelling.

The hint is the whole diagnostic method — cross retrieval success against generation quality:

| | retrieval hit | retrieval miss |
|---|---|---|
| **good answer** | working as designed | answered from parametric memory — check it isn't luck |
| **bad answer** | retrieved but ignored → prompt problem | context never there → embedding/chunking problem |

Same symptom, opposite fixes. That 2×2 is the failure taxonomy.

Done means: a **table of concrete failure cases**, each traced to a cause — bad chunk
boundary · embedding miss · top_k too low · retrieved-but-ignored · genuinely absent from
the corpus. Two pages, not two sentences. This is the strongest interview artifact in the
project: diagnosing failures from metrics is the QA skill, and this section is where it
converts from background into the reason to hire.

### Decisions locked in Stage 5 (9 Sep)

| Decision | Value | Reason |
|---|---|---|
| Label source | **Reverse generation** (chunk → query) | Labelling forward — write a query, then hunt 15,726 chunks for what answers it — is person-months. Generating the question *from* a seed chunk makes the label free by construction. Lineage: Doc2Query → InPars → Promptagator; the mechanism RAGAS uses. |
| Gold expansion | **LLM-judged pooling** over top-20 | The seed chunk is *a* relevant chunk, not the only one — 64-token overlap and 19 duplicate documents guarantee neighbours carry the same content. Single-positive labels make recall a lower bound. This is TREC pooling with an LLM assessor. |
| Judge model | **not** `gpt-oss-20b` | Judging the pipeline with the model under test is circular. |
| Judge validation | 40–50 hand-labelled (query, chunk) pairs, agreement reported | An unvalidated judge is an oracle nobody checked. |
| Control queries | ~20 hand-written, **written before any generated query is read** | Anchoring is not reversible: once you have read LLM-written queries you cannot write un-influenced ones. Cells 92–98 are the seed of this set. |
| Round-trip check | **triage, not filter** | Deleting every query the retriever fails tunes the benchmark to flatter the system under test. Failures are quarantined, read by hand, classified *bad question* vs *genuine retrieval miss*, and both counts reported. |
| Set size | 80–100 queries | ±~5% sampling noise at n=80 — small enough to read every failure by hand. Unread test cases are decoration. |
| Near-miss out-of-scope | included as its own query type | "What is my name?" does not test the guardrail. A HuggingFace-dense question whose answer is genuinely absent (e.g. A100 endpoint pricing) returns high-scoring plausible chunks and dares the prompt to hallucinate. That is the real test. |
| Thresholds | **declared and justified in the notebook before any score is read** | Per Task 5.4. A threshold picked after seeing the distribution is a rationalisation. |
| Persistence | `processed/eval/` + `stage5_manifest.json` | Same contract as Stages 1 and 2. |

### Imported labelled set — `processed/eval_set_v1.jsonl` (9 Sep)

Built by `scripts/build_eval_set.py` from a 96-query LLM-labelled pool
(`rag_eval_set_v1.jsonl`). **29 kept: 19 in-scope + 10 out-of-scope.**

| Property | Raw pool | Imported set |
|---|---|---|
| Queries | 96 | 29 |
| Distinct gold chunks | 35 | 22 |
| Distinct source docs | 23 | 16 |
| Max queries sharing one gold chunk | **7** (chunk 6002) | **1**, by construction |

**Why the pool was filtered rather than imported whole.** 96 queries resolving to 35
gold chunks is not 96 independent observations. 30% of the answerable pool depended on
five chunks, so a single embedding miss would move recall@k by up to 7 queries at once —
the metric moves in correlated blocks and its apparent precision is fake. Two filters:
`answer_support_coverage >= 0.75` (below that the labeller itself flagged the gold set as
incomplete, which makes *correct* retrievals score as misses), and no gold chunk backing
more than one query.

**The cost of that filter — state this in the row-7 writeup.** Coverage correlates with
question simplicity, so filtering on it strips out the hard cases. Surviving mix is
13 lookup / 3 paraphrase / 2 inference / 1 multi-hop; 13 medium / 4 easy / 2 hard.
**This set will understate the failure rate.** Hard and multi-hop cases must be
hand-written on top of it — the reverse-generation plan above is still the main source.

**Nothing in the file is ground truth yet.** All 29 ship `verification.status:
"pending_human"`. Records carry `gold_sources` so a label can be checked without
re-opening `chunks.jsonl`; notebook cell has a `show_label()` helper for the pass.
Until verification is done, a failed retrieval cannot be distinguished from a bad label.

Labels are pinned to `corpus_sha256_16: d9e9b63fe132de23`. `load_eval_set()` re-hashes
`chunks.jsonl` and refuses to run on a mismatch — gold chunk ids are positions in one
specific corpus, and re-chunking silently repoints them.

### Verification pass complete (10 Sep) — and what it exposed about method

All 29 records now carry `verification.status: "verified"`. **The two categories were
verified by different methods, and only one of them is easy.**

- **19 in-scope** — read the gold chunk, confirm it answers the query. Cheap and reliable.
- **10 out-of-scope** — the label is not a claim about a chunk, it is a claim about the
  *absence* of an answer across all 16,071 chunks. That cannot be established by reading.
  Verified instead by adversarial lexical probe: regex the whole corpus for each topic,
  then read every hit to confirm none answers the query. All 10 held up.

**Row-7 finding: 3 of the 10 out-of-scope queries are near-misses, not clean misses.**
Tagged `oos_subtype: "near_miss"`.

| qid | query | lexically adjacent material in corpus | failure mode it baits |
|---|---|---|---|
| q072 | k8s ingress controller | "data ingress/egress" in an AWS Fargate cost breakdown (chunks 234–235) | assemble a deployment answer out of cost prose |
| q096 | rotate AWS IAM access keys | HF access-token rotation (chunk 592, `security-tokens.md`); image rotation in Gradio demos | confident answer about the **wrong credential system** |
| q073 | Anthropic Claude API pricing | 28 chunks mention Anthropic, none contain pricing | topic present, answer absent — the classic |

This matters because the guardrail test currently in the notebook (cell 89,
*"What is my Name and what do i do?"*) proves nothing — no corpus chunk is remotely close,
so any retriever scores low and any prompt refuses. The near-miss queries return
**high-scoring plausible chunks** and are the actual test of whether the prompt is
grounded or just pattern-matching. Report guardrail behaviour split by `oos_subtype`;
a system that passes `clean` and fails `near_miss` is the expected and interesting result.

**Method note worth stating in the writeup:** a raw lexical hit count is not evidence of
answerability. q072 returns 38 regex hits for `kubernetes|ingress` and q096 returns 77 for
`rotate|iam|access key`, yet neither topic is actually covered. Counting matches would have
produced two false "mislabelled" verdicts. Every hit has to be read.

Re-running `scripts/build_eval_set.py` now refuses to overwrite verified labels without
`--force`.

### Corrected in the notebook (9 Sep) — provenance was wrong

Both manifests named **`bge-small-en-v1.5`**, and `stage2_manifest.json` recorded
`embedding_dimension: 768` alongside it — internally impossible, bge-small is 384-dim.
`embeddings.npy` is `(16071, 768)` and the notebook loads `BAAI/bge-base-en-v1.5`.
**The pipeline ran bge-base throughout.** Manifests corrected; cell 32's model note
(which claimed bge-base is 384-dim) rewritten; cell 38 now asserts the dimension so a
model swap fails loudly instead of corrupting the index. Qdrant is `Distance.DOT` on
unit-norm vectors, so the retrieval side was always correct.

⚠️ **The bge-base choice is still undefended.** There is no measurement comparing it to
bge-small — only the (wrong) note in cell 32. Row 8 needs the number, not the assertion.

### Open debt discovered while designing Stage 5

- **The BGE query instruction prefix is missing.** `search_vector_db` (notebook cell 72)
  embeds the raw query — no `"Represent this sentence for searching relevant passages: "`.
  `CLAUDE.md` names this as one of the two things the rubric is quietly testing, and it
  silently degrades retrieval with no error anywhere. **Do not fix it blind:** the test set
  is the instrument that measures it. Run the sweep both ways, report the delta, then fix.
  A free, measured row-8 number that most submissions will not have.
  **Status 9 Sep:** `search_vector_db` now takes `use_instruction=` and defaults to
  `USE_QUERY_INSTRUCTION = True`. The switch exists so the A/B can be run — the delta
  has *not* been measured yet, so the default is currently an assumption, not a result.

---

## Stage 6 of 6 — *not yet revealed*
