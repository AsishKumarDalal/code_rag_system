# 📘 Phase 4 — RAG: The Library (Theory)

## 1. Where we are in the storyline

```
Phase 3: CPT (knowledge → weights)   Phase 4: RAG (everything → the library)   Phase 5: SFT (behavior)
            └──────────────── the two-memories architecture ───────────────┘
```

**One-sentence goal:** build an **external memory** — an index over your entire corpus (code + docs, all ~500GB-scale of it) — that any model can *search* at question time, and that returns exactly the right 5–20KB slices into the prompt.

Think of a doctor again: Phase 3 read the textbooks into long-term memory; Phase 4 is **giving the doctor a well-organized library card** — the skill of looking things up. Crucially, Phase 4 involves **no training of your model at all**: it's data engineering + a stock embedding model + a CPU/GPU indexing job. It's the cheapest phase and one of the highest-ROI ones.

---

## 2. Why a library at all — weights alone can't do this job

| Question | Weights alone | Weights + RAG |
|---|---|---|
| "What's the syntax of this library's API?" | Depends on whether the *exact* file was in the CPT slice (and not deprecated since) | Retrieves the **actual current file** and cites it |
| "What does function `foo_bar_123` do in *this* repo?" | Doubtful — likely never in weights | BM25 nails the **exact identifier** |
| Coverage ceiling | A 4B model's weights ≈ a few billion tokens of compressed knowledge; your corpus is orders of magnitude larger and grows | The whole corpus is *addressable*, constantly |
| Correctness/attribute-ability | No provenance | Every answer ships with the **source file it came from** |
| Update cost | Retraining (hours–days) | Re-index just the changed files (seconds) |

The division of labor (the core design of this project):

- **Weights** = skills, style, judgment, *frequently-used* domain intuition (Phase 3's slice, then SFT/RL behavior).
- **Library/RAG** = all the facts, all the code, all the docs — with provenance.

---

## 3. The retrieval loop, end to end

At question time, the trained model (Phase 5+) runs inside a **harness** and can emit a search action:

```
user question
   → the model may answer directly OR emit <search>…
   → harness runs the query through the pipeline below
   → top chunks (with citations) are injected into context
   → the model reads, maybe searches again (multi-hop), then composes the answer
```

The pipeline the harness calls (this phase builds it):

```
                ┌─ BM25 (sparse, exact identifiers)  ─┐
query ──▶ search┤                                     ├─ fuse ─▶ rerank ─▶ top-k chunks + citations
                └─ dense vectors (meaning, synonyms)  ─┘
```

Every stage exists to fix a different failure; hybrid retrieval + reranking is not extra complexity for its own sake — each layer exists because a different class of query breaks without it.

---

## 4. Stage 1 — Chunking: cutting the corpus into retrievable units

Retrieval never returns whole files; it returns **chunks**. Chunking decides what a chunk *is*, and it's the highest-leverage decision in RAG.

| Strategy | What it does | Problem it has |
|---|---|---|
| **Fixed-size sliding window** | cut every N tokens with overlap | chops functions/half-thoughts; cheap, dumb, robust |
| **Structural (code: AST)** | chunk at real boundaries — functions, classes, methods — bounded to ~N tokens | needs a parser per language; huge win for code (this is the **cAST**-style idea) |
| **Structural (docs/MD)** | split at headings → section chunks carrying their heading path | tables/lists/long code blocks still need size caps |
| **Contextual (Anthropic-style)** | *prepend a 50–100-token "who am I" summary* to each chunk at index time | costs one generation per chunk (we batch it on TPU/T4) |

Why structural chunking matters for code: a function that references its class and imports stays one unit; the embedding (next stage) describes *the behavior*, not a fragment. When a chunk must exceed the budget (giant function), fall back to size-splits inside the unit, ordered.

**Contextual retrieval — the one stat worth memorizing:** prepending a small situational context to each chunk (which repo, which file, which section, one line on purpose) cut **failed retrievals by ~49%**, and **~67%** when combined with reranking, in Anthropic's published evals. Why: an embedding vector is only as good as the *meaning compressed into that chunk*; a 6-line helper with no file/repo context is nearly meaningless on its own ("connect()", "run()" — of which repo? of which class?).

Each chunk also carries **metadata**: repo, file path, language, heading path, symbols, license — metadata is what makes citations and filters ("only this repo") possible downstream.

---

## 5. Stage 2 — Embeddings: turning chunks into points in meaning-space

An **embedding model** maps text → a vector (~768–1024 numbers) such that *similar meaning ⇒ nearby points* (cosine similarity as the score). At query time we embed the query once and find the nearest chunk vectors.

Key choices, in plain terms:

- **Use a stock model, don't train one.** Your model is a small LLM, not an embedding model — different objective, different data appetite. Good open options: `bge-m3`, `e5-large-v2`, `gte-large` (multilingual, long-context handled, free, Apache/MIT). This is not cheating; it's the standard stack.
- **Same model at index time and query time.** Otherwise you're matching coordinates from two different spaces — the classic silent failure.
- **Vector stores:** FAISS (HNSW) or LanceDB/Qdrant — approximate nearest neighbor, sub-ms lookup over millions of vectors.
- **Why dense alone fails:** embeddings are fuzzy. Ask for `RateLimiter.configure_lease` and dense search returns things "about rate limiting," not *that identifier*. That's why:

---

## 6. Stage 3 — Hybrid search: dense + BM25, fused

| Retrieval | Blazes at | Misses |
|---|---|---|
| **Dense (vectors)** | paraphrases, "how do I X", meaning | exact identifiers, rare symbols |
| **BM25 (sparse keyword)** | exact strings: function names, error messages, API names | "answer this differently worded question" |

Real code questions use **both** modes ("the doc describes `connect()` failing with `ECONNRESET`…"). So we run both, then **fuse** the ranked lists (Reciprocal Rank Fusion is the standard trick — literally "add up `1/(60+rank)`" scores, no tuning). Hybrid costs a few extra CPU cycles and usually gives the single biggest retrieval-quality jump after chunking itself.

---

## 7. Stage 4 — Reranking: the quality floor at the end of the queue

Retrievers are fast and shallow; a **cross-encoder reranker** is slow and deep: it reads query+chunk *together* and scores true relevance strongly. Policy: hybrid search pulls a generous top-50 → reranker picks the best ~5–8 chunks for the prompt. It's the difference between "3 of the right files in the neighborhood" and "the right 5 blocks on screen," and it's cheap at inference (one small model pass over 50 pairs).

This is also why the *trained* model helps beyond lookup: after Phase 5 the model writes *better queries* (dead-specific, symbol-heavy, multi-hop) and Phase 6's RL rewards exactly that.

---

## 8. What the model actually receives (the context contract)

```
Q: How do we invalidate the token cache in service X?

<retrieved: repo=service-X, file=auth/cache.py, chunk 3 of 12>
```python
def invalidate(user_id):
    ...
```

<retrieved: repo=service-X, file=docs/auth.md, section="Cache invalidation">
... call invalidate() after role changes ...

<Sources: auth/cache.py · docs/auth.md>
```

- Chunks arrive **labeled with their provenance** — the model must cite them back (SFT will enforce the format; RL will reward correct attribution).
- Budget: ~5–8 chunks × ~300–500 tokens ≈ 2–4k tokens of context + the question. Long enough to answer precisely, short enough to leave room for thinking.

---

## 9. Scale math on YOUR corpus and Kaggle limits

Rough numbers so the design is provably sane:

| Item | Estimate |
|---|---|
| Corpus extracted (`docs.jsonl`) | ~100–200GB text ≈ 25–50B tokens |
| Chunks (300–600 tok each) | **~50–100M chunks** |
| Dense index (1024-d fp16) | ~100–200 GB |
| BM25 index (sparse) | ~compressed 10–30GB |
| Metadata/citations | few GB |

- **Index lives on HF, not on Kaggle** — same relay pattern as everything else: embed/encode in batches → shard files flushed & removed → final index shards banked to the dataset repo. Building 100M embeddings on Kaggle T4s is *the one slow part* (batched streaming embedder, ~1–2 GPU sessions; contextual prefixes too).
- **Query time runs anywhere** — a laptop CPU + BM25 gets answers (just slower); T4 GPU makes dense retrieval instant.

---

## 10. How we'll know Phase 4 worked (the checks)

| Check | Pass |
|---|---|
| Recall@5 / Recall@10 on a dev question set | the true containing chunk appears in the top-k, ≥ 90% @10 after reranking |
| Identifier recall (`SomeClass.method` queries) | ≈ 100% via BM25 leg |
| Cited-answer smoke test | model answers from retrieved chunks, cites real files |
| Latency | retrieval < ~1s (hybrid+rerank) for the k it needs |
| No leakage | eval-set questions absent from the index (same decontamination story as Phase 3) |

Two evaluation datasets we already earmarked fit here perfectly: `wanghaofei/rag_test` (pre-chunked RAG indexes + questions) and `rag-datasets/rag-mini-wikipedia` — we can wire the harness end-to-end *before* the big corpus finishes indexing, then swap in our real corpus with zero code change.

---

## 11. Failure modes to watch (theory → practice checklist)

1. **Chunker chops functions** → semantic drift, half-signatures retrieved → prefer AST boundaries with size fallback.
2. **No contextual prefixes** → ambiguous identifiers ("connect() of which repo?") → stage-1 gap.
3. **Mismatched embed models** (index ≠ query model) → everything looks far away → pin one model, record it in the manifest.
4. **Dense-only for identifiers** → famous failure mode in code RAG; the BM25 leg exists for exactly this.
5. **Rerank-less top-8** → 3 right neighbors + 5 decoys clog context → reranker is the floor.
6. **Stale index** after corpus changes → incremental re-index/patch, not rebuild (citation correctness requires version-pinned `docs_manifest`).
7. **Prompt stuffing** → retrieval that "worked" (found *a* chunk) but context bomb → keep to top-5 after rerank; the trained model handles *when to stop looking* — that's a Phase 5/6 matter.

---

## 12. Phase 4 — inputs / outputs (the contract)

| | |
|---|---|
| **In** | canonical docs (from Phase 2 `extraction.py`) · chunk/size budget · embed model choice · decontaminated corpus guarantee |
| **Do** | AST/heading-aware chunking → context-prefixed chunks → embeddings (GPU, batched) → FAISS + BM25 index → shards banked to HF |
| **Out** | `rag/` = vector index + sparse index + chunk metadata + retrieval manifest; recall@10 ≥ 90% with rerank; usable by the harness in Phase 5/6 for `<search>` calls |

---

**Where Phase 5 goes next:** now that the model *has* domain knowledge in its weights (Phase 3) and a library to consult (Phase 4), it needs to learn **when and how to call the library** — that's SFT: supervised decompose → retrieve → solve → compose traces, seeded from `corag/multihopqa`, `Nemotron-Cascade-SFT-Stage-2`, and our own corpus-generated samples.

---

*Previous:* `docs/phase3.md` (CPT theory + dataset/cleaning/preparation plan) · *Next:* Phase 5 SFT theory, then `data/rag_build.py` implementation.
