# 📘 Phase 3 — Continued Pre-Training (CPT): Theory

## 1. Where we are in the storyline

```
Base model (Qwen3-4B)  →  Phase 3: CPT  →  Phase 4: RAG  →  Phase 5: SFT  →  Phase 6: RL  →  Deploy
        🧠 general        📚 inject domain    🔎 build library   🎭 install format   🏆 sharpen judgment
```

**One-sentence goal:** take a model that already "speaks human" (the base) and make it *really* speak **your domain** — code + docs — by pushing that text through its weights.

Think of training a specialist doctor: CPT = reading the specialty's textbooks (knowledge), SFT = supervised clinical practice (behavior), RL = residency with real cases and pass/fail feedback (judgment). Phase 3 is the textbook year.

---

## 2. What CPT actually is (and isn't)

| Term | Meaning |
|---|---|
| **Pre-training** | Train a language model from random weights on trillions of tokens. Months on 1000s of GPUs. We are **not** doing this. |
| **CPT (continued pre-training)** | Take an already-trained model and **keep doing the same task it was trained for** — next-token prediction — but on *curated new data*. Just more training, on better-chosen text. |
| **SFT** | Same gradient machinery, but on question→answer demonstrations (behaviors, not knowledge). |

The twist: because the base is already good, we don't need trillions of tokens — **hundreds of millions of well-chosen ones visibly move the needle.** CPT is the cheapest way to buy domain knowledge per FLOP.

---

## 3. The only mechanism: next-token prediction + gradient descent

Everything in Phase 3 is one loop, repeated millions of times:

1. Show the model a block of ~4096 domain tokens (a file of code, a doc page).
2. It predicts the next token; we compare with truth; get a loss (how off it was).
3. Backpropagate: compute for every one of the ~4B weights "nudge this up/down to reduce the loss" (the **gradient**).
4. Update: `weight ← weight − learning_rate × gradient`.

**The learning rate (LR) is the nudge size** — and it's the most important knob in this phase (Section 6).

Because the model must predict *the next token everywhere in the document*, it's forced to internalize syntax, APIs, idioms, how docs are structured — not just memorize answers. That's why raw text (not Q&A pairs) is what CPT eats.

---

## 4. The data: what goes in decides what comes out

In pretraining, data mixture is everything; in CPT it's *even more* true because we have no token budget to waste. Design rules:

| Rule | Why |
|---|---|
| **Quality > quantity.** Dedup, filter junk, keep license-clean. 100M clean tokens beat 1B noisy ones. | Garbage permanently shapes weights. |
| **Dense, knowledge-rich slice.** Code files + reference docs + StackExchange Q&A + arXiv-style text. | Each document should "teach," not chat. |
| **Don't train on 500GB.** Pick the **10–20GB core** (≈2–5B tokens max; we use 0.1–0.3B as our first pass). | The rest serves the **RAG memory** (Phase 4), not the weights. Weights = compressed essence; index = entire library. |
| **The rest of the banked corpus is NOT wasted** — it's exactly what Phase 4 indexes. | Two memories: *weights* vs *library*. |

---

## 5. The big risk: catastrophic forgetting

The model learns by rewiring itself. If **100%** of tokens are Python source, the "old" skills (general reasoning, LaTeX, other languages, chatty prose) get overwritten — the model becomes a narrow Python-guesser that's suddenly worse at everything else.

**The fix is trivial: replay.** Mix back 25–50% general data (FineWeb-Edu + Wikipedia) into the blend. New knowledge gets encoded while old circuits keep firing. Two knobs control this:

| Knob | Effect |
|---|---|
| Domain replay % in the mixture | 25–40% is the standard range |
| Keep LR moderate | Violent weight rewrite is *how* forgetting happens |

---

## 6. The learning-rate recipe — the secret sauce of CPT

LR governs the tradeoff between **learning new** and **damaging old**:

- **Too high** → weights get violently rewritten → new stuff memorized-ish, old skills broken. (Like studying on 5 hours' sleep.)
- **Too low** → barely moves; the model "samples" the domain without absorbing it.
- **Right** → a few ×10⁻⁵-ish region for LLMs (CPT uses roughly 1/10 to 1/3 of the base's final pre-training LR).

Modern CPT/mid-training uses the **WSD schedule (Warmup → Stable → Decay)**:

```
loss
 │                    ╭ ╭ ← (small) final dip
 │                  /
 LR│  warmup ↗      / stable (cruise, most training)
 │      ___      _/
 │     /   \____/
 │    /
 └──────────────────────→ steps
    ~1%  |———— 60–80% ————|—— 20–30% ——
```

- **Warmup** (~few hundred steps): grow LR 0→peak, avoid wrecking a good model on step 1.
- **Stable**: cruise at peak — most learning happens here.
- **Decay** (to ~1/20–1/100 of peak): the "consolidation" phase — careful, final refinement. Loss always dips noticeably during decay.

**Why this lets us rescue interrupted sessions:** the *high-value state* is recoverable at any point in the stable zone; you branch a decay run for as long as remaining budget allows, and get a fine final model (this is also exactly how we handle a Kaggle session ending — checkpoint every ~1h and each resumed run re-warms-up at a small LR).

---

## 7. Data passing / batching details (in plain terms)

| Concept | Simple version |
|---|---|
| **Packing** | Concatenate docs into one stream, slice into fixed ~4096-token blocks. No padding = no wasted compute (and on TPU: one fixed shape = no recompile). |
| **Batch** | e.g., 64 blocks × 4096 tokens = **262k tokens/step** (on v5e-8). |
| **Update (step) count** | 300M tokens ÷ 262k ≈ ~1,100 steps. Each step is one LR nudge. |
| **bf16** | Everything in half precision — on TPU this is native, not a hack. |
| **AdamW** | The optimizer: keeps a running mean/variance of each weight's gradient history so nudges are well-scaled per-weight. Pure plumbing. |
| **Checkpointing** | Every ~1h: weights + optimizer state + data position → HF repo. A 9h session kill costs minutes, not the phase. |

---

## 8. Full fine-tune vs LoRA

| | Full fine-tune | LoRA |
|---|---|---|
| What changes | **All 4B weights** | Only tiny "adapter" matrices bolted onto attention/MLP layers (0.1–1% of params) |
| Memory | Weights + optimizer states ≈ 16×N bytes | Frozen base + small adapters |
| Knowledge uptake | ✅ Best — real capacity changes | ⚠️ Weaker for *large* new knowledge; fine for style/format |
| Fits v5e-8 (128GB) | ✅ bf16, yes | ✅ trivially |
| Fits 2×T4 (32GB) | ❌ | ✅ (QLoRA) |

**Our pick with the v5e-8 in play:** full bf16 for **CPT** (knowledge absorption is the whole point), then **LoRA is usually enough for SFT** (behavior-change step) — we can still full-FT later if data supports it.

---

## 9. How we'll know it worked (the checks)

| Check | Pass looks like |
|---|---|
| **Domain loss / perplexity** on held-out code+docs | Clearly ↓ vs base (e.g., ppl 8 → 5 on your docs) |
| **General benchmarks** (the base eval suite) | Flat/down < ~2–3% (forgetting contained) |
| **Quick probes** | Ask domain questions before/after: answers reference your corpus's concepts |
| Loss curve | Smooth drop during stable phase, extra dip during decay |

⚠️ Honest expectation: a 0.1–0.3B-token pass doesn't turn the model into a domain *oracle* — it shifts its "center of mass": better vocabulary of the domain, better intuitions, better behavior in Phases 5–6. **The bulk of factual recall will come from RAG (Phase 4)** — CPT makes the model a better *user* of that library.

---

## 10. Failure modes to watch (theory → practice checklist)

1. **LR too high** → eval collapse in both directions. Remedy: re-run from checkpoint with LR ÷2.
2. **Replay too low** → domain loss ↓ but general benchmarks crater → raise replay %.
3. **Contaminated data** (e.g., benchmark text inside corpus) → inflated scores, invalid. We decontaminate against eval suites.
4. **Duplicates** → model memorizes repeats, learning "plateaus" — we dedup at extraction.
5. **Wrong mixture ratio** → e.g., 95% code / 5% docs → model overfits file syntax, weak on *describing* — keep the blend diverse (code 40–50%, docs 30%, QA 10–20%, papers 5%, replay 25%+).
6. **No checkpoints** → a 9h session dies with 9h lost. Checkpoint every hour, resume at a tiny warmup.

---

## 11. Phase 3 — inputs / outputs (the contract)

| | |
|---|---|
| **In** | Qwen3-4B base · curated slice: 100–300M domain tokens (+100–150M replay) · LR plan (WSD: peak ~1e-4→2e-4, warmup ~1%, decay to ~1×10⁻⁶) · blend recipe |
| **Do** | Full bf16 next-token training on packed 4096 blocks · replay in the mix · checkpoint hourly to HF |
| **Out** | `cpt.v1` — same model, now domain-fluent enough for Phase 5 to build real behavior on top |

---

# Part II — The CPT Dataset: Selection, Cleaning & Preparation (detailed)

> Goal of this part: every token that enters Phase 3 has **earned its place** — license-clean, junk-free, decontaminated, uniquely useful — and the whole path from HF repo → packed training blocks is auditable and resumable.

---

## 12. The decision: public pre-cleaned corpus as the CPT supply

Our CPT budget is tiny (~0.1–0.3B tokens vs. their 2–5B slice plans). At this size, **data quality is the highest-ROI pre-processing in the entire project** — a bad 5% of tokens hurts more here than in any other phase, because every token is a permanent weight-nudge. Cleaning raw code dumps ourselves would burn days of CPU for a result measurably worse than a corpus whose cleaning was already validated by a published 7B training run.

**Decision:** `common-pile` (Common Pile v0.1, paper arXiv 2506.05209) — 8TB of public-domain and openly-licensed text, with a **filtered + deduplicated** version published per source. Its consolidated training set (`comma_v0.1_training_dataset`) trained **Comma v0.1-1T / -2T** (7B), which scores *comparably to budget-matched models trained on unlicensed data* — proof the cleaning works.

### ⚠️ Two gotchas found the hard way (so you don't repeat them)

| Gotcha | Truth |
|---|---|
| `common-pile/comma-v0.1-2t` | That is the **model** page (7B Llama-3-architecture base model), not the dataset. |
| `common-pile/comma_v0.1_training_dataset` | The consolidated dataset has **only one column: `text`** (784M rows, 521 GB). **No `source` field** — per-source labels were stripped in consolidation. We therefore do **not** use it directly. |

**The correct entry point:** the **"Common Pile v0.1 Filtered Data"** collection — the same data pre-consolidation, split into separate named repos. Each repo is already quality-filtered, deduplicated, and license-attributed *per source*. Downloading only the six we want = six small targeted downloads instead of one anonymous 521GB pile with no column to filter on.

---

## 13. The shopping list (verified Oct 2026, exact repo names)

| # | Repo (exact HF name) | Role in blend | Rows | Tokens (B, unique) |
|---|---|---|---|---|
| 1 | `common-pile/stackv2_edu_filtered` | **Code** — The Stack v2 run through the HuggingFaceTB "stack-edu" quality classifiers; **15 programming languages**, English | 57M | 67.8 |
| 2 | `common-pile/github_archive_filtered` | **Docs-ish prose** — real GitHub issues/PRs/comments ("explaining code in the wild") | 23.3M | 11.0 |
| 3 | `common-pile/stackexchange_filtered` | **Worked Q&A** — pre-parsed question+answers | 27.5M | 23.9 |
| 4 | `common-pile/wikimedia_filtered` | **General anchor** — Wikipedia plaintext | 12.9M | 15.8 |
| 5 | `common-pile/arxiv_papers_filtered` | **Formal register** — full arXiv papers | 309k | 6.0 |
| 6 | `HuggingFaceFW/fineweb-edu` (10BT slice) | **Replay** (anti-forgetting) — educational web text, already quality-scored | — | ~0.1 of 1B-token sample |

(Reference: the full Common Pile mixture that trained Comma-2T: stackv2_edu 67.8B tokens ×2 repeats, stackexchange 23.9B ×6, peS2o 43.3B ×6, wikimedia 15.8B ×6, github_archive 11.0B ×6, arxiv_papers 6.0B ×6 → 1034B effective. Our blend = the code+docs-heavy corner of that mixture, scaled to our budget.)

---

## 14. How each source was chosen — the 5 selection rules

1. **License-clean or filterable to license-clean.** Common Pile = open licenses only (public domain, CC, etc.); code carries per-repo SPDX metadata. We only *carry* the attribution ledger forward; we never launder licenses.
2. **Knowledge density.** Every slot should *teach* — source files (syntax/APIs), reference docs (explanations), Q&A (problem→solution pattern), papers (formal register). No chatter, no marketing pages.
3. **Domain match without monoculture.** All six sources are code-adjacent but disagree in style/format/register. Monoculture → the model learns one register and gets weird everywhere else. Same ratios, two tracks: when our own banked corpus is extracted later, these six slots swap in one-for-one.
4. **Pre-deduped.** All five Common Pile repos are deduplicated upstream. This is worth a lot: dedup is the most expensive cleaning step, and it's already done.
5. **Zero eval contamination by design.** Common Pile predates our eval sets, so decontamination is still *our* job (Section 19) — they could never have decontaminated against *our* Phase-7 benchmark list.

---

## 15. Division of labor: what they cleaned vs. what we do

| Cleaning stage | Who did it |
|---|---|
| Source curation (8TB raw, open licenses only) | ✅ Common Pile |
| Per-source quality filters, boilerplate stripping (public code: `r-three/common-pile`) | ✅ Common Pile |
| Toxicity tagging (Dolma tagger) | ✅ Common Pile |
| Exact + MinHash deduplication | ✅ Common Pile |
| Mixture design + repeat rates (their 463.6B-token run) | ✅ Common Pile (we borrow the ratios, re-scale for 0.3B) |
| **Verify their work (audit)** — trust but verify | ✅ **ours** |
| **Stratified slice** of their corpora to our quotas | ✅ ours |
| **Decontamination vs our evals** (HumanEval, MBPP, LiveCodeBench, HotpotQA…) | ✅ **ours — mandatory** |
| Shuffle → pack into fixed blocks → manifest → bank to HF | ✅ ours |

**Philosophy: verify, don't re-clean.** Four real steps for us: **① verify ② slice ③ decontaminate ④ pack.**

---

## 16. Step ① — Verify (the audit pass, ~30 min CPU per repo)

| Check | Pass criteria |
|---|---|
| Schema census | Columns match expectation per repo (text + per-source metadata; names differ per repo — confirm at runtime, never assume) |
| Token math | `rows × avg tokens ≈ published unique tokens` (e.g. stackv2: 57M × ~1.2k ≈ 67.8B ✓). Conclusion to check too: **repeat rates are NOT baked into rows** — Comma applied repeats at sampling time — so rows are unique docs and *we* control epochs (Section 18) |
| Exact-dup rate | sha256 on sampled texts ≈ 0%. (If > 1%, the consolidation duplicated by design → collapse on our side) |
| Length histogram | Smooth; no spikes at ~0 or ~1M tokens |
| Eyeball 20 random rows | Real text, no HTML skeletons, no base64 blobs, license headers sane |

**Verdict gate:** all pass → these six repos are the whole Track-A supply. Any fail → fall back to cleaning raw per-source data ourselves (the old heavy path).

---

## 17. Step ② — Take our slice (stratified, seed-pinned sampling)

Target blend for the first CPT pass — **domain 200M + replay 100M = 300M tokens**:

| Source | Quota | Share | Sampling rule |
|---|---|---|---|
| `stackv2_edu_filtered` (code) | **90M** | 45% | uniform random across the whole subset — never bias toward the first shards |
| `github_archive_filtered` (repo prose) | **40M** | 25% | uniform; length-trimmed to ≤ ~2k tokens per doc |
| `stackexchange_filtered` (Q&A) | **30M** | 15% | uniform; keep Q+A joined as one record (they already are) |
| `wikimedia_filtered` (anchor) | **20M** | 10% | uniform |
| `arxiv_papers_filtered` (papers) | **20M** | 10% | uniform |
| `fineweb-edu` replay | **100M** | of total 33% | uniform from the 10BT split |

Mechanics:

- **Seed-pinned uniform sampling** (`seed` recorded in the manifest) → re-running tomorrow yields the *identical* slice. Resumability starts here.
- **Long-doc policy:** docs > 4096 tokens keep their best contiguous window(s) or head+middle+tail windows. Never orphan 3-line leftovers.
- **Tiny-doc policy:** < 80 tokens → skipped.
- **One record = one self-contained span.** Source labels stay in the *manifest*, not in the text stream (CPT packs raw text; provenance is bookkeeping, not training signal).

---

## 18. Epoch logic (repeats — the "how often do we see it" decision)

The data-scaling literature (Muennighoff et al., *"Scaling Data-Constrained LMs"*) says: **up to ~4 epochs over the same tokens ≈ as good as fresh data; beyond that, gains collapse.** Our 300M-token budget is tiny, so we don't stretch repeats:

- **Plan: 3 epochs** over the 300M-token slice → ~900M effective tokens seen.
- No physical duplicates in the slice (they're unique per Section 16); repetition happens *during training*, at sampling time, exactly like Common Pile does it.
- If loss plateaus early during training (recitation signature), reduce to 2 epochs and pull fresh docs from the same subsets instead.

---

## 19. Step ③ — Decontamination (ours alone, non-negotiable)

1. Load eval surfaces: **HumanEval, MBPP, LiveCodeBench prompts**, plus any QA probes we plan (HotpotQA/MuSiQue-style).
2. Extract **13-grams** (13-byte/token windows) from every eval prompt. 13 is the standard: catches rewrites, survives paraphrase noise, few false positives.
3. Scan all 300M tokens; a doc matching any 13-gram → **kill the whole doc** (surrounding text is suspect too; don't surgical-edit mid-article).
4. Log removals into the manifest (`doc_id → eval_set`). Expected rate: **0.0–0.5%**.
5. **0% found is suspicious** → first assumption is that the matcher isn't running; verify with a planted canary (insert one eval prompt into a fake doc and confirm it's caught).

Why: any benchmark text that leaks into CPT makes the model self-trained on its exams — Phase 7 scores would be fiction.

---

## 20. Step ④ — Shuffle → pack → manifest → bank (the hand-off to training)

- **Global shuffle:** one seed-pinned permutation. Sources interleaved randomly — never "all Python in week 1, docs later." (The Comma run interleaves by up/downweight sampling; ours is simpler and fine at 0.3B scale.)
- **Pack:** tokenize once with the base model's tokenizer → continuous stream (docs joined with a separator, e.g. `

`) → slice into **fixed 4096-token blocks, int32, zero padding**:
  - 300M tokens → **≈73,240 blocks ≈ 600MB tokenized**, written as `cpt/block_NNNNN.bin` shards (~200MB each).
  - Fixed shapes = no recompilation on TPU; this is a *performance-critical* format choice, not tidiness.
- **`cpt_manifest.json`** (the audit + resume record):
  - per-repo: docs sampled, tokens, seed, dup-check results, decon removals (`doc_id → eval_set`)
  - per-block provenance index (which blocks carry which subsets — for debugging "loss won't drop")
  - tokenizer name + version; total epoch plan
- **Bank to the HF dataset repo** (same relay pattern as everything else): `data/cpt/block_*.bin` + `cpt_manifest.json`. A Kaggle session restart resumes at block N, not at zero.

---

## 21. Sanity marks from the Comma-2T model card (our recipe, validated)

| Their choice | Our equivalent |
|---|---|
| Main stage: 1.93T tokens, cosine LR | Our stable-phase cruising (WSD ≈ their design intent) |
| **Cooldown stage: 75.5B tokens from only the high-quality sources** | Our "decay phase with the domain-heavy blend" — precedent confirmed |
| **Final model = average of 10 cooldown checkpoints** | Checkpoint souping at decay end — we keep ≥3 decay checkpoints and average |
| Batch 8.3M tokens/step → 2T tokens ≈ 240k steps | Ours: 262k tokens/step × ~1,100 steps for 300M — same math, 1/200th scale |

---

## 22. Pre-flight QA gates (before the first GPU-second)

| Gate | Pass |
|---|---|
| Per-repo census matches published stats | ±20% |
| Slice ratio table (§17) | within ±3% |
| Exact-dup rate after our checks | ≈ 0% |
| Decon removals | 0.0–0.5% (and canary caught) |
| Length histogram | smooth, no spikes |
| Eyeball 10 packed blocks | real text, no HTML debris, no license-header spam |
| Replay share | 33% of blocks by token count |
| `cpt_manifest.json` complete | any block → provenance in one lookup |

---

## 23. The full pipeline, end to end

```
six *_filtered repos (+ fineweb-edu replay)
   │
   ├─ ① audit: census · rows×tokens math · sha256 dup check · histogram · eyeball
   ├─ ② slice: seed-pinned stratified quotas (90/40/30/20/20 + 100M replay)
   ├─ epochs: unique rows; repetition handled at training time (plan: 3 epochs)
   ├─ ③ decontaminate: 13-gram vs our evals → kill whole docs → log
   └─ ④ shuffle → pack into 4096-token int32 blocks → cpt_manifest.json → bank to HF
                          │
                          ▼
        data/cpt/block_*.bin  →  "ready for the GPU/TPU" input of Part I
```

**One-line summary:** the filtered Common Pile gives us a *pre-validated* wall of rejections; we add only the four steps they could never do for us — verify it, size it to our 300M budget, decontaminate against *our* exams, and pack it into TPU-shaped blocks with a manifest that makes every one of those tokens traceable.

---

## 24. Runbook — Kaggle TPU v5e-8 (free)

**One-time setup.** Add-ons → Secrets → add `HF_TOKEN` (write scope), attach it to the notebook. In the notebook:

```python
%cd /kaggle/working/code_rag
# %run cpt/cpt_train.py --libs          # optional: versions only
%run cpt/cpt_train.py --pilot 60        # 60-step smoke test + save/reload proof
```

On a TPU runtime the cell **auto-relaunches as a plain `python` process**; that is
required because `xmp.spawn` cannot pickle code defined in `%run`. Logs keep
streaming into the cell.

**What the pilot should show, in order**

```
libs: … torch … transformers … peft … torch_xla …
model: pre-downloading Qwen/Qwen3-4B …          # once, shared by all 8 cores
TPU: spawning one process per core (data-parallel) …
backend: TPU · world 8 · rank 0 · bf16
model: Qwen/Qwen3-4B · bf16 · attn sdpa → xla:0 …
peft: 504 lora matrices on 252 Linear layers
requires_grad rewired · trainable ~264M (lora-only)
[rank N] init sha1 <same for all 8>             # ranks 0…7 MUST print the same sha
model ready · trainable ~264M / 4.0B (6.5% · LoRA r=128)
probe baseline: loss ~2–4
[one silent pause of 1–4 min: XLA compiles every core]
grad check ✔ 504 LoRA tensors received grads
train: 60/60 …                          # ~1–3 s/step after compile
selftest: adapter save→load ✅ equal
✔ CPT done …
```

If the eight `init sha1` lines are **not all identical**, stop: the replica
seeds diverged and the run would silently train eight different models. (The
launcher seeds LoRA right before adapter creation; this line is the proof.)

If the loss is `nan` or flat while LR is moving, stop and inspect. If you see
`no LoRA gradients after the first backward`, checkpointing wiring failed — do
not let it run.

**The real run** (new cell, same session is fine):

```python
%run cpt/cpt_train.py
```

- 69,399 train blocks (512 held out) × 3 epochs → **1,084 steps/epoch · 3,252
  steps total · 262k tokens/step ≈ 852M tokens**, ~9 h at the 27k tok/s planning
  number (TPU quotas may end the session first — that is expected).
- Checkpoints go to `ckpt/cpt/step_NNNNNN/` on HF roughly every 55 min and at
  every probe (500 steps), with `latest.txt` updated last.
- Ending early is safe. On a fresh session, rerun the **same command** with no
  flags: it finds the newest complete checkpoint, loads adapter + optimizer,
  fast-forwards the deterministic data stream to the exact row, and continues.
- The final decay snapshots are averaged (soup), probed against the final
  model, merged into bf16 on CPU, and banked to `models/cpt.v1/`.

**If memory is tight** (each v5e chip has 16GB): rerun the pilot with
`--rank 64` (≈132M trainable, ~0.8GB less optimizer state); `--rank 32` is the
last resort. Keep gradient checkpointing on. `--attn eager` is *not* a memory
fix on TPU — it materialises attention scores.

**If `xmp.spawn` fails** on the Kaggle image, the env fix for
`TPU_PROCESS_ADDRESSES` is already applied in the launcher. Last-resort debug
path: `--single-core` (one core, ~8× slower — pilots only, not a full run).

**Token hygiene.** The token is read from `HF_TOKEN` env → Kaggle Secrets →
Colab userdata, never from the repo. Rotate the token you used during earlier
debugging before starting the multi-hour run.

---

*Next up (Phase 4 theory):* how we turn the *other* ~480GB of banked text into the library — chunking, embeddings, hybrid search, and contextual chunk prefixes so retrieval "knows" which repo/file it's in.
