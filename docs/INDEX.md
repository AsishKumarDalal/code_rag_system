# 📚 code_rag · map of phases, docs, and scripts

Every phase = one folder for code + one central doc (copied into the phase
folder). HF dataset repo `sddfhy546y/code_rag_dataset` is data ground truth;
git holds code/docs only.

| Phase | What | Doc (central) | Code | HF artifacts |
|---|---|---|---|---|
| 0/1 | corpus fetch + extract | — | `data/injestion.py`, `data/extraction.py` | `data/stackexchange/…`, `data/wiki/…`, `manifest.json` |
| 2 | theory: lifecycle/SFT/RL | *(phase 1-2 notes in chat/old docs)* | — | — |
| 3 | **CPT** (knowledge → weights) | [`docs/phase3.md`](phase3.md) ≡ `cpt/docs/phase3.md` | `cpt/cpt_prep.py`, `cpt/cpt_train.py` | `data/cpt/*_block_*.bin`, `cpt_manifest.json`, `ckpt/cpt/*`, `models/cpt.v1/` |
| 4 | **RAG** (the library, no training) | [`docs/phase_4_rag.md`](phase_4_rag.md) | (*later:* `rag/`) | `data/rag/…` index shards (later) |
| 5 | **SFT** (behavior → weights) | [`docs/phase_5_sft.md`](phase_5_sft.md) ≡ `sft/docs/phase_5_sft.md` | `sft/sft_build.py`, `sft/sft_train.py` | `data/sft/episodes_*.jsonl`, `sft_manifest.json`, `ckpt/sft/*`, `models/sft.v2/` |
| 6 | RLVR/GRPO (judgment) | *(pending)* | (*pending:* `rl/`) | `ckpt/rl/*`, `models/rl.v3/` |
| 7 | eval + serve | *(pending)* | (*pending*) | — |

Lineage: `Qwen3-4B → cpt.v1 → sft.v2 → rl.v3` — every checkpoint keeps its
parent link in metadata.

Hardware reality (re-confirmed Oct 2026): **GPU-first** (2×T4; Kaggle TPU not
available now) — all training scripts run on CUDA; keep `/kaggle/working`
under quota (<2GB peak: everything banks to HF; caches live on `/tmp`).
