# irm-2B
Ikirere Reasoning Model 2B

# 0) North Star and hard constraints

**Goal:** 2B param decoder-only model. 200k input, Live window 512k now → 1M later. Rolling 10M memory via retrieval. Text + code only. Tools baked in. Distillation approved for code + math. Exec-guided multi-sample allowed, but the **served** result must meet pass\@1 bar.

**Success gates (ship or don’t):**

* HumanEval pass\@1 ≥ 80 with exec-guidance, MBPP ≥ 85
* GSM8K ≥ 70 with self-consistency k≤5, MMLU core ≥ 70
* **p95 latency ≤ 200 ms/token on 200k-token prompts**; for longer contexts (512k and above) we amortize with cache/spec decoding and report real p95
* Long-context sanity: exact-match recall across **512k** with sparse markers ≥ 99% on synthetic probes

**Deliverable:** 1-page North Star with these metrics and how you’ll measure them.

---

# 1) Infra, repos, and roles

**Infra:** 20× H200 (FP8, TransformerEngine), FSDP/ZeRO-3, FlashAttention-3, TensorRT-LLM at serve.
**Repos (mono-org):**

* `tokenizer/` (build + eval)
* `data/` (ingest, clean, dedup, pack, stats)
* `pretrain/` (configs, sched, eval)
* `posttrain/` (SFT, DPO, exec-feedback, tools)
* `memory/` (Tiers 1–3)
* `serve/` (engines, router, budgets)
* `eval/` (bench harness, anti-leak, reports)
* `safety/` (policies, reward model, red-team)

**People:** owner per package; single DRI for decisions.

**Deliverable:** README per package with scope + interfaces.

---

# 2) Tokenizer (phase 1, then A/B)

**Choice:** SentencePiece **Unigram \~150k** vocab (code-biased, byte-fallback). Later **A/B** against **Llama-3 128k**.

**Training mix:** 60% code, 20% math/LaTeX, 20% high-quality text.

**Rules:**

* Preserve case; protect snake\_case, camelCase, `::`, `.`, `(){}[];`
* Specials: `<nl> <tab> <indent> <dedent> <scratchpad> <call_tool> <tool_result>`

**Acceptance:**

* Bytes-per-token ≤ 1.2 on code holdout
* Tokens/line on code ≤ Llama-3 baseline
* Round-trip detok 99.99% exact
* OOV ≈ 0 with byte-fallback

**Deliverables:** vocab files, metrics table, A/B plan vs Llama-3.

---

# 3) Data acquisition and hygiene

**Sources (permissive only):**

* Code: permissive subsets, issues/PRs, tests
* Math/algorithms: arXiv math/cs, StackExchange math/cs
* Text: refined web, wiki, books (cleaned)

**Cleaning:**

* URL + near-dup dedup (MinHash/SimHash; Jaccard < 0.8)
* Language ID; remove non-targets
* PII + policy filter (regex + classifier)
* Quality filter (small LM perplexity + doc-quality classifier)
* License filter for code; keep SPDX map

**Packing:** variable-length packing to maximize token throughput.

**Splits:** train/val/test by **domain buckets** with leak seals (hash families blocked across splits).

**Acceptance:**

* Dedup rate per domain reported
* License report (counts by SPDX)
* Tokens per bucket (target \~100B total; 40% code, 15% math, 30% text, 15% synthetic)

**Deliverables:** manifests, stats report, leakage policy doc.

---

# 4) Synthetic set & teachers (distillation: code + math)

**You approved distillation.** We do **signal-filtered** distill, not blind copying.

**Teacher stack:**

* Ensemble of strong open models (diverse) for code/math
* Programmatic teachers: unit tests, compilers, CAS/SMT
* Verifier model: checks steps/diffs, trained with corruptions

**Pipeline:**

1. Prompt tasks → ensemble answers (n=3–5)
2. Verify (run tests / math identity)
3. Score & keep only high-confidence signals (majority + verified)
4. Store rationales separately from final answers

**Schema (parquet):** `task_id, prompt, input_files[], expected_tests[], candidate[], passed, rationale, tool_calls[], confidence`

**Acceptance:**

* ≥ 95% of kept samples **programmatically verified**
* Confidence distribution reported
* No eval overlap (hash checks)

**Deliverables:** distilled SFT and DPO datasets.

---

# 5) Pretrain curriculum (to 512k)

**Model:** 2B decoder-only, \~30 layers, d\_model \~3072, heads 24 (GQA 6), ffn \~12288, RoPE, RMSNorm, SwiGLU.

**Context curriculum:**

* Stage A: train at **128k** context on 60–70% of tokens
* Stage B: scale to **512k** with RoPE/NTK scaling + **hybrid attention** (local 8–16k, sparse global tokens, ring/streaming links)
* Long-doc synthetic curriculum: interleave 20–30% long sequences; add retrieval-aug batches late

**Optim/infra:**

* AdamW (0.9, 0.95), wd=0.1, cosine LR, warmup 2–3%
* FP8 TE, FSDP ZeRO-3, grad-ckpt, FlashAttn-3
* Train by **tokens/step**; log FLOPs utilization

**Eval during pretrain:**

* Perplexity by domain bucket
* Long-context probes: exact span retrieval at 256k/512k
* If loss spikes after context jump: pause and bisect the recent data push

**Acceptance:**

* Stable loss vs tokens; no collapse after context jump
* ≥ 99% success on synthetic long-range probes at 512k

**Deliverables:** curves, probes, checkpoint catalog.

---

# 6) Extend to 1M (later)

**When:** after post-training V1 ships and is stable.

**Steps:**

* Increase local chunk; increase global stride density
* More long-doc synthetic + retrieval in batches
* KV quantization tuning (4–8b), eviction policy validated

**Acceptance:**

* Probes pass at 1M
* Latency profile within budgets using paging, with reported p95

**Deliverables:** ablation (512k vs 1M), serve configs.

---

# 7) SFT — reasoning style

**Data:** distilled math/code traces + curated QA.

**Conventions:**

* Tag rationales with `<scratchpad>…</scratchpad>`
* Final answers short; rationale optional at inference

**Acceptance:**

* No regression in generic text perplexity
* Rationale length distribution within target (concise)

**Deliverables:** SFT dataset cards, SFT checkpoints, style guide.

---

# 8) Preference learning — DPO first

**Pairs:** better/worse answers emphasizing **correctness, brevity, tool choice**; mix harmlessness prefs.

**Acceptance:**

* Win-rate uplift on held-out pairwise eval ≥ +10%
* No increase in refusal on safe coding tasks

**Deliverables:** DPO pairs, reward-free preference logs, tuned policy.

---

# 9) Tool-use SFT (baked tools, includes **web**)

**Tools registry:**

* `python_exec` (sandboxed)
* `compiler_run` (py, cpp, java, js)
* `retrieval` (Tier-1, tuned k, chunk 6–12k, multi-hop)
* `web` (allowlist, rate-limit; post-fetch parser/verifier)
* `calculator`

**Router head learns:** when to call, which tool, when to stop.

**Acceptance:**

* Tool precision/recall on synthetic tool-need tasks ≥ 90/90
* Cost budgets respected (bounded calls/task)

**Deliverables:** tool schemas, router metrics, misuse analysis.

---

# 10) Code — execution-feedback training

**Loop:** generate → run tests → read error → patch → verify.

**Signals:** pass/fail, error classes, diff quality.

**Serving policy:** **exec-guided decoding** allowed; N parallel samples behind the curtain; **return first passing sample**; report pass\@1 honestly.

**Acceptance:**

* HumanEval pass\@1 ≥ 80 with exec-guidance
* Repair success rate ≥ 60% on failing first attempts

**Deliverables:** EF dataset, repair taxonomy, benchmark report.

---

# 11) Safety alignment

**Constitutional SFT:** refusal style, policy compliance.
**Safety RM:** pairwise safety preferences.
**Short DPO** blend for helpful/harmless.
**Sandboxing:** strict caps for `python_exec`/`compiler_run` (CPU time, memory, FS isolation, net off by default).

**Acceptance:**

* Zero known unsafe tool calls in red-team suite
* No drop in code benchmarks from safety layers

**Deliverables:** policies, red-team results, safety RM card.

---

# 12) Memory hierarchy (10M effective)

**Tier-0:** live **512k** hybrid attention, GQA, KV quant/paging.
**Tier-1:** vector store over **10M tokens**

* Chunk **6–12k**, overlap \~10–15%
* HNSW/FAISS index; store embeddings, doc IDs, trust scores
* Query reformulation + **multi-hop** retrieve (2 hops)
* Post-retrieve filter: dedup, recency, trust

**Tier-2:** scratchpad → compressor

* Model writes notes; compressor produces **fact tables / code outlines**
* Keep lineage (source spans + hashes)

**Tier-3:** **SSM sidecar** (RetNet/Mamba-2)

* Ingests long logs and emits **memory frames** (outlines, keys)
* Frames get indexed into Tier-1 and appended into Tier-2

**Acceptance:**

* Retrieval hit-rate on eval tasks ≥ 90%
* Memory frame usefulness: ablation ≥ +5–10% lift on long-doc QA

**Deliverables:** index stats, retrieval policies, compressor eval, sidecar card.

---

# 13) Serving stack (to hit latency at 200k)

**Target clarified:** **p95 ≤ 200 ms/token for 200k-token prompts**.
For 512k/1M prompts: we **amortize** with prompt cache, paged KV, and speculative decoding; report real p95 and throughput.

**Compile path:** TensorRT-LLM, FP8, FlashAttn-3.
**Speculative decoding:** 2B target + \~300M draft, track acceptance rate.
**KV:** quant 4–8b, paged KV, sliding window; sparse global tokens kept hot.
**Prompt cache:** system + tool schemas; cache for frequent long doc headers.
**Budgets:** per-tool time/memory; async calls with strict caps.

**Acceptance:**

* **p95 ≤ 200 ms/token at 200k inputs** with live 512k window not fully saturated (we keep sliding window and globals efficient)
* Degradation curve documented for 512k and 1M prompts, with cache/spec improvements

**Deliverables:** serving profiles, budgets, fallbacks when draft acceptance drops.

---

# 14) Evaluation harness (anti-leak)

**Bench sets:** GSM8K, MATH subset, HumanEval, MBPP, MMLU core, ARC-C, HellaSwag.
**Anti-leak:** hash docs; check pretrain data vs eval; any overlap → drop and note.

**Long-context probes:**

* Needle-in-haystack across **200k** and **512k**
* Cross-segment reasoning (answer depends on two far-apart spans)
* Sparse-global token ablations

**Acceptance:**

* Meets North Star gates; no contamination findings

**Deliverables:** eval dashboard, raw logs, reproducible seeds.

---

# 15) Monitoring, drift, incidents

* Online eval slices (code, math, long-context)
* Tool misuse alerts (rate spikes, timeouts)
* Regression guardrails: canary routing to previous checkpoint
* Incident runbook: rollback, disable tool X, reduce globals/context, etc.

**Deliverables:** dashboards, alert thresholds, runbook.

---

# 16) Security & compliance

* Sandbox specs for tools; network egress policy for `web`
* Dataset license ledger; privacy posture; redaction logs
* Model card with known limits and mitigations

**Deliverables:** security review, model card, license report.

---

# 17) Notebook template (the “great notebook”)

**Sections:**

1. **Config**: model/optimizer/context/memory toggles
2. **Tokenizer report**: bpt, tokens/line, A/B vs Llama-3
3. **Data stats**: domain sizes, dedup, license counts
4. **Pretrain run**: curves, FLOPs, bucket losses
5. **Context probes**: 128k→512k results
6. **SFT**: rationale length hist, style examples
7. **DPO**: win-rates, length control
8. **Tools**: router precision/recall, cost
9. **Code EF**: pass/fail curves, repair taxonomy
10. **Safety**: harmlessness checks, sandbox tests
11. **Memory tiers**: retrieval hit-rate, compressor quality, sidecar ablation
12. **Serving**: **latency plots at 200k**, spec-decoding acceptance
13. **Final benches**: full scorecard vs gates
14. **Changelog**: what changed since last run

**Deliverable:** one canonical notebook per stage + an overview index.

---

# 18) A/B plan (locked before training)

* **Tokenizer:** Unigram-150k vs Llama-3-128k (compression, latency, code tokens/line)
* **Context:** 128k-only vs 128k→512k curriculum
* **Teachers:** ensemble size n=3 vs n=5; with/without programmatic gates
* **Retrieval:** k=4 vs k=8; chunk 6k vs 12k; multi-hop on/off
* **Spec decoding:** with vs without draft model

**Deliverables:** experiment table with pass/fail criteria.

---

# 19) Risk register

* 1M live window instability → hold at 512k until stable
* Teacher noise → programmatic verification **mandatory**
* Reward gaming (verbosity) → DPO length penalties + verifier
* Latency spikes at 200k → reduce globals, raise KV quant, tighten budgets, cache more aggressively

**Deliverable:** risk log with triggers/mitigations.

---

# 20) Handover pack (for partners)

* Model card + safety report
* Serving configs + Terraform/K8s manifests
* Eval harness + seeds
* Memory index bootstrap + maintenance scripts
* Versioned dataset manifests + hashes

**Deliverable:** zip + README.

---

## Final

* Tokenizer plan: Unigram-150k now; later A/B vs Llama-3-128k.
* Live context plan: ship 512k now; extend to 1M after V1 stabilizes.
* Distillation: approved for code + math, **always** programmatically verified.
* Tools: python, compiler, retrieval, web, calculator — routed with budgets.
* **Latency target updated:** **200 ms/token p95 at 200k-token inputs**.
