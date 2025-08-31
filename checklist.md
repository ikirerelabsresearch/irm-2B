# SLM-2B Program — Stage-by-Stage Checklist

## 0) North Star

* [ ] Document goals: 2B decoder, 512k live now → 1M later, 10M memory via retrieval, text+code, tools baked, distill approved for code+math.
* [ ] Ship gates: HumanEval p\@1 ≥ 80 with exec-guidance, MBPP ≥ 85, GSM8K ≥ 70 (SC k≤5), MMLU core ≥ 70.
* [ ] Latency SLO: p95 ≤ 200 ms/token at 200k-token inputs. Record degradation for 512k, 1M.
* [ ] Long-context sanity: ≥ 99% exact span recall at 512k on synthetic probes.
  **Artifact:** 1-page North Star with metric definitions and measurement scripts listed.

## 1) Org & Infra

* [ ] Repos: `tokenizer/ data/ pretrain/ posttrain/ memory/ serve/ eval/ safety/`
* [ ] Assign a DRI per repo. Define interfaces.
* [ ] Infra: 20× H200, FP8+TE, FSDP ZeRO-3, FlashAttn-3, TensorRT-LLM for serving.
  **Artifacts:** READMEs per repo, cluster inventory, kernel and driver versions.

## 2) Tokenizer v1 then A/B

* [ ] Train **SentencePiece Unigram \~150k**, code-biased, byte-fallback.
* [ ] Specials: `<nl> <tab> <indent> <dedent> <scratchpad> <call_tool> <tool_result>`.
* [ ] Protect snake\_case, camelCase, `:: . () {} [] ;`.
* [ ] Holdouts: code repos + math LaTeX + text.
* [ ] Metrics: bytes/token ≤ 1.2 on code, tokens/line ≤ Llama-3, round-trip 99.99%, OOV≈0.
* [ ] Prepare **A/B vs Llama-3 128k** after pretrain Stage A.
  **Artifacts:** vocab files, tokenizer report, A/B plan.

## 3) Data Hygiene

* [ ] Sources: permissive **code** 40, **math/algos** 15, **text** 30, **synthetic CoT** 15 (percent of tokens).
* [ ] Dedup: URL + near-dup (MinHash/SimHash). Keep Jaccard < 0.8 across sets.
* [ ] Language ID, PII/policy filter, perplexity + doc-quality filter.
* [ ] SPDX license mapping for code. Drop non-permissive.
* [ ] Packing: variable-length sequence packing.
* [ ] Splits with leak seals by domain.
  **Artifacts:** manifests, license ledger, stats report, leak policy.

## 4) Distilled Datasets (code + math)

* [ ] Teacher ensemble n=3–5, diverse. Add programmatic teachers: unit tests, compilers, CAS/SMT.
* [ ] Verifier model to grade steps/diffs. Train with corruptions.
* [ ] Keep only verified majority-agree outputs. Confidence scores stored.
* [ ] Schema (parquet):
  `task_id, prompt, files[], tests[], candidate, passed, rationale, tool_calls[], confidence`
* [ ] ≥ 95% of kept samples programmatically verified. Zero eval overlap.
  **Artifacts:** distilled SFT set, DPO pairs, confidence histograms.

## 5) Pretrain to 512k

* [ ] Model: 2B, ~~30 layers, d\_model~~3072, heads 24 (GQA 6), ffn\~12288, RoPE, RMSNorm, SwiGLU.
* [ ] Optim: AdamW β 0.9/0.95, wd 0.1, cosine, warmup 2–3%.
* [ ] Precision/infra: FP8 TE, FSDP ZeRO-3, grad-ckpt, FlashAttn-3.
* [ ] **Context curriculum**: 128k for 60–70% tokens → scale to **512k** with RoPE/NTK scaling + **hybrid attention**
  local 8–16k, sparse global tokens, ring/streaming links.
* [ ] Add 20–30% long sequences; late-stage RAG-augmented batches.
* [ ] Eval every 2–4k steps: bucketed loss, long-context probes 256k/512k.
  **Gate:** stable loss; ≥ 99% long-range probe success at 512k.
  **Artifacts:** curves, probe report, ckpt catalog.

## 6) Extend to 1M (post-V1)

* [ ] Increase local chunk, raise global stride density.
* [ ] More long-doc synthetic + RAG batches.
* [ ] KV quant 4–8b tuning, eviction policy validated.
  **Gate:** probes pass at 1M; serving p95 documented.
  **Artifacts:** 512k vs 1M ablation.

## 7) SFT — Reasoning

* [ ] Train with distilled math/code traces. Tag `<scratchpad>…</scratchpad>`.
* [ ] Keep final answers terse; rationale optional at inference.
  **Gates:** no generic perplexity regression; median rationale length target met.
  **Artifacts:** SFT dataset card, checkpoints, style guide.

## 8) DPO — Preference Learning

* [ ] Build pairs emphasizing correctness, brevity, tool choice. Blend harmlessness prefs.
* [ ] Train DPO. Monitor length control and refusal rates.
  **Gate:** ≥ +10% win-rate on held-out pairs.
  **Artifacts:** pairs, policy deltas, refusal audit.

## 9) Tool-Use SFT (includes **web**)

* [ ] Tools: `python_exec`, `compiler_run` {py, cpp, java, js}, `retrieval` {k tuned, chunk 6–12k, multihop}, `web` {allowlist, rate-limit, parser/verifier}, `calculator`.
* [ ] Router head learns when/which/stop. Budgeted calls.
  **Gate:** precision/recall ≥ 90/90 on synthetic tool-need suite.
  **Artifacts:** tool schemas, router PR/RC report, misuse cases.

## 10) Code — Execution-Feedback

* [ ] Loop: generate → run tests → read error → patch → verify.
* [ ] Signals: pass/fail, error class, diff quality.
* [ ] Serving: exec-guided decoding, N parallel samples behind the curtain, **return first passing sample**; report pass\@1.
  **Gates:** HumanEval p\@1 ≥ 80 with exec-guidance. Repair ≥ 60% on first-fail.
  **Artifacts:** EF dataset, repair taxonomy, benchmark report.

## 11) Safety

* [ ] Constitutional SFT.
* [ ] Safety RM + short DPO helpful/harmless.
* [ ] Sandboxing: strict CPU/mem/time caps, FS isolation, network off by default for exec tools.
  **Gates:** zero unsafe tool calls on red-team suite; no code benchmark drop.
  **Artifacts:** policies, RM card, red-team logs.

## 12) Memory Hierarchy — 10M effective

* **Tier-0** live: 512k hybrid attention, GQA, KV quant/paging.
* **Tier-1** retrieval: 10M-token vector store
  chunks 6–12k, \~10–15% overlap, HNSW/FAISS, multihop=2, dedup+recency+trust.
* **Tier-2** scratchpad→compressor: fact tables, code outlines, with lineage.
* **Tier-3** sidecar SSM (RetNet/Mamba-2): emits “memory frames” into Tier-1/2.
  **Gates:** retrieval hit-rate ≥ 90%; sidecar ablation +5–10% on long-doc QA.
  **Artifacts:** index stats, compressor eval, sidecar card.

## 13) Serving — 200k input SLO

* [ ] Compile path: TensorRT-LLM, FP8, FlashAttn-3, paged KV, sliding window, sparse globals hot.
* [ ] Speculative decoding: 2B target + \~300M draft, track acceptance.
* [ ] Prompt cache: system + tool schemas, frequent headers.
* [ ] Budgets: per-tool time/mem, async calls.
  **Gates:** p95 ≤ 200 ms/token at 200k inputs. Record curves for 512k and 1M with cache/spec.
  **Artifacts:** latency profiles, acceptance curves, fallback policy.

## 14) Evaluation Harness

* [ ] Benches: GSM8K, MATH subset, HumanEval, MBPP, MMLU core, ARC-C, HellaSwag.
* [ ] Anti-leak: hash checks against pretrain. Any collision → drop and note.
* [ ] Long-context probes at 200k and 512k. Cross-segment reasoning tests.
  **Gate:** meets North Star gates.
  **Artifacts:** dashboard, raw logs, seeds.

## 15) Monitoring & Incidents

* [ ] Online eval slices. Tool misuse alerts. Canary rollback to previous ckpt.
* [ ] Runbook: disable tool X, reduce globals, lower context, raise KV quant, etc.
  **Artifacts:** dashboards, alert thresholds, incident runbook.

## 16) Security & Compliance

* [ ] Egress policy for `web`.
* [ ] License ledger and privacy posture.
* [ ] Model card with limits and mitigations.
  **Artifacts:** security review, model card, license report.

## 17) Notebook Set — Index + Stage Notebooks

* [ ] One **overview notebook** linking to per-stage notebooks below.
* [ ] Per-stage notebooks: Config → Tokenizer → Data → Pretrain → Context Probes → SFT → DPO → Tools → Exec-Feedback → Safety → Memory → Serving → Final Benches → Changelog.
  **Artifacts:** 1 index + 13 stage notebooks with charts and tables only.

## 18) A/B Plan (locked pre-train)

* [ ] Tokenizer: Unigram-150k vs Llama-3-128k.
* [ ] Context: 128k-only vs 128k→512k curriculum.
* [ ] Teachers: ensemble n=3 vs n=5; with/without programmatic gates.
* [ ] Retrieval: k=4 vs 8; chunk 6k vs 12k; multihop on/off.
* [ ] Spec decoding: with vs without draft.
  **Artifacts:** experiment table with pass/fail criteria.

## 19) Risk Register

* [ ] 1M instability → hold at 512k until stable.
* [ ] Teacher noise → programmatic verification mandatory.
* [ ] Reward gaming → DPO length penalties + verifier checks.
* [ ] Latency spikes at 200k → reduce globals, raise KV quant, tighten budgets, cache more.
  **Artifacts:** active risk log with triggers and mitigations.

## 20) Handover Pack

* [ ] Model card, safety report.
* [ ] Serving configs, infra manifests.
* [ ] Eval harness + seeds.
* [ ] Memory index bootstrap + maintenance scripts.
* [ ] Versioned dataset manifests + hashes.
  **Artifact:** deliverable ZIP + README.

---

# Minimal Spec Blocks to Copy Into Repo

## tools.json

```json
{
  "tools": [
    {"name":"python_exec","timeout_ms":1500,"mem_mb":256},
    {"name":"compiler_run","langs":["py","cpp","java","js"],"timeout_ms":3000,"mem_mb":512},
    {"name":"retrieval","k":6,"chunk_tokens":[6000,12000],"multihop":2},
    {"name":"web","allow_domains":["docs","standards","reliable_news"],"rate_limit_qpm":6,"post_fetch_verifier":true},
    {"name":"calculator"}
  ]
}
```

## attention.yaml

```yaml
attention:
  scheme: hybrid
  local_chunk: 8192
  global_stride: 1024
  global_tokens: learned_summaries
  streaming_links: ring
  rope_scaling: yarn
  gqa: 6
  kv_quant_bits: 4
```

## context.yaml

```yaml
context:
  train_base: 128000
  extend_at_tokens: 0.7
  target_stage1: 512000
  target_stage2: 1000000
  latency_slo_tokens: 200000
```

## tokenizer.yaml

```yaml
tokenizer:
  algo: sentencepiece_unigram
  vocab_size: 150000
  byte_fallback: true
  preserve: ["_", "/", ".", "::", "()", "{}", "[]", ";", ","]
  special_tokens: ["<bos>","<eos>","<pad>","<nl>","<tab>","<indent>","<dedent>","<scratchpad>","<call_tool>","<tool_result>"]
  training_mix: {code: 0.60, math: 0.20, text: 0.20}
  eval_metrics: ["bytes_per_token","tokens_per_line_code","oov_rate","roundtrip_detok"]
```

---

# Notebook Outline (Index)

1. **Config**
   what: model dims, context targets, tools registry, SLOs
   show: one table of hyperparams and SLOs

2. **Tokenizer Report**
   what: Unigram-150k metrics + A/B plan vs Llama-3-128k
   show: bytes/token, tokens/line on code, round-trip, OOV

3. **Data Stats**
   what: token counts by domain, dedup rates, license ledger
   show: stacked bars, dedup table

4. **Pretrain Curves**
   what: loss vs tokens by domain, FLOPs util
   show: line charts, alerts if spikes

5. **Context Probes**
   what: 200k and 512k needle tests, cross-segment reasoning
   show: pass rates by distance, ablations

6. **SFT Reasoning**
   what: rationale policy, length hist, examples
   show: histograms, 3 anonymized samples

7. **DPO**
   what: win-rates, length control, refusal delta
   show: pairwise win curves, refusal table

8. **Tool-Use**
   what: router PR/RC, cost per task, misuse cases
   show: PR/RC table, cost boxplots

9. **Execution-Feedback Code**
   what: pass\@1 curves, repair taxonomy, exec budget use
   show: pass/fail over time, error classes

10. **Safety**
    what: harmlessness checks, RM calibration, sandbox tests
    show: ROC for safety RM, tool-sandbox outcomes

11. **Memory Tiers**
    what: retrieval hit-rate, compressor quality, sidecar ablation
    show: hit-rate vs k, BLEU/ROUGE for compressor, ablation bars

12. **Serving**
    what: p95 latency at **200k** inputs, spec-decoding acceptance, KV paging stats
    show: latency CDFs, acceptance line, KV residency heatmap

13. **Final Benches**
    what: GSM8K, MATH subset, HumanEval, MBPP, MMLU core, ARC-C, HellaSwag
    show: scorecard vs ship gates

14. **Changelog**
    what: date, change, effect on metrics
    show: table with links to runs
