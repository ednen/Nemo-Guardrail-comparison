# Prompt Injection Classifier — Benchmarks

End-to-end evaluation of a custom-trained Llama-3.1-8B LoRA classifier for prompt injection detection, compared against Meta's Llama Prompt Guard 2 (86M) and NVIDIA's Llama Guard 3 (8B) across English and Turkish benchmarks.

## TL;DR

| Dimension | Result |
|---|---|
| **English PromptShield (2K)** | Custom LoRA wins: AUC 0.958, TPR@1%FPR 0.838 vs. PG2 0.171, LG3 0.004 |
| **English deepset (662)** | PG2 wins: AUC 0.923 vs. LoRA 0.838 vs. LG3 0.666 |
| **NeMo Guardrails integration (70)** | Custom LoRA: 97.2% recall, 15× lower latency than LLM-as-judge baseline |
| **Turkish benchmark (1170)** | Headline AUC 0.974 but per-layer breakdown shows **native Turkish attack recall is 2%** — model needs Turkish continued training before production use |

Three separate benchmark runs, three different conclusions about where the model is strong and where it isn't. Details below.

---

## 1. NeMo Guardrails integration benchmark

**Setup.** 70-sample curated test set across 9 categories (direct injection, indirect injection, jailbreak personas, prompt leak, delimiter attacks, tricky-benign, plus Turkish healthcare benigns). The Llama LoRA classifier is wired as a NeMo Guardrails input rail alongside NeMo's standard `self_check_input` rail backed by Llama-3.3-70B via Groq.

**Purpose:** verify the integration works end-to-end and compare the two rail strategies on a realistic, if small, sample.

### Results

| Metric | Llama LoRA (Focal Loss) | NeMo Self-Check (Groq) |
|---|---|---|
| Accuracy | 0.957 | 0.971 |
| Precision | 0.946 | **1.000** |
| Recall | **0.972** | 0.944 |
| F1 | 0.959 | **0.971** |
| AUC | **0.992** | 0.972 |
| Latency p50 | **149 ms** | 2,237 ms |
| Latency p95 | **154 ms** | 2,487 ms |

At F1-optimal threshold — the custom LoRA caught 35/36 injections with 2 false positives (recall 0.972, precision 0.946). Groq caught 34/36 with zero false positives.

**Confusion matrices**

Llama LoRA @ threshold = 2.9 × 10⁻⁶:
```
                  Predicted
                  Benign  Injection
True  Benign        32        2
      Injection      1       35
```

NeMo Self-Check @ threshold = 0.9:
```
                  Predicted
                  Benign  Injection
True  Benign        34        0
      Injection      2       34
```

### Per-category accuracy (F1-optimal threshold)

| Category | Llama LoRA | Groq |
|---|---|---|
| benign_general | 0.80 | 1.00 |
| benign_healthcare | 1.00 | 1.00 |
| benign_turkish | 1.00 | 1.00 |
| delimiter_attack | 1.00 | 1.00 |
| direct_injection | 1.00 | 1.00 |
| indirect_injection | **1.00** | 0.80 |
| jailbreak_persona | 1.00 | 1.00 |
| prompt_leak | 1.00 | 1.00 |
| tricky_benign | 0.67 | 0.88 |

**Read.** Custom LoRA wins on indirect injection (RAG-relevant) and recall. Groq wins on tricky-benign (benign prompts that look like attacks), reflecting broader world knowledge. Latency gap is 15×.

**Deployment implication.** For a security input rail where false negatives represent successful attacks, the recall + latency profile favors the custom model. A tiered deployment (fast LoRA primary, slow LLM-as-judge for borderline cases) is a reasonable production pattern.

---

## 2. Large-scale benchmark (PromptShield + deepset)

**Setup.** Three classifiers evaluated on two established injection benchmarks, 2K stratified samples each (Llama Guard 3 capped at 500 due to latency). Datasets:

- **PromptShield eval split** (`hendzh/PromptShield`) — Jacob, Alzahrani, Hu, Alomair, Wagner (2025) benchmark, curated from Ultrachat, LMSYS, Alpaca, natural-instructions, SPP, HackAPrompt
- **deepset/prompt-injections** — classic HF benchmark, simpler distribution, widely cited

### Models

- **Llama LoRA (Focal Loss)** — custom 8B fine-tune, LoRA rank 64, focal loss (α=0.75, γ=2.0), trained on PromptShield training split
- **Llama Prompt Guard 2 (86M)** — `meta-llama/Llama-Prompt-Guard-2-86M`, mDeBERTa-v3-base backbone, multilingual
- **Llama Guard 3 (8B)** — `meta-llama/Llama-Guard-3-8B`, generative safety classifier; injection is one category among 14

### Results

**Standard metrics (at F1-optimal threshold)**

| Model | Dataset | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Llama LoRA (Focal Loss) | PromptShield | 0.944 | 0.957 | 0.930 | **0.943** | **0.958** |
| Llama LoRA (Focal Loss) | deepset | 0.781 | 0.709 | 0.760 | 0.734 | 0.838 |
| Prompt Guard 2 (86M) | PromptShield | 0.764 | 0.695 | 0.940 | 0.799 | 0.837 |
| Prompt Guard 2 (86M) | deepset | 0.840 | 0.759 | 0.875 | **0.813** | **0.923** |
| Llama Guard 3 (8B) | PromptShield | 0.500 | 0.500 | 1.000 | 0.667 | 0.557 |
| Llama Guard 3 (8B) | deepset | 0.516 | 0.508 | 0.988 | 0.671 | 0.666 |

**Low-FPR metrics (PromptShield-style, production-headline)**

| Model | Dataset | TPR@0.05%FPR | TPR@0.1%FPR | TPR@0.5%FPR | TPR@1%FPR |
|---|---|---|---|---|---|
| **Llama LoRA (Focal Loss)** | PromptShield | **0.691** | **0.773** | **0.811** | **0.838** |
| Llama LoRA (Focal Loss) | deepset | 0.118 | 0.118 | 0.228 | 0.304 |
| Prompt Guard 2 (86M) | PromptShield | 0.018 | 0.018 | 0.159 | 0.171 |
| Prompt Guard 2 (86M) | deepset | 0.183 | 0.183 | 0.266 | 0.297 |
| Llama Guard 3 (8B) | PromptShield | 0.000 | 0.000 | 0.004 | 0.004 |
| Llama Guard 3 (8B) | deepset | 0.032 | 0.032 | 0.076 | 0.076 |

**Latency**

| Model | Dataset | p50 (ms) | p95 (ms) |
|---|---|---|---|
| Prompt Guard 2 (86M) | PromptShield | **5.4** | 5.6 |
| Prompt Guard 2 (86M) | deepset | 5.3 | 5.5 |
| Llama Guard 3 (8B) | PromptShield | 22.0 | 34.8 |
| Llama Guard 3 (8B) | deepset | 20.5 | 21.5 |
| Llama LoRA (Focal Loss) | PromptShield | 34.6 | 45.1 |
| Llama LoRA (Focal Loss) | deepset | 34.0 | 35.1 |

### Reading

**PromptShield (the realistic attack distribution).** The custom LoRA dominates. TPR@0.1%FPR of 0.773 means at a threshold that allows only 1 benign prompt in 1000 to be wrongly flagged, the model catches 77.3% of attacks. The PromptShield paper's headline Llama-3-1-8B model reports TPR@0.1%FPR = 0.6533 on the full eval split — the focal-loss variant in this work scores higher at 2K samples, suggesting focal loss contributes to low-FPR performance.

**deepset (the simpler distribution).** Prompt Guard 2 wins by AUC (0.923 vs 0.838) and F1 (0.813 vs 0.734). PG2's broader training (includes deepset-style attacks) gives it an edge on this distribution. The custom LoRA remains well ahead of Llama Guard 3 (0.838 vs 0.666) but gives up ground to PG2's training-distribution overlap.

**Llama Guard 3 is not a prompt injection detector.** AUC of 0.557-0.666 is near-random. At the default 0.5 threshold, accuracy is exactly 0.500 — it classifies every input as safe. The F1-optimal threshold collapses to ~0.05, which tells you the model has almost no discriminative signal for injection specifically. This is expected: Llama Guard 3 is trained for harmful content (violence, hate speech, CBRN, etc.); most PromptShield attacks don't match those categories. **Treat this result as confirmation that Llama Guard 3 should not be relied on as an injection detector.**

---

## 3. Turkish-language benchmark

**Setup.** Constructed test set across three layers since no publicly available Turkish prompt injection dataset exists (verified: PolyGuardPrompts, XSafety, MultiJail, AyaRedTeaming, PolygloToxicityPrompts all exclude Turkish).

| Layer | Method | Count |
|---|---|---|
| Translated attacks | Google-translated PromptShield + deepset injections | 645 |
| Native Turkish benign | `Metin/WikiRAG-TR` + curated healthcare questions | 476 |
| Native Turkish attacks | Hand-crafted attacks exploiting Turkish linguistic features (evidential markers, register shifts, code-switching, cultural framing) | 49 |
| **Total** | | **1,170** |

### Headline results

| Metric | Llama LoRA (Focal Loss) | Prompt Guard 2 (86M) |
|---|---|---|
| AUC | **0.974** | 0.935 |
| F1 (opt threshold) | **0.937** | 0.892 |
| TPR @ 1% FPR | **0.863** | 0.520 |
| TPR @ 0.1% FPR | **0.765** | 0.393 |
| Latency p50 | 191 ms | **25 ms** |

### The headline is misleading — look at the per-layer breakdown

The aggregate metrics above are dominated by the 645 translated-attack samples. The per-layer breakdown reveals where each model actually works:

**Injection recall @ threshold 0.5**

| Source | n | Llama LoRA | Prompt Guard 2 |
|---|---|---|---|
| PromptShield_translated | 400 | **0.603** | 0.360 |
| deepset_translated | 245 | 0.012 | **0.069** |
| **turkish_native_attack** | **49** | **0.020** | **0.143** |

**False positive rate on native Turkish benign @ threshold 0.5**

| Source | n | Llama LoRA | Prompt Guard 2 |
|---|---|---|---|
| turkish_native | 476 | 0.000 | 0.000 |

### Reading

**The model is broken on native Turkish attacks at realistic thresholds.** Recall of 2.0% on hand-crafted native Turkish attacks means 48 of 49 attacks get through. Prompt Guard 2 catches 7× more (14.3%) on the same set, reflecting mDeBERTa's multilingual pretraining advantage.

**The high headline AUC is a ranking metric, not an accuracy metric.** The model ranks Turkish attacks higher than Turkish benign on average, which pulls AUC to 0.974. But the absolute confidence scores are tiny — the F1-optimal threshold sits at 7.8 × 10⁻⁶, meaning the model is barely distinguishing attacks above benign. In production at a reasonable threshold, it will miss most Turkish attacks.

**The deepset_translated row is the clearest signal.** 245 attacks that are translations of the easiest English injections (`ignore previous instructions` style). The model catches 1.2% in Turkish — vs. 87.5% for the same attack style in English on the deepset benchmark above. The drop isn't a small shift in capability; it's a near-total loss of function.

**Why this happens.** The training corpus was essentially all English. Llama-3.1-8B's multilingual pretraining gave the model enough structural understanding of Turkish to rank attacks above benigns on average, but the learned classification boundary is calibrated around English attack patterns. Turkish-phrased attacks don't cross that boundary at production-relevant thresholds.

### Deployment recommendation

**Do not deploy the current model for Turkish input.** The headline numbers flatter it; actual Turkish attack detection at production thresholds is near-zero.

Required next step: continued training with Turkish injection data mixed in. Plan:

1. Generate 2-3K Turkish training samples (translated + LLM-synthesized native attacks, plus matched Turkish benigns)
2. Continued LoRA training, 1 epoch at lr=1e-5, rank 64 unchanged, mixed English + Turkish batches
3. Re-run this benchmark; target `turkish_native_attack` recall above 0.80 at threshold 0.5 with no regression on English PromptShield metrics
4. Hold out 10-20% of generated Turkish data for final evaluation (no train/test contamination)

---

## 4. Methodology notes

**Caching.** Every per-sample prediction is persisted to disk immediately after each model completes a dataset. Re-runs of any notebook skip already-completed combinations. This means a mid-run crash or API rate limit costs at most one (model × dataset) pair, not the whole benchmark.

**Model loading.** The Llama LoRA classifier requires force-loading the `score` classification head directly from `adapter_model.safetensors` — PEFT's `modules_to_save` auto-restore produced silently random-initialized weights across several library version combinations tested. The loading cell includes a weight-change verification step that aborts with a clear error if the head didn't restore correctly.

**Threshold selection.** All models report metrics both at the default 0.5 threshold and at the F1-optimal threshold derived from the precision-recall curve. The F1-optimal thresholds are computed on the test set itself, which introduces a small optimistic bias but is necessary because the custom model's `temperature_scaler.pkl` was saved with `T = 1.0` (default init, suggesting `.fit()` was never called during training). TPR@FPR metrics are ROC-derived and don't share this concern.

**Llama Guard 3 confidence extraction.** LG3 is generative, not a classifier. Its confidence score comes from softmax over the `safe` vs `unsafe` next-token logits only. This is the standard technique for turning a generative classifier into a scoring model.

**PG2 label calibration.** Prompt Guard 2's HuggingFace config returns generic label names (`LABEL_0`, `LABEL_1`). The loading code runs one known-benign and one known-malicious probe at initialization to empirically determine which class index represents "malicious," rather than trusting the default.

**Test set construction (Turkish).** Translated attacks use `deep-translator` with Google Translate (source=en, target=tr). Native Turkish benigns are sampled from `Metin/WikiRAG-TR` and a curated healthcare seed list. Native Turkish attacks are hand-crafted, covering agglutinative morphology, evidential/hearsay markers (`-miş`), formal/informal register shifts (`siz`/`sen`), Turkish-English code-switching, cultural authority framing, indirect injection, prompt leakage, and delimiter attacks adapted to Turkish contexts.

---

## 5. Caveats

**Sample size.** The large-scale benchmark used 2K stratified samples per dataset. This is enough to distinguish models at scale but not to establish tight confidence intervals on fine differences. The PromptShield result at 2K aligns with the paper's reported full-eval performance, suggesting the sample is representative.

**Threshold optimization on test data.** F1-optimal thresholds were derived on the same data used for evaluation. In production, threshold selection should happen on a held-out validation set. TPR@FPR metrics are immune to this issue.

**Calibration.** The custom model's saved temperature scaler had `T = 1.0` exactly, indicating `.fit()` was never called during training. Optimal thresholds land at values like 7.8 × 10⁻⁶, which are calibration artifacts rather than model issues. Post-hoc Platt or temperature scaling on held-out data would bring these to interpretable values without changing AUC, TPR@FPR, or any ranking-based metric.

**Llama Guard 3 comparison is off-label.** Using a general-purpose content safety classifier as a prompt injection detector is a category mismatch. The results are included to document empirically that this common substitution does not work, not to argue LG3 is a bad model in its actual trained task.

**Turkish test set size.** 1,170 samples total, with only 49 hand-crafted native attacks. Enough to establish that the model underperforms on native Turkish, not enough to fine-grain where exactly. A larger hand-crafted native attack set (200-500 samples) would give better per-attack-category insight.

**Train/test contamination check for PromptShield.** The benchmark used the eval split of `hendzh/PromptShield`; the custom model was trained on the train split. No overlap was observed, but the two were both derived from the same source corpora (Ultrachat, LMSYS, etc.) so some distributional similarity is expected and likely inflates the PromptShield numbers above what would be seen on fully out-of-distribution attacks.

---

## 6. Artifacts

**Notebooks**
- `nemo_guardrails_benchmark.ipynb` — small-scale NeMo integration test
- `large_scale_benchmark.ipynb` — PromptShield + deepset comparison
- `turkish_benchmark.ipynb` — Turkish test set construction and evaluation

**Metrics CSVs**
- `benchmark_summary.csv` — NeMo integration headline metrics
- `large_scale_metrics.csv` — per-(model, dataset) metrics for English benchmark
- `turkish_summary.csv`, `turkish_layer_breakdown.csv` — Turkish benchmark

**Per-sample predictions** (for error analysis)
- `results_{MODEL}_{DATASET}.csv` — per-sample confidence, prediction, latency

**Visualizations**
- `nemo_benchmark_results.png` — NeMo 4-panel comparison
- `large_scale_results.png`, `low_fpr_deep_dive.png` — English benchmark
- `turkish_results.png` — Turkish benchmark

**Deployment config**
- `guardrails_config/` — NeMo Guardrails config wiring the custom classifier as an input rail, with Turkish healthcare dialog flows

---

## 7. Conclusion

The custom-trained Llama LoRA classifier is production-ready for English prompt injection detection. On PromptShield — the realistic attack distribution — it achieves AUC 0.958 and TPR@1%FPR of 0.838, outperforming Prompt Guard 2 by 5× on low-FPR recall and Llama Guard 3 by two orders of magnitude. Integrated into NeMo Guardrails, it provides 15× lower latency than an LLM-as-judge baseline while matching F1.

For simpler attack distributions (deepset), Prompt Guard 2's broader training gives it the edge, suggesting value in a tiered deployment where PG2 handles simple attacks and the custom model handles PromptShield-distribution attacks that matter more operationally.

For Turkish-language input, the current model is not production-ready. Headline metrics flatter; actual recall on native Turkish attacks at practical thresholds is 2%. Continued training with Turkish injection data is required before deployment in Turkish-serving contexts.

Llama Guard 3 is a content safety classifier and does not function as a prompt injection detector. Deployments that rely on it for injection protection are effectively unprotected against this threat.
