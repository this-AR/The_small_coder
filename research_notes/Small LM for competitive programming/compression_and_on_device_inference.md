# Compression and On-Device Inference for Small Reasoning/Code LLMs (~1.5B Qwen2 architecture, phones)

Source-reliability tags used below: **[V]** = vendor/author self-reported about their own product or method; **[I]** = independent measurement (third-party academic or individual tester); **[S]** = secondary summary or search snippet only (primary not opened in full, so verify before quoting). Research date: 2026-11-19. Most numbers come from 2025–2026 papers and docs. Where something is superseded, that is flagged.

---

## 1. Quantization (PTQ, QAT, extreme low-bit, GGUF): measured accuracy impact on reasoning/code for ~1–2B models

### Takeaway
For the exact base model (DeepSeek-R1-Distill-Qwen-1.5B):
- **8-bit is effectively lossless.** This holds for W8A8KV8 with QuaRot/FlatQuant.
- **4-bit weight-only (AWQ/GPTQ) costs about 2–2.5 average points**, but on code it costs 2.5–4.8 points of LiveCodeBench. That is a 15–27% relative drop. Code is one of the most quantization-sensitive tasks at this size.
- **3-bit weight-only and 4-bit weight+activation (W4A4) cut accuracy sharply.** They can also make chain-of-thought (CoT) up to twice as long, which is a direct latency and battery cost on a phone.
- **Getting to 2–3 bits needs QAT (quantization-aware training) with distillation.** 2026 papers show that on-policy distillation (training on the quantized model's own generations) is what brings back long-form math and code reasoning.

### Cited Findings

#### PTQ on the exact base model (DeepSeek-R1-Distill-Qwen-1.5B)
- **BF16 baseline:** AIME-120 23.3, MATH-500 84.7, GSM8K 84.5, GPQA-Diamond 36.2, LiveCodeBench (LCB) 16.4; average 49.0. **[I]** — [Liu et al., "Quantization Hurts Reasoning?", arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2)
- **W4A16** (4-bit weights): AWQ average 46.9 (−2.1; LCB 13.9). GPTQ 46.6 (−2.4; LCB 13.4). — [arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2)
- **W3A16:** AWQ 30.4 (−18.6; LCB 3.2; MATH-500 48.1). GPTQ 41.4 (−7.6; LCB 8.7). — [arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2)
- **W8A8KV8:** SmoothQuant 45.2 (−3.9, LCB 15.7). QuaRot 49.0 (−0.1). FlatQuant 49.2 (+0.2; LCB 16.8). — [arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2)
- **W4A4KV4:** MXFP4 4.0 and QuaRot 2.9 (collapse). FlatQuant 38.2 (−10.9; LCB 7.5). — [arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2)
- **KV-cache-only quantization, 1.5B:**
  - KV4 with KVQuant*: −1.4 (LCB 15.9).
  - KV4 with QuaRot: collapses to an average of 2.5.
  - KV3 with KVQuant*: −12.3.
  - The authors blame extreme outlier channels in the K cache of the 1.5B and 7B models.
  - — [arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2)
- **Comparison with 7B:** at W4 the 7B loses much less on code (LCB 36.7 → 35.1 AWQ / 33.3 GPTQ). The paper states that smaller models are more sensitive to quantization. — [arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2)
- **Calibration data matters at 3 bits (Table 2 ablation, 1.5B):** for GPTQ 3-bit, calibrating on Numina-Math instead of WikiText2 raised the average from 28.3 to 38.1 (+9.8). AWQ, KVQuant*, SmoothQuant and FlatQuant moved by at most about 1.8. — [arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2)
- **The paper's recommendations:**
  - W8A8 and W4A16 are the "lossless" options.
  - 3-bit weights are risky.
  - W4A4 is still an open problem for reasoning models.
  - Prefer AWQ for weight-only quantization, and KVQuant* for KV on small models.
  - Large low-bit models beat smaller BF16 models at similar memory.
  - Quantized models gain less from longer test-time reasoning than BF16 models.
  - — [arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2)
- **Independent re-measurement with CoT lengths, same 1.5B model** (Lotfi, Kirichenko, Li, Liu; submitted 29 May 2026). Format: accuracy % / average CoT tokens.

  | Config | LiveCodeBench | MATH-500 | AIME-120 |
  |---|---|---|---|
  | BF16 | 17.5 / 19,066 | 85.6 / 5,156 | 24.4 / 17,465 |
  | GPTQ W4 | 12.7 / 20,529 | – | – |
  | AWQ W4 | 12.7 / 23,711 | – | – |
  | GPTQ W3 | 6.3 / 25,854 | – | – |
  | AWQ W3 | 4.5 / 38,793 | 47.0 / 23,444 | 4.4 / 33,015 |
  | FlatQuant W4A4KV4 | 7.5 / 25,383 | – | – |
  | FlatQuant W8A8KV8 | 17.2 / 19,947 | – | – |

  — [arXiv 2606.00206 (HTML)](https://arxiv.org/html/2606.00206v1)
- **"Overthinking" failures, from the same paper.** These are cases where the model reached the correct answer mid-trace but did not commit to it.
  - Share of MATH-500 errors: 26% for BF16 (19/72), 52% for AWQ-3-bit (139/265, a 7.3× increase), about 38% for FlatQuant W4A4KV4.
  - **Fix:** a training-free logit penalty on about 50 "overthinking markers" ("Wait", "But", "Alternatively", "perhaps", "reconsider"...).
  - Effect: CoT is 12–23% shorter on average.
  - AWQ-3-bit 1.5B: accuracy 29.0 → 35.2, CoT 28.0K → 21.6K tokens (−28%), overthinking errors 139 → 58.
  - BF16 1.5B: accuracy 48.5 → 50.7, CoT 11.2K → 10.1K. LCB stayed at 17.5 while CoT went 19.1K → 18.3K.
  - — [arXiv 2606.00206 abstract](https://arxiv.org/abs/2606.00206); [HTML](https://arxiv.org/html/2606.00206v1)
- **Conflict on output length.**
  - 2504.04823 concludes that quantized models "do not generally think longer". It also notes that aggressive settings (W3, W4A4KV4) lengthen outputs, most strongly for the 1.5B.
  - 2606.00206 argues that aggressive PTQ consistently lengthens CoT.
  - Both agree that mild quantization (W4 weight-only, W8A8) keeps lengths close to BF16.
  - — [arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2); [arXiv 2606.00206](https://arxiv.org/html/2606.00206v1)

#### Larger distilled reasoning models (context for scale effects)
- **4-bit:** AWQ, GPTQ, GPTAQ and ANY4 are near-lossless on R1-Distill-Llama-70B (81.8 → 80.4–81.2).
- **4-bit at 7–8B:** GPTQ hurt (Llama-8B 65.2 → 58.1; Qwen-7B 66.8 → 60.7), while AWQ held (66.6; 65.7).
- **3-bit collapsed** (Llama-8B GPTQ 47.8, ANY3 29.4; Qwen-7B GPTQ 38.4).
- **Protecting weights helps:** keeping the final-layer MLP modules (about 2% of weights) in 16-bit raised 3-bit AWQ on Llama-8B from 46.0 to 52.57.
- **Extreme low-bit on the full model:** Unsloth's dynamic 2.51-bit R1 (671B) averaged 84.8, versus 83.1 at full precision.
- — [Zhang et al., "When Reasoning Meets Compression", arXiv 2504.02010v2](https://arxiv.org/html/2504.02010v2)

#### Code-specific quantization studies (non-reasoning code LLMs, mostly ~7B)
- **Lua code generation, five ~7B code LLMs at 2/4/8-bit:**
  - Going from 2-bit to 4-bit helps much more (0.668) than going from 4-bit to 8-bit (0.094).
  - CodeGemma produced incoherent output at 2-bit.
  - **[S]** — [Moonlight review](https://themoonlight.io/review/evaluating-quantized-large-language-models-for-code-generation-on-low-resource-language-benchmarks)
- **"Differentiated replication" (arXiv 2503.07103; CodeLlama, DeepSeek-Coder; 8/4/3/2-bit):** 3-bit and 2-bit benefit strongly from calibration sets that include code, and larger models are more resilient. **[S]** — [Moonlight review](https://www.themoonlight.io/review/quantizing-large-language-models-for-code-generation-a-differentiated-replication)
- **Afrin et al. (arXiv 2607.14181; GPTQ, AWQ, QuIP#, AQLM, BitsAndBytes, GGUF on Qwen2.5-Coder/CodeLlama):** AQLM matches or exceeds full precision, and QuIP# degrades the most. **[S]** — [papers.cool listing](https://papers.cool/arxiv/2607.14181)

#### QAT (quantization-aware training)
- **Meta Llama 3.2 1B/3B: QAT+LoRA and SpinQuant (a PTQ method).** **[V]**
  - Scheme: 4-bit groupwise weights (group 32) with 8-bit dynamic per-token activations in linear layers; 8-bit per-channel embeddings and classifier.
  - On average: 56% smaller, 41% less memory, 2–4× faster.
  - Quality: QAT+LoRA reaches "competitive accuracy to the BF16 model". SpinQuant is less accurate than QAT+LoRA but needs no training data.
  - — [Meta AI blog](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)
- **Gemma 3 QAT.** **[V]**
  - About 5,000 QAT steps, using the non-quantized checkpoint's probabilities as targets.
  - Google says this "reduces the perplexity drop by 54%" at Q4_0 (llama.cpp perplexity).
  - Gemma 3 1B weights shrink from 2 GB to 0.5 GB.
  - — [Google Developers Blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)
  - The Decoder notes that Google released no updated benchmarks to back the robustness claim. — [The Decoder](https://the-decoder.com/gemma-3-27b-it-qat-q4_0-gguf-sounds-like-a-wi-fi-password-but-its-googles-leanest-ai-yet/)
- **ParetoQ (NeurIPS 2025):**
  - It finds a "learning transition between 2 and 3 bits". At 3 bits and above, models stay close to the pretrained distribution; at 2 bits and below, representations change drastically.
  - Its ternary 600M model beats the previous state-of-the-art ternary 3B model.
  - Ternary, 2-bit and 3-bit perform comparably, and generally beat 4-bit and binary on the size-accuracy trade-off.
  - 2-bit is the most promising for hardware.
  - — [arXiv 2502.02631](https://arxiv.org/abs/2502.02631)
- **"What Makes Low-Bit QAT Work for Reasoning LLMs?" (Jan 2026):**
  - Knowledge distillation is a robust QAT objective, under both SFT and RL.
  - A PTQ checkpoint is a strong starting point.
  - RL works on quantized models if they get a viable cold start.
  - Matching the PTQ calibration domain to the QAT data domain helps.
  - On Qwen3-0.6B, the resulting "Reasoning-QAT" "surpasses GPTQ by 44.53% on MATH-500" and recovers performance in the 2-bit regime.
  - — [arXiv 2601.14888](https://arxiv.org/abs/2601.14888)
- **"Train Where the Quantized Model Goes" (Sep 2026):**
  - Quantization-aware distillation on a fixed corpus recovers short-form QA below 3 bits, but leaves math and code impaired, with repetitive loops.
  - Adding on-policy distillation fixes this: the student generates its own prefixes and gets token-level guidance from the FP teacher plus verifier rewards.
  - At 2.79 and 1.88 effective bits, retention of BF16 accuracy rose from 35% to 70% on MATH-500 and from 66% to 91% on HumanEval.
  - — [arXiv 2609.26708](https://arxiv.org/abs/2609.26708)
- **OnQ (Oct 2026):** on-policy distillation plus a candidate-wise dynamic KL, on Qwen3-1.7B/4B/8B at 2–3 bits. It reports the best 9-benchmark average, with +10.26 points at 2 bits on Qwen3-8B over the strongest baseline. **[S]** (abstract via search snippet) — [arXiv 2610.29378](https://arxiv.org/abs/2610.29378)
- **Apple on-device model (about 3B, 2025).** **[V]**
  - Decoder weights at 2 bits per weight via QAT, embeddings at 4-bit, KV cache at 8-bit.
  - LoRA adapters trained to recover the quality lost to compression.
  - Apple reports about a 4.6% MGSM regression and a 1.5% MMLU improvement for the compressed model.
  - — [Apple ML Research](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)

#### Extreme low-bit (ternary / 1.58-bit)
- **BitNet Distillation ("BitDistill", Oct 2025):**
  - Converts Qwen3 0.6B/1.7B/4B to ternary weights for specific downstream tasks.
  - Recipe: SubLN, MiniLM-style attention distillation, a 10B-token continual-pretraining warm-up, and logit distillation.
  - MNLI accuracy (0.6B / 1.7B / 4B): FP16-SFT 88.01 / 89.61 / 91.48; BitDistill 88.17 / 89.53 / 91.40; naive 1.58-bit SFT 74.09 / 75.27 / 76.11.
  - CPU (16 threads): 427 → 1,135 tok/s (2.65×). Memory: 1.20 → 0.11 GB.
  - Only classification and summarization were evaluated, with no reasoning, math or code tasks.
  - — [arXiv 2510.13998](https://arxiv.org/html/2510.13998v1)
- **MediaTek Dimensity 9500 NPU:** reported to support BitNet-style low-bit models in hardware. **[S]** (I could not confirm which press source states this) — [MediaTek Dimensity 9500](https://www.mediatek.com/dimensity-9500)

#### GGUF / llama.cpp quant types (sizes for this exact model, and phone measurements)
- **File sizes, bartowski imatrix quants of DeepSeek-R1-Distill-Qwen-1.5B** (llama.cpp b4514):

  | Quant | Size |
  |---|---|
  | f16 | 3.56 GB |
  | Q8_0 | 1.89 GB |
  | Q6_K | 1.46 GB |
  | Q5_K_M | 1.29 GB |
  | Q4_K_L (Q8_0 embed/output) | 1.29 GB |
  | Q4_K_M | 1.12 GB |
  | Q4_0 | 1.07 GB |
  | IQ4_XS | 1.02 GB |
  | Q3_K_M | 0.92 GB |
  | IQ3_M | 0.88 GB |
  | Q2_K | 0.75 GB |
  | IQ2_M | 0.70 GB |

  — [HF bartowski GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF)
- **Superseded formats, same page:** the Q4_0_4_4/4_8/8_8 formats are deprecated and cannot run since b4282. Use Q4_0, which is repacked automatically ("online repacking") for ARM/AVX. I-quants are not compatible with Vulkan. — [HF bartowski GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF)
- **Galaxy S25 Ultra (llama.cpp CPU, 8 threads, 0.5B–9B models):** Q4_K_M beat IQ4_XS on both speed and energy. IQ4_XS reduced memory but not energy. **[I]** — [Ehsani et al., arXiv 2603.26603v2](https://www.alphaxiv.org/abs/2603.26603.md)
- **Pixel 8 and iPhone 14 study** (llama.cpp via the PocketPal app; Llama 3.2 1B/3B, Gemma 3 1B/4B, Qwen3 1.7B/4B at Q2_K/Q4_0/Q6_K). **[I]**
  - Composite accuracy covers CommonsenseQA, GSM8K, HumanEval and TruthfulQA.
  - Q2 → Q4 raised it by 100.6%; Q4 → Q6 by only 9.7%.
  - Q4_0 was the energy sweet spot on both phones.
  - — [Guégain & Coignion, arXiv 2609.11940](https://arxiv.org/html/2609.11940v1)
- **NPU caveat:** on Snapdragon NPUs, W4A16 pipelines raised perplexity by 10.1–41.6% relative to CPU-W4 baselines. **[I]** — [Cai et al., arXiv 2607.05475](https://www.alphaxiv.org/abs/2607.05475)

### Inferences
- **Treat Q8_0 (1.89 GB) or W8A8 as the safe default.** If the fine-tuned coder behaves like the base model on LiveCodeBench (16.4 → 13.4–13.9, and 17.5 → 12.7), 4-bit should cost roughly 15–27% of Pass@1 in relative terms. At a Pass@1 of 8–11%, that is about 1–3 absolute points, about the same size as the Stage-2 and M3 gains. So always report Pass@1 at the deployed precision, not just in BF16.
- **If 4-bit is needed,** prefer Q4_K_M with Q8_0 embedding/output (Q4_K_L-style, 1.29 GB), or Q5_K_M/Q6_K. Avoid every 3-bit and W4A4 configuration for the coder. Beyond the accuracy collapse, they make traces longer (LCB 19K → 26–39K tokens), which wipes out any per-token speed gain on a phone.
- **Calibrate on in-domain data.** Build the imatrix/AWQ/GPTQ calibration set from competitive-programming prompts plus the model's own blueprint traces. The +9.8-point gain from math calibration and the code-calibration findings both point this way.
- **For ≤4-bit with ample compute, the 2026 recipe is:** start from a PTQ init, run QAT with KD from the BF16 fine-tuned teacher, then do on-policy distillation on the quantized student's own generations (optionally RL with a verifier, such as unit tests).
  - Hypothesis: short, structured "blueprint" outputs may survive quantization better than 19K-token CoT, because overthinking failures cluster in long traces. This should be tested.
- **Ternary (BitDistill-style) cuts memory about 10×,** but it is unproven for code generation or reasoning. Treat it as a research bet, not a deployment path.

### Gaps
- No study measured quantization of short-trace, structured-reasoning code models. Every reasoning-model quantization number found is for long-CoT R1 distills.
- No paper measured GGUF k-quant/i-quant accuracy with an imatrix (Q4_K_M, IQ4_XS) on reasoning or code benchmarks for 1–2B reasoning models. The GGUF evidence is perplexity, composite or general benchmarks.
- No reasoning-specific results were found for HQQ, OmniQuant or EfficientQAT. SpinQuant only appears in Meta's non-reasoning Llama 3.2 results.
- The full texts of 2601.14888, 2609.26708 and 2610.29378 were not opened, so per-model code (HumanEval/LCB) numbers at each bit-width are unverified.
- The absolute J/token figures per quant level in 2609.11940 appear only in figures.

---

## 2. Pruning (structured and unstructured): what works for reasoning models, and how much distillation recovers

### Takeaway
Pruning is the weakest deployment lever for small reasoning models:
- **50% unstructured sparsity (SparseGPT/Wanda)** costs 3–14 average points on 7–8B R1 distills even with the best calibration data, and 14–35 points with generic C4 calibration.
- **Standard structured pruning** can collapse reasoning at about 20% sparsity unless it is calibrated on reasoning traces.
- **Hard tasks fail first.** AIME collapses earliest, and competitive programming is similarly hard.
- **What does help:** calibrate on the model's own reasoning traces and keep sparsity at or below 20–30%.
- **Quantization is the better first lever.** I found no evidence that mobile runtimes turn unstructured or 2:4 sparsity into speed.

### Cited Findings
- **"Think Before You Prune" (Nov 2025)** tested DeepSeek-R1-Distill-Qwen-7B and Llama-8B at 50% unstructured sparsity, calibrating on 128 sequences of 16,384 tokens.
  - **Qwen-7B dense average 65.2.**
    - SparseGPT: 50.9 with C4 calibration, 55.9 with MATH/GSM8K training data, 60.1 with self-generated traces, 62.0 with *selected* self-generated traces (SSGR).
    - Wanda: 45.4 (C4) → 58.0 (SSGR).
    - AIME: dense 43.8; SparseGPT 21.7 (C4) → 39.2 (SSGR); Wanda with C4 only 13.3.
  - **Llama-8B degrades more:** 64.8 dense → 36.0 (SparseGPT, C4) / 49.6 (SSGR); Wanda 29.8 → 38.2.
  - **The thinking process matters:** removing it from the calibration responses drops the average from 60.1 to 55.7.
  - **Length:** pruned models write longer outputs on complex tasks, and the less degraded the model, the closer its length stays to the dense model.
  - Semi-structured 2:4 and 4:8 results appear only in a figure.
  - — [arXiv 2511.18864](https://arxiv.org/html/2511.18864v1)
- **"When Reasoning Meets Compression" (Apr 2025), SparseGPT on R1 distills:**
  - At 50% sparsity: Llama-70B average 81.8 → 64.2; Qwen-32B 82.2 → 66.2.
  - Llama-70B at 40/60/70/80% sparsity: 76.8 / 53.5 / 21.8 / 8.1.
  - AIME collapses between 40% and 50%. The paper says "pruning only 50% of the weights causes significant degradation, rendering the pruned LRMs unusable".
  - It recommends quantization over pruning for knowledge-heavy tasks, and protecting the final-layer `up_proj`.
  - — [arXiv 2504.02010v2](https://arxiv.org/html/2504.02010v2)
- **RESP (Dec 2025):**
  - Existing pruning methods "severely damage" reasoning LLMs, with "even 20% sparsity collapsing accuracy".
  - Method: self-generated calibration, decode-only gradient importance, and progressive regeneration.
  - Qwen3-8B stays near dense accuracy at 20–30% structured sparsity. At 40%: GSM8K 81.3%, MathQA 59.6%. It reports +66.87% and +47% over the strongest baselines (the abstract does not say whether these are points or relative).
  - — [arXiv 2512.02185](https://arxiv.org/abs/2512.02185)
- **"Doing More With Less" (2026; s1.1-7B, Qwen3-8B):** Wanda at 10–20% unstructured sparsity matched or beat the unpruned models on MATH500, AIME24, AMC23 and GPQA-D, while structured pruning degraded them. **[S]** — [EmergentMind](https://www.emergentmind.com/papers/2604.25098); [Moonlight review](https://www.themoonlight.io/review/doing-more-with-less-revisiting-the-effectiveness-of-llm-pruning-for-test-time-scaling)
- **OBC-Prune (Sep 2026):** compares SparseGPT, Wanda and ALPS at 40–50% sparsity on reasoning models, including AIME 2025, with outcome-based calibration. Its numbers were not verified. **[S]** — [arXiv 2609.17890](https://arxiv.org/pdf/2609.17890)
- **Minitron (structured width/depth pruning plus distillation; non-reasoning).** **[S]**
  - Llama-3.1-Minitron-4B width-pruned: MMLU 60.5, GSM8K 41.2.
  - Depth-pruned: MMLU 58.7, GSM8K 16.8.
  - Depth pruning gave the bigger throughput gain (2.7× vs 1.8×).
  - — [Athina summary](https://hub.athina.ai/research-papers/llm-pruning-and-distillation-in-practice-the-minitron-approach) of [arXiv 2408.11796](https://arxiv.org/html/2408.11796v3)

### Inferences
- **For a 1.5B model at 8–11% Pass@1, pruning will likely cost more accuracy than it saves.** On phones, speed and memory gains come from reading fewer bytes per token. 4-bit quantization already cuts bytes about 3.2× versus f16 (1.12 vs 3.56 GB GGUF) with no retraining, while unstructured sparsity does not speed up standard mobile kernels.
- **If structured pruning is used** (for example, a ~0.5–1B router or draft model cut from the fine-tuned coder):
  - Follow Minitron: prefer width over depth for math/code, and distill from the fine-tuned 1.5B teacher on its own traces.
  - Expect substantial recovery training.
  - Depth pruning gives more speed but hurt GSM8K the most.
- **Vocabulary pruning is an untested but cheap option.** In this model about 26% of parameters are the untied embedding and LM head (151,936 × 1,536 each; see Q5). Trimming tokens never used in Python/English competitive programming could shrink memory, but I found no published study of this.
- **The most consistent advice across the pruning papers:** calibrate on the model's own reasoning traces (here, its blueprint outputs), and keep sparsity modest.

### Gaps
- No pruning results were found for ≤2B reasoning models or for code benchmarks (LiveCodeBench/HumanEval). The studies cover 7–70B models on math and science.
- No quantified "pruning + distillation recovery" numbers exist for reasoning models. The Minitron numbers are for non-reasoning models.
- ShortGPT/layer dropping, LLM-Pruner, SliceGPT and Sheared-LLaMA have no reasoning-model results that I found.
- No source showed llama.cpp, LiteRT-LM, ExecuTorch or MLC speeding up unstructured or 2:4 sparsity on phones. This is absence of evidence, not proof.

---

## 3. Phone runtimes: measured prefill/decode tokens per second and memory for 0.5B–4B models on recent phones

### Takeaway
On 2024–2026 flagships (Snapdragon 8 Gen 3 / 8 Elite / 8 Elite Gen 5, A19 Pro, Dimensity 9500-class), a 1.5–1.7B model at 4–8 bits typically runs at:
- **Decode:** about 23–58 tok/s. The top of that range is vendor best-case; independent measurements are about 23–56.
- **Prefill:** about 100–300 tok/s on CPU, about 900–1,700 tok/s on GPU, and over 1,400 tok/s on a Qualcomm NPU.

Decode is memory-bound, so NPUs mostly help prefill and time-to-first-token (TTFT). Decode slows by about 20–40% as context grows to a few thousand tokens, and further under sustained thermal load.

### Cited Findings

#### Google LiteRT-LM (successor to the MediaPipe LLM Inference API)
- **Vendor table** (prefill / decode tok/s). **[V]**

  | Model (file size) | Device | CPU | GPU |
  |---|---|---|---|
  | Qwen2.5-1.5B (1,598 MB) | Galaxy S25 Ultra | 298 / 34 | 1,668 / 31 |
  | Qwen3-0.6B (586 MB) | Vivo X300 Pro | 165 / 9 | 580 / 21 |
  | Qwen2.5-0.5B (521 MB) | S24 Ultra | 251 / 30 | – |
  | Gemma3-1B (1,005 MB) | S24 Ultra | 177 / 33 | 1,191 / 24 |
  | FunctionGemma (289 MB) | S25 Ultra | 2,238 / 154 | – |
  | Gemma4-E2B (2.58 GB) | S26 Ultra | 557 / 47 (TTFT 1.8 s) | 3,808 / 52 (TTFT 0.3 s) |
  | Gemma4-E2B | iPhone 17 Pro | 532 / 25 | 2,878 / 56 |
  | phi-4-mini | S24 Ultra | – | 314 / 10 |

  - NPU support is listed for Android and Windows.
  - — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)
- **DeepSeek-R1-Distill-Qwen-1.5B (litert-community build, dynamic_int8, 4,096 context, S24 Ultra).** **[V]**
  - CPU (XNNPACK, 4 threads): 166.5 prefill / 26.35 decode tok/s, TTFT 6.41 s, peak RSS 2,221 MB.
  - GPU: 927.5 / 26.98 tok/s, TTFT 5.46 s, GPU memory 1,659 MB.
  - .litertlm file: 1,831 MB.
  - The card also mentions a 1,280 KV-cache size, which it doesn't reconcile with the 4,096 context.
  - The card says the MediaPipe LLM Inference API is in "maintenance-only mode" (**superseded**) and recommends LiteRT-LM (Android, iOS, desktop, web; Kotlin/Swift/C++/Python/JS).
  - — [HF litert-community/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/litert-community/DeepSeek-R1-Distill-Qwen-1.5B)
- **LiteRT-LM v0.19.0 features:** constrained decoding (accelerated logit masking), function calling, and MTP speculative decoding. LoRA is not mentioned. **[V]** — [GitHub google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
- **Single-author test (S25 Ultra, LiteRT-LM 0.11.0-rc1, Gemma 4 E2B).** **[I]**
  - NPU TTFT under 100 ms vs 366–381 ms on GPU; decode about 42 vs about 25 tok/s.
  - But the NPU run generated more tokens overall, so end-to-end latency was not better.
  - — [DEV Community](https://dev.to/jdshah/what-i-learned-benchmarking-the-same-llm-on-my-phones-npu-gpu-and-cpu-4c80)

#### Qualcomm (AI Hub / Genie / GenieX / QNN) and the llama.cpp Hexagon backend
- **Qwen3-1.7B, w4a16, 4,096 context, Qualcomm's own numbers.** **[V]**
  - **Genie runtime:** 58.30 tok/s on Snapdragon 8 Elite Gen 5 (for Galaxy); 56.73 on 8 Elite (for Galaxy).
  - **GenieX-QAIRT on 8 Elite**, by context length:

    | Context | 512 | 1,024 | 2,048 | 4,096 |
    |---|---|---|---|---|
    | tok/s | 39.46 | 41.59 | 36.01 | 29.96 |

    On 8 Elite Gen 5 the same runtime goes from 42.81 (512) to 33.36 (4,096).
  - **GenieX-llama.cpp (q4_0) on 8 Elite Gen 5:** 45.19–49.14 tok/s at 512 context, but 12.91–27.40 at 4,096.
  - The page says "Genie support will be deprecated soon" (**superseded by GenieX**).
  - — [HF qualcomm/Qwen3-1.7B](https://huggingface.co/qualcomm/Qwen3-1.7B)
- **Larger models (Genie, w4a16).** **[V][S]**
  - Qwen3-4B: about 24.4 tok/s on 8 Elite and about 29.2 on 8 Elite Gen 5.
  - Qwen3-8B: about 13.7 tok/s, with TTFT up to about 15 s.
  - — [qualcomm/Qwen3-4B (HF mirror)](https://huggingface-proxy.pages.dev/qualcomm/Qwen3-4B); [AI Hub Qwen3-8B](https://aihub.qualcomm.com/models/qwen3_8b)
- **Independent study (Tsinghua/BJTU, v2 Oct 2026).** **[I]**
  - Setup: four devices, including the OnePlus 15 and Xiaomi 17; Qwen2.5-1.5B and Llama-3.2-1B/3B at 4-bit; over 400 configurations.
  - **Qwen2.5-1.5B on OnePlus 15, prefill:** Genie on NPU 1,463.7 tok/s, more than 15× llama.cpp's 115.1 tok/s.
  - **Decode:** Genie on NPU only 23.0 tok/s, versus 55.7 tok/s for llama.cpp on CPU. The snippet says the llama.cpp 115.1 prefill configuration also decoded at about 23 tok/s.
  - **Context:** NPU decode drops 20.0% (Qwen2.5-1.5B) and 23.8% (Llama-3.2-1B) going from context 16 to 4,096.
  - **Overhead:** NPU sync/dispatch costs 3,449 ms per 256 decoded tokens, versus 71 ms on CPU.
  - **Kernels:** llama.cpp's Hexagon path uses only the HVX vector units, while QNN uses the HMX matrix unit.
  - **Energy:** an energy-oriented Genie configuration cuts SoC energy by about 53% at about +15% latency; CPU frequency tuning cuts decode energy per token by 39.7%.
  - — [arXiv 2607.05475](https://www.alphaxiv.org/abs/2607.05475)
- **Conflict:** Qualcomm reports about 57–58 tok/s decode (Qwen3-1.7B, Genie), while the independent study measured 23 tok/s (Qwen2.5-1.5B, Genie NPU) on similar-generation Snapdragon phones. The models and settings differ, so treat vendor decode numbers as best-case. — [HF qualcomm/Qwen3-1.7B](https://huggingface.co/qualcomm/Qwen3-1.7B) vs [arXiv 2607.05475](https://www.alphaxiv.org/abs/2607.05475)
- **llama.cpp Snapdragon backend docs.** **[V/community]**
  - Backends: CPU, Adreno GPU (OpenCL), and Hexagon NPU (labelled "experimental"; HTP library builds v73/v75/v79/v81).
  - Q4_0 is the only weight type demonstrated on Hexagon.
  - Llama-3.2-1B Q4_0: llama-bench pp128 169.4 tok/s, tg64 51.5 tok/s.
  - Gemma-2-2B Q4_0: prompt 3,200 tok/s (1,024 tokens), eval 47.6 tok/s.
  - The device is not named; the logs show Hexagon arch v79.
  - — [llama.cpp Snapdragon README](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md)

#### Meta ExecuTorch
- **Llama 3.2 on OnePlus 12 (Snapdragon 8 Gen 3; XNNPACK + KleidiAI; 64-token prompt).** **[V]**

  | Model | Decode (tok/s) | Prefill (tok/s) | File (MiB) | RSS (MiB) |
  |---|---|---|---|---|
  | 1B BF16 | 19.2 | 60.3 | 2,358 | 3,185 |
  | 1B SpinQuant | 50.2 | 260.5 | 1,083 | 1,921 |
  | 1B QAT+LoRA | 45.8 | 252.0 | 1,127 | 2,255 |
  | 3B SpinQuant | 19.7 | 89.7 | – | – |
  | 3B QAT+LoRA | 18.5 | 88.8 | 2,529 | – |

  - Llama 3 8B at 4-bit: about 10.9–11.2 tok/s on Galaxy S24 and OnePlus 12.
  - Qwen is not mentioned on this page.
  - — [ExecuTorch Llama README](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md)

#### Apple (MLX, Core ML, Foundation Models framework)
- **MLX on iPhone 17 Pro** (A19 Pro, 12 GB, iOS 26.3 RC; post dated Feb 8, 2026; medium prompt, p50). **[I]** (individual tester)
  - Qwen3-1.7B 4-bit (984 MB): 39.5 tok/s, TTFT 360 ms.
  - Qwen3-0.6B 4-bit: 62.2 tok/s. Llama 3.2 1B 4-bit: 58.1. Gemma 3 1B QAT: 37.1.
  - Qwen3-0.6B decode drops 38% from short to long prompts (69.7 → 43.4).
  - iPad Pro M5 is 1.2–2.2× faster.
  - iOS jetsam kills apps at about 50–67% of RAM, so Llama 3.2 3B and Qwen3 4B were excluded.
  - Mild throttling: about 10% over 10 runs.
  - — [Takkar MLX benchmark (reader copy)](https://hn.jakechampion.name/reader/https://rickytakkar.com/blog_russet_mlx_benchmark.html); original URL [rickytakkar.com](https://rickytakkar.com/blog_russet_mlx_benchmark.html) returned 404 when fetched directly
- **Argmax (Sep 21, 2025):** the iPhone 17 Pro's A19 Pro GPU (which has Neural Accelerators) is up to 3.1× faster than the iPhone 16 Pro GPU on large transformer inference, while the Neural Engine improved only about 1–1.15×. No LLM tok/s figures were given. **[I]** — [Argmax](https://www.argmaxinc.com/blog/iphone-17-on-device-inference-benchmarks)
- **llama.cpp on iPhone 16 Pro (A18 Pro):** 1B models at 40–50 tok/s. **[S]** (vendor blog, methodology unclear) — [Ertas](https://www.ertas.ai/blog/llm-run-on-iphone-benchmarks)
- **Apple Foundation Models framework.** **[V]**
  - The system model is about 3B parameters with 2-bit QAT.
  - Guided generation (`@Generable`) uses constrained and speculative decoding run by an OS daemon. Tool calling is supported.
  - Apple says it is "not designed to be a chatbot for general world knowledge".
  - — [Apple ML Research](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)
- **Older phones (1–4B models, llama.cpp via PocketPal):** TTFT 0.24–1.13 s on iPhone 14 (A15) and 0.48–1.41 s on Pixel 8 (Tensor G3). The iPhone 14 used on average 33.3% less energy per token than the Pixel 8. **[I]** — [arXiv 2609.11940](https://arxiv.org/html/2609.11940v1)

#### Alibaba MNN
- **MNN-LLM** (Xiaomi 14, Snapdragon 8 Gen 3; Qwen2-1.5B, Qwen2-7B, Llama3-8B). **[V]**
  - **CPU:** prefill 8.6× and decode 2.3× faster than llama.cpp.
  - **GPU (OpenCL):** prefill up to 25.3× faster than llama.cpp and 2.8× faster than MLC-LLM; decode up to 7.1× and 1.7×.
  - **Quantization:** W4A8/W8A8 on CPU, W4A16 on GPU. Embeddings are stored in flash in bf16. KV keys are int4/int8 and values fp8.
  - **LoRA:** supported both merged and online.
  - Absolute numbers appear only in a figure.
  - — [arXiv 2506.10443](https://arxiv.org/html/2506.10443v1)
- **MNN-AECS:** 23% less energy than plain MNN with no slowdown. Versus llama.cpp, ExecuTorch, mllm and MediaPipe, it reports 39–78% energy savings and 12–363% speedups across 7 devices. **[V]** — [arXiv 2506.19884](https://arxiv.org/abs/2506.19884)

#### MLC-LLM, PowerInfer-2, MediaTek, Google Tensor, and general findings
- **MLC-LLM and llama.cpp measured with Llama-2-7B 4-bit (2024–25).** **[I]**
  - MLC uses only about 5–20% of mobile GPU arithmetic units, and loads about 5–7× slower than llama.cpp.
  - Longer contexts slow prefill and decode by about 10–20%.
  - 20 continuous rounds dropped Xiaomi 14 Pro (8 Gen 3) decode from 12.7 to 8.9 tok/s (about −30%).
  - Snapdragon decode improved by about 80–110% per generation.
  - — [arXiv 2410.03613v2](https://arxiv.org/html/2410.03613v2)
- **PowerInfer-2 (2024):**
  - Serves a 47B model on a smartphone at 11.68 tok/s, "up to 27.8×" faster than state-of-the-art frameworks.
  - Method: neuron clusters (dense ones on NPU, sparse ones on CPU) with I/O pipelining.
  - It targets models larger than RAM, so it is **not needed for a 1.5B model**.
  - **[V]** — [arXiv 2406.06282](https://arxiv.org/abs/2406.06282)
- **MediaTek Dimensity 9500:** claims 2× faster 3B LLM output than the 9400, an NPU 990 rated up to 100 TOPS, and 128K-token processing, all from internal testing. No absolute tok/s were found. **[V]** — [MediaTek](https://www.mediatek.com/dimensity-9500); [HotHardware](https://hothardware.com/news/mediatek-unveils-dimensity-9500-big-ai-gaming-uplifts)
- **Google Tensor G5 (Pixel 10):** Google claims Gemini Nano runs 2.6× faster and 2× more efficiently than on the previous generation. No llama.cpp or LiteRT tok/s were found. **[V]** — [Google blog](https://blog.google/intl/en-in/products/hardware/5-reasons-why-google-tensor-g5-is-a-game-changer-for-pixel/)
- **Galaxy S25 Ultra study (llama.cpp CPU):** only models up to about 3B were practical for interactive use, and autoregressive generation dominates both latency and energy. **[I]** — [arXiv 2603.26603](https://www.alphaxiv.org/abs/2603.26603.md)

### Inferences
- **For the 1.5B coder,** llama.cpp on CPU (Q4_0 with ARM repacking, or Q4_K_M) and LiteRT-LM on GPU are the most portable options. Plan on about 30–40 tok/s sustained decode on 8 Gen 3 / 8 Elite / A19 Pro-class phones; 50–58 tok/s is the burst or vendor best case.
- **NPU paths (Genie/QNN, LiteRT NPU) mainly shorten TTFT** for long prompts such as retrieved blueprints plus the problem statement. For long-output reasoning, independent data shows CPU or GPU decode matching or beating the NPU. NPU W4A16 pipelines may also cost extra accuracy (perplexity +10–42%).
- **Memory fits on iOS.** A Q4_K_M 1.5B model (1.12 GB) plus KV cache fits comfortably under iOS jetsam limits. 3–4B models may not fit in third-party apps on 8–12 GB iPhones.
- **MediaTek data point (inference):** the Vivo X300 Pro entry (Qwen3-0.6B, CPU decode only 9 tok/s) is, I believe, a Dimensity 9500 phone. If so, LiteRT-LM CPU support there looks less mature than on Snapdragon.
- **Chip identities (inference):** I believe Hexagon v79 corresponds to the Snapdragon 8 Elite, and that the OnePlus 15 and Xiaomi 17 use the 8 Elite Gen 5. The sources don't state either.

### Gaps
- No independent tok/s measurements were found for Tensor G4/G5, Dimensity 9400/9500 with open runtimes, or rigorous A17 Pro/A18 Pro numbers.
- No fresh (2025–2026) MLC-LLM phone benchmarks were found, and its maintenance status is unverified.
- No rigorous llama.cpp measurement of DeepSeek-R1-Distill-Qwen-1.5B GGUF on phones was found. Community sites give estimates such as about 31 tok/s on a Galaxy S25, without methodology.
- The Qualcomm AI Hub page for Qwen2.5-1.5B-Instruct returned 404 when fetched, so its numbers are unknown.

---

## 4. Decoding acceleration on device, and how reasoning length turns into seconds and battery

### Takeaway
- **Long reasoning is very expensive on a phone.** The base R1-distill-1.5B averages about 19K CoT tokens per LiveCodeBench problem. At 26–57 tok/s that is about 6–12 minutes of pure decoding per problem, likely 7–15 minutes with long-context slowdown and throttling. It also costs roughly 3–5% of a phone battery (estimate).
- **A 500–1,000-token "blueprint + code" answer takes about 10–40 s.** Cutting tokens is therefore a 10–30× lever.
- **Decoding tricks are worth only about 1.1–3.8× on phones.** Retrieval/n-gram drafting suits phones and code better than a separate draft model.
- **8-bit KV cache is safe.** 4-bit KV needs outlier-aware methods at this model size.

### Cited Findings

#### Reasoning length and on-device latency
- **Average CoT length of R1-Distill-Qwen-1.5B (BF16):** LiveCodeBench 19,066 tokens, AIME 17,465, GPQA 11,887, MATH-500 5,156. With 3-bit AWQ, LCB rises to 38,793. — [arXiv 2606.00206](https://arxiv.org/html/2606.00206v1)
- **Qualcomm AI Research, "Efficient Reasoning on the Edge" (Mar/Jun 2026).** **[V, research]**
  - They note that CoT traces "can exceed 4,000 tokens" and that token-by-token reasoning can take "several minutes" on mobile processors.
  - **Approach:**
    - LoRA reasoning adapters (rank 128, alpha 256; Qwen2.5-7B; trained on OpenThoughts3).
    - RL "budget forcing" to shorten responses.
    - Dynamic adapter switching that "activates reasoning only when needed".
    - Parallel test-time scaling.
    - KV-cache sharing during prompt encoding to cut TTFT.
    - Tooling: Qualcomm FastForward for quantization and Genie for inference.
  - **Accuracy, W4A16 LoRA-128:** AIME25 0.37, MATH500 0.90, GPQA 0.38, AMC23 0.80. The FP LoRA scored 0.38 / 0.93 / 0.43 / 0.82.
  - — [arXiv 2603.16867](https://arxiv.org/abs/2603.16867); [project page](https://qualcomm-ai-research.github.io/llm-reasoning-on-edge)
  - A press headline says the reasoning chains were shrunk "2–4x" (or 2.4×; unverified). **[S]** — [The Decoder](https://the-decoder.com/qualcomm-shrinks-ai-reasoning-chains-by-2-4x-to-fit-thinking-models-on-smartphones/)
- **Longer responses are more often wrong.** Within each R1 model, the shortest 30% of responses scored higher than the longest 30% (R1 on AIME: 88.9 vs 33.3), and the length ratio between the groups was usually above 4. — [arXiv 2504.02010v2](https://arxiv.org/html/2504.02010v2)
- **Decode slows as context grows:**
  - NPU: −20% for Qwen2.5-1.5B from context 16 to 4,096 — [arXiv 2607.05475](https://www.alphaxiv.org/abs/2607.05475)
  - GenieX-llama.cpp q4_0 on 8 Elite Gen 5: 45–49 → 13–27 tok/s from 512 to 4,096 context — [HF qualcomm/Qwen3-1.7B](https://huggingface.co/qualcomm/Qwen3-1.7B)
  - MLX on iPhone 17 Pro: −38% from short to long prompt — [Takkar](https://hn.jakechampion.name/reader/https://rickytakkar.com/blog_russet_mlx_benchmark.html)
  - Sustained use: −30% after 20 continuous rounds on 8 Gen 3 — [arXiv 2410.03613v2](https://arxiv.org/html/2410.03613v2)

#### Energy and battery
- **Measured battery drain per round** (Llama-2-7B 4-bit; 64-token prompt plus 128-token generation). **[I]**
  - Vivo Pad3 Pro (Dimensity 9300): 4.54 mAh per round at 8.22 tok/s decode.
  - Huawei MatePad 12.6 Pro (Kirin 9000E): 8.28 mAh per round.
  - The MatePad's temperature rose from 42.6 to 66.8 °C during one inference.
  - — [arXiv 2410.03613v2](https://arxiv.org/html/2410.03613v2)
- **Phones vs servers.** **[I]**
  - On-device inference (Pixel 8, iPhone 14) is about 3× less energy-efficient per token than batched server inference (A100 with vLLM).
  - 8 of 18 model/quantization configurations sit on the accuracy-vs-energy Pareto front, which the authors say "could support battery-aware model routing".
  - Battery capacities: Pixel 8 4.575 Ah at 3.89 V; iPhone 14 3.279 Ah at 3.87 V.
  - — [arXiv 2609.11940](https://arxiv.org/html/2609.11940v1)
- **Galaxy S25 Ultra (llama.cpp):** Qwen2-0.5B used about 50 J per bounded summarization run (at most 100 output tokens); Gemma-2-9B used over 300 J. Generation dominates energy. These per-run totals include overheads, so they are not per-token decode costs. **[I]** — [arXiv 2603.26603](https://www.alphaxiv.org/abs/2603.26603.md)
- **Energy scales with output length:** total energy grows almost linearly with output length and decode latency, and prompt wording that deepens reasoning raises both tokens and energy. **[S]** (search snippet and abstract) — [arXiv 2607.22568](https://arxiv.org/abs/2607.22568)
- **Decode dominates energy:** the memory-bound decode phase accounts for most of the energy. **[V]** — [MNN-AECS, arXiv 2506.19884](https://arxiv.org/abs/2506.19884)
- **Power draw (vendor estimate):** about 2–3 W while generating with a 1B model and 3–5 W with a 3B model; continuous 3B generation drains about 1% of battery per minute. **[S]** (vendor blog, methodology not given) — [Ertas](https://www.ertas.ai/blog/llm-android-benchmarks-snapdragon-tensor)

#### Speculative decoding and early exit on phones
- **sd.npu (2025–26).**
  - Setup: Redmi K60 Pro (8 Gen 2), K70 Pro (8 Gen 3) and OnePlus 13 (8 Elite); Qwen2.5-0.5B/1.5B and Llama3.2-3B.
  - Method: retrieval-based drafting with NPU-coordinated verification.
  - **Overall:** 1.06–3.81× per-token speedups and 1.07–4.71× energy reductions.
  - **Qwen2.5-1.5B (K60 Pro, summarization):** 57.41 → 15.08 ms/token, about 3.8×.
  - **Speculative decoding barely helps on CPU:** 0.92–1.13× with retrieval drafting, and 0.56–1.18× with an EAGLE-2 draft model.
  - **Draft models cost memory:** model-based drafters add 0.84–1.22 GB of peak memory.
  - **Acceptance length, Qwen2.5-1.5B:** prompt-lookup decoding (PLD) 2.53 tokens (2.91 with their improvements); SAM 3.60 → 4.21.
  - Code generation was not tested.
  - — [arXiv 2510.15312v4](https://arxiv.org/html/2510.15312v4)
- **llama.cpp speculative types:**
  - `draft-simple` (separate draft model).
  - `draft-eagle3` (one-layer draft that reads the target's hidden states; trained per target).
  - `draft-dflash`, `draft-mtp`, `ngram-cache`.
  - `ngram-simple`, suggested for "rewriting source code".
  - `ngram-map-k` and `ngram-map-k4v`.
  - `ngram-mod`, suggested for "iterating over text or code" and for "reasoning models repeating their thinking in the answer".
  - Example acceptance rates shown are 0.58 and 0.70. No speedups are claimed.
  - — [llama.cpp speculative docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
  - Server flags are `--spec-type` and `--spec-draft-n-max` (default 3). The old `--draft-max`/`--draft-min` flags were removed (**superseded**). — [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- **EAGLE-3:** up to 6.5× speedup, about 1.4× over EAGLE-2, and 1.38× throughput in SGLang at batch 64. These are server-GPU results. — [arXiv 2503.01840](https://arxiv.org/abs/2503.01840)
- **LayerSkip (early exit plus self-speculative decoding):** up to 2.16× on summarization, 1.82× on coding and 2.0× on semantic parsing. It needs training with layer dropout and an early-exit loss. — [arXiv 2404.16710](https://arxiv.org/abs/2404.16710)
- **Apple** uses speculative decoding inside the Foundation Models framework, and the adapter toolkit ships a draft model. **[V]** — [Apple ML Research](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates); [Apple adapter toolkit](https://developer.apple.com/apple-intelligence/foundation-models-adapter/)

#### KV-cache quantization and compression
- **Accuracy on the 1.5B (KV8 / KV4 / KV3):**
  - KV8 (with W8A8) is lossless.
  - KV4 with KVQuant* costs −1.4 average (LCB 16.4 → 15.9), but KV4 with QuaRot collapses.
  - KV3 costs −12.3.
  - — [arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2)
- **llama.cpp KV types:** f16 by default; options include q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, bf16 and f8_e4m3, set with `-ctk`/`-ctv`. — [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- **R-KV (NeurIPS 2025, reasoning models):**
  - Keeping only 10–34% of the KV cache reaches parity with the full cache.
  - The peak result is 105% of full-cache performance, but the versions disagree on the cache budget: 16% in the arXiv version vs 38% in the NeurIPS version.
  - — [arXiv 2505.24133](https://arxiv.org/html/2505.24133v4); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/57e0358aed314784cad14a26e1eba642-Abstract-Conference.html)
- **Apple** shares KV cache across two model blocks (5:3 depth split), cutting KV memory by 37.5%. **[V]** — [Apple ML Research](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)

### Inferences (worked arithmetic; label as estimates)
- **Latency per long-CoT problem:**
  - 19,066 tokens ÷ 26.5 tok/s (LiteRT int8 build of this exact model, S24 Ultra) ≈ 12.0 min.
  - ÷ 39.5 tok/s (MLX, Qwen3-1.7B 4-bit, iPhone 17 Pro) ≈ 8.0 min.
  - ÷ 57 tok/s (Genie vendor best case) ≈ 5.6 min.
  - Add 20–40% for context-length slowdown and throttling: realistically about 7–15 min per problem.
  - A 3-bit model's 38.8K-token traces would take over 20 minutes and approach context limits.
  - A 500–1,000-token blueprint + code answer takes about 10–40 s.
- **KV-cache memory.** From the config (Q5), per token = 28 layers × 2 KV heads × 128 head-dim × 2 (K and V) × 2 bytes = 28,672 bytes in FP16.

  | Context | FP16 KV cache |
  |---|---|
  | 4K tokens | ≈ 117 MB |
  | 19K tokens | ≈ 547 MB |
  | 32K tokens | ≈ 940 MB |

  q8_0 roughly halves these. Long traces can therefore come close to doubling the 1.12 GB Q4_K_M weight footprint.
- **Battery, order of magnitude:**
  - The Dimensity 9300 measurement (4.54 mAh × 3.87 V ≈ 63 J for 192 tokens with a 7B 4-bit model) implies about 0.3–0.5 J/token for 7B.
  - Scaling by weight bytes (about 3.5–3.8 GB down to about 1.1 GB) gives about 0.1–0.15 J/token for a 1.5B 4-bit model.
  - A 19K-token trace would then use about 2–3 kJ (0.5–0.8 Wh), about 3–5% of a Pixel 8 battery (4.575 Ah × 3.89 V ≈ 17.8 Wh).
  - Cross-check: the vendor estimate of 2–3 W × 12 min gives about 0.4–0.6 Wh.
  - A 600-token answer costs about 0.1–0.2% of the battery.
- **Ranking of levers for this project:**
  1. Train for short blueprints and budget-force reasoning length. This is a 10–30× token reduction.
  2. Add a training-free logit penalty on overthinking tokens (12–23% shorter, and it helps more when quantized).
  3. Use n-gram/prompt-lookup drafting. It needs no extra RAM, and the final code often re-copies pseudocode from the trace, which is the case llama.cpp's `ngram-mod` targets.
  4. Use EAGLE-3 or MTP only on GPU/NPU paths, and only if the extra RAM is acceptable. On a phone CPU, speculative decoding may give no speedup at all.
- **The router idea is consistent with the literature.** It matches Qualcomm's reasoning switcher ("activates reasoning only when needed") and the battery-aware routing suggestion in arXiv 2609.11940. A router is cheap: FunctionGemma (289 MB) decodes at 154 tok/s and prefills at 2,238 tok/s on S25 Ultra CPU. A MiniLM-size classifier or embedder (tens of MB, my estimate) would add well under 1 s, while letting the coder skip thousands of tokens.

### Gaps
- No text-reported J/token figures for 1–2B models on Snapdragon or A-series phones (2609.11940 has them only in figures; 2410.03613 gives mAh per round for 7B).
- No on-device study measured end-to-end latency or battery for a code model doing thousands of tokens of CoT. The minute-scale and battery numbers above are computed.
- On-phone speculative decoding has not been measured for code generation (sd.npu excludes code).
- The reduction factor in Qualcomm's "Efficient Reasoning on the Edge" (2.4× vs "2–4×") and its on-device timings were not verified.

---

## 5. Practical pipeline: LoRA-fine-tuned Qwen2 1.5B to Android/iOS (formats, tools, file sizes, LoRA hot-swap)

### Takeaway
- **Lowest-friction path:**
  1. Merge the LoRA into the base model.
  2. Convert to GGUF (llama.cpp) with an in-domain importance matrix (imatrix).
  3. Quantize to Q4_K_M (1.12 GB), Q4_K_L-style (1.29 GB), Q6_K (1.46 GB) or Q8_0 (1.89 GB).
  4. Run on Android with llama.cpp (CPU / OpenCL / Hexagon) and on iOS with llama.cpp (Metal) or MLX.
- **Vendor-optimized alternatives:** LiteRT-LM (.litertlm; the int8 build of this base is 1.83 GB), Qualcomm AI Hub/GenieX (w4a16) and ExecuTorch.
- **Runtime LoRA hot-swap:**
  - Mature in llama.cpp (several GGUF adapters, per-request scales) and MNN (online LoRA).
  - Possible in Qualcomm Genie/QNN, but adapters must be built with the same SDK and quantization-compatible.
  - In ExecuTorch, done by sharing base weights across one program per adapter.
  - Apple's adapters only work with Apple's own model.

### Cited Findings
- **Model architecture (DeepSeek-R1-Distill-Qwen-1.5B):**
  - Qwen2ForCausalLM; hidden size 1,536; 28 layers; 12 attention heads and 2 KV heads; intermediate size 8,960.
  - Vocabulary 151,936; `tie_word_embeddings: false`; max position embeddings 131,072; bf16; sliding window off.
  - — [HF config.json](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/main/config.json)
- **GGUF sizes:** see the table in Q1. All are imatrix quants, and the "_L"/"_XL" variants keep embedding and output weights at Q8_0. — [HF bartowski](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF)
- **LiteRT-LM build of this base:**
  - File: `DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm`, 1,831 MB.
  - Run with the CLI `litert-lm run --from-huggingface-repo=...`; in Android apps, use the Kotlin dependency `com.google.ai.edge.litertlm:litertlm-android`.
  - The model card does not document the conversion steps.
  - — [HF litert-community](https://huggingface.co/litert-community/DeepSeek-R1-Distill-Qwen-1.5B)
- **Qualcomm AI Hub:** the AI Hub Models Python library can export models with custom weights, input shapes and target runtime. Pre-exported w4a16 Genie bundles exist, and Genie is being deprecated in favor of GenieX. **[V]** — [HF qualcomm/Qwen3-1.7B](https://huggingface.co/qualcomm/Qwen3-1.7B)
  - Qwen2.5-1.5B-Instruct appears in the AI Hub catalog (search listing; the page returned 404 when fetched). — [AI Hub](https://aihub.qualcomm.com/models/qwen2_5_1_5b_instruct)
- **ExecuTorch quantization options:** groupwise 4-bit PTQ, SpinQuant, QAT+LoRA, and experimental torchao low-bit (1–8 bit) kernels for arm64. — [ExecuTorch Llama README](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md)
- **llama.cpp LoRA at runtime:**
  - Load adapters with `--lora a.gguf,b.gguf` or `--lora-scaled FNAME:SCALE`.
  - `--lora-init-without-apply` loads them without applying, so they can be applied later through `POST /lora-adapters`.
  - A request can carry its own `"lora": [{"id": 0, "scale": 0.5}]` list; adapters not listed default to scale 0.
  - Caveat: "requests with different LoRA configurations will not be batched together".
  - — [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- **MNN-LLM:** supports LoRA merged and online. Its optimized LoRA computation order uses about 0.5% of the original memory access (Qwen2-7B, r=8). — [arXiv 2506.10443](https://arxiv.org/html/2506.10443v1)
- **ExecuTorch "program-data separation":**
  - Several LoRA programs (.pte) can share one foundation-weights file (.ptd).
  - There is a Qwen3-0.6B LoRA demo, and quantization with program-data separation was merged in Oct 2025.
  - The design RFC notes that loaded external data is not necessarily cached.
  - I found no documented API for hot-swapping several adapters on one loaded base.
  - — [ExecuTorch export docs](https://docs.pytorch.org/executorch/main/using-executorch-export.html); [PR #14864](https://github.com/pytorch/executorch/pull/14864); [RFC issue #8118](https://github.com/pytorch/executorch/issues/8118)
- **Qualcomm QNN/Genie LoRA:**
  - LoRA v2: the Genie Dialog API can apply an adapter and set its alpha at runtime between executions (via `QnnContext_applyBinarySection`). The adapter binary must come from the same SDK version that built the base context binary.
  - LoRA v3: adds adapter "groups". Mismatched adapter weights and quantization parameters cause execution to fail.
  - These docs are on a pages.dev mirror of the QAIRT docs.
  - — [QAIRT LoRA v2](https://qairt-2-44-docs.pages.dev/QNN/general/tutorial/lora_v2_0_overview); [QAIRT LoRA v3](https://qairt-2-44-docs.pages.dev/QNN/general/tutorial/lora_v3_optimizations)
- **Separately trained LoRA adapters have incompatible quantization** (each has its own scales and offsets), which blocks runtime switching under one compiled edge model. The paper proposes unified adaptive distillation as the fix. **[S]** — [arXiv 2603.29535](https://arxiv.org/html/2603.29535v1)
- **Multi-LoRA serving of one LLaMA-based model on the NPU** of Galaxy S24/S25 phones (SM8650/SM8750) has been described. **[S]** — [arXiv 2604.18655](https://arxiv.org/html/2604.18655v1)
- **Qualcomm's chat ↔ reasoning adapter switch reuses the same KV cache,** so the prompt need not be re-encoded. **[S]** — [arXiv 2603.16867](https://arxiv.org/pdf/2603.16867)
- **Apple Foundation Models adapters.** **[V]**
  - About 160 MB each.
  - Each is compatible with one specific system-model version, so a new adapter must be trained for every version.
  - Deployment needs the Foundation Models Framework Adapter Entitlement, and adapters are delivered through Background Assets.
  - Toolkit 26.0.0 is "the last release of this toolkit and is not compatible with macOS, iOS, iPadOS, or visionOS 27 and later" (**superseded or discontinued for OS 27+**).
  - Only Apple's system model can be adapted.
  - — [Apple Developer: adapter toolkit](https://developer.apple.com/apple-intelligence/foundation-models-adapter/)
- **LiteRT-LM:** its README does not mention LoRA. — [GitHub LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)

### Inferences
- **Parameter count:** "1.5B" is really about 1.78B parameters. The f16 GGUF is 3.56 GB, which is about 1.78B × 2 bytes. The untied embedding and LM head (151,936 × 1,536 ≈ 233M each, about 467M together, about 26% of the model) explain the gap.
  - Keeping those two at Q8 costs only about +0.17 GB (Q4_K_L 1.29 GB vs Q4_K_M 1.12 GB).
  - MNN-style flash-resident embeddings could save about 0.47 GB of DRAM in bf16.
- **Recommended GGUF pipeline** (standard tooling; exact commands not re-verified this session):
  1. Use PEFT `merge_and_unload()` to merge the LoRA into the BF16 base.
  2. Run llama.cpp `convert_hf_to_gguf.py` to get an F16 GGUF.
  3. Run `llama-imatrix` on in-domain prompts plus the model's own blueprint traces.
  4. Run `llama-quantize` to Q4_K_M, with Q8_0 output and token-embedding overrides. Also produce a Q4_0 build for ARM repacking and the Hexagon backend (Q4_0 is the only type demonstrated on Hexagon).
  5. Measure Pass@1 at each quant level with identical decoding settings.
  6. Ship with llama.cpp: Android via JNI (CPU / OpenCL / Hexagon); iOS via Metal. Alternatively, on iOS use MLX (4-bit conversion).
- **RAM budget:** Q4_K_M weights (1.12 GB) + q8_0 KV for 4–10K tokens (about 60–150 MB) + runtime overhead ≈ 1.4–1.8 GB. A MiniLM-class router or embedder adds tens of MB.
- **Per-topic adapters on llama.cpp:**
  - Keep one base model in memory, preload N small GGUF LoRA adapters with `--lora-init-without-apply`, and pick one per request with the scale field.
  - My estimate is that rank-16–64 adapters for a 1.5B model are tens of MB each.
  - The cost is the unapplied LoRA matmuls during decode, which I could not quantify.
- **Per-topic adapters on NPU (Genie):** adapters need joint quantization with the base and must be built with the same SDK.
- **A simpler design than adapters:** the router picks retrieved blueprint text for the prompt (like the current M3 retrieval) instead of swapping weights. This needs no adapter infrastructure, and a fixed system prompt can be prefix-cached.
- **Apple Foundation Models is not a target for a custom Qwen2 model.** Use llama.cpp, MLX or Core ML on iOS instead.

### Gaps
- No measured decode overhead of unmerged LoRA adapters in llama.cpp on phones.
- MLC-LLM and LiteRT-LM runtime LoRA support could not be confirmed. MLC's current status is also unverified.
- No documented Hugging Face → .litertlm conversion recipe for a custom Qwen2 checkpoint was captured. The LiteRT-LM README only shows prebuilt downloads.
- The performance numbers for Qwen2.5-1.5B on the Qualcomm AI Hub page are unknown (404).

---

## 6. Can a phone run a sandboxed Python interpreter to execute and test generated code?

### Takeaway
Yes, with platform-specific limits:
- **Android** can embed CPython (official builds, Chaquopy, Briefcase, Buildozer). Subprocesses are "possible but officially unsupported", so a separate watchdog process is feasible.
- **iOS forbids all subprocess, multiprocessing and IPC.** Generated code must run in-process (embedded CPython) or inside WebAssembly (Pyodide in a WebView or worker), which can be torn down.
- **WebAssembly Python has no working subprocess, threads, sockets or `resource` limits.**
- **App Store guideline 2.5.2 restricts executing code that changes an app's functionality.** It has an educational exception that requires the code to be fully viewable and editable by the user.

### Cited Findings
- **CPython on Android:**
  - Embedded mode only: no system install, no `python` executable, no console or REPL.
  - The app packages `libpython*.so` and the stdlib in its assets.
  - Recommended tools: Briefcase (BeeWare), Buildozer (Kivy), Chaquopy, Flet, pyqtdeploy, Termux. Wheels can be cross-built with cibuildwheel.
  - — [Python docs: Using Python on Android](https://docs.python.org/3/using/android.html)
- **Mobile platforms (Python stdlib docs):**
  - "Mobile platforms can only use Python in 'embedded' mode. There is no Python REPL, and no ability to use separate executables such as python or pip."
  - Android: "creating subprocesses is possible but officially unsupported". There is no System V IPC, so `multiprocessing` is unavailable.
  - iOS: "An iOS app cannot use any form of subprocessing, multiprocessing, or inter-process communication. If an iOS app attempts to create a subprocess, the process creating the subprocess will either lock up, or crash."
  - "Mobile apps have no usable stdin at all"; `curses` and `readline` are unavailable.
  - — [Python docs: library intro, Mobile platforms](https://docs.python.org/3/library/intro.html)
- **WebAssembly (Emscripten/WASI):**
  - `subprocess` is "importable but does not work". `fork`, `execve`, `waitpid` and `kill` are unavailable or always fail.
  - Sockets are limited (always non-blocking on Emscripten), and blocking I/O is not permitted.
  - — [Python docs: library intro, WebAssembly](https://docs.python.org/3/library/intro.html)
- **Pyodide (docs page shows version 314.0.8):**
  - `multiprocessing`, `threading` and sockets are included but non-functional.
  - Removed modules include `resource`, `termios`, `curses`, `tkinter`, `venv`, `dbm`, `fcntl`, `pwd` and `grp`. `pty`/`tty` are unimportable.
  - `ssl` is a stub, and HTTP goes through the browser.
  - — [Pyodide WASM constraints](https://pyodide.org/en/stable/usage/wasm-constraints.html)
- **Apple App Review Guideline 2.5.2** (verbatim excerpt):

  > Apps ... may not download, install, or execute code which introduces or changes features or functionality of the app, including other apps. Educational apps designed to teach, develop, or allow students to test executable code may, in limited circumstances, download code provided that such code is not used for other purposes. Such apps must make the source code provided by the app completely viewable and editable by the user.

  - Guideline 4.7 allows HTML5/JavaScript mini apps.
  - Guideline 2.5.6 requires web browsing to use WebKit.
  - — [Apple App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)

### Inferences
- **The tests themselves need no subprocess.** Competitive-programming tests are stdin/stdout, so running `exec` with `sys.stdin`/`sys.stdout` redirected to `io.StringIO` works on both OSes.
- **The hard part is stopping bad code:**
  - In-process CPython can't be hard-killed on an infinite loop. Use `sys.settrace` or line-count budgets for cooperative timeouts.
  - On Android, consider running tests in a separate app process or service with a watchdog. "Possible but unsupported" subprocesses, or Android's isolated-process services, are the options, but I did not verify the latter in a source.
  - On iOS, Pyodide inside a Web Worker in a WKWebView can be terminated from JavaScript, which gives a real kill switch at some speed cost.
- **Memory blowups can kill the whole app.** Pyodide has no `resource` module (no rlimits), and iOS jetsam terminates apps at about 50–67% of RAM (Q3). Use small test inputs and per-test time limits, and size limits based on the device rather than on a judge's limits.
- **Official CPython mobile support:** I believe iOS and Android became officially supported targets in Python 3.13 (PEP 730/738). This is from background knowledge, not re-verified this session.
- **App Store risk.** Code produced on-device by the app's own model arguably isn't "downloaded". But executing it could fall under 2.5.2's "changes features or functionality". Framing the app as educational, where the user can view and edit the code before it runs, lines up with the explicit exception. I found no comparable restriction for Android.

### Gaps
- No measurements of Pyodide vs native CPython speed on phones, and no Chaquopy performance data, were found.
- No authoritative statement was found on whether App Review accepts on-device execution of model-generated Python.
- No source was fetched describing Android isolated-process sandboxing for embedded Python, or per-test CPU and memory limits on Android.
