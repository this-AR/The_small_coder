# On-device / mobile-first small LM families (≤4B) and the best phone-deployable base for a competitive-programming coder (as of 2026-11-19)

Tag legend: **[SR]** = the model's developer reports the number. **[XL]** = another lab (often a competitor) reports it in its own comparison table. That makes it independent of the developer, but not neutral. **[3P]** = a neutral third party (an independent evaluator or a journalist's hands-on test). **[2nd]** = a secondary outlet relaying a vendor claim; the primary source was not fetched. **[CALC]** = my own arithmetic from published configs. LiveCodeBench (LCB) v5, v6 and other dated windows use different problem sets, so scores are not comparable across versions. Most reported small-model LCB scores also assume very long generation budgets (32K–82K tokens; see Section 6). The ranked shortlist is in the final section (Section 6).

## 1. Gemma 3 270M, Gemma 3n (E2B/E4B) and newer Gemma small releases (2026): architecture, training, fine-tuning use, on-device numbers

### Takeaway
Gemma 3n (June 2025) introduced Google's phone techniques: MatFormer nesting, Per-Layer Embeddings (PLE) kept on the CPU, and KV-cache sharing. Gemma 4 E2B/E4B (April 2, 2026, now under Apache 2.0) keep PLE and KV sharing and add 128K context and a thinking mode. Coding scores rose from LCB v5 18.6/25.7 (Gemma 3n) to LCB v6 44.0/52.0 [SR]. Gemma 4 E2B is Google's most phone-ready option: 0.8 GB mobile-quantized, +0.05 GB int8 KV cache at 32K, about 47–52 tok/s decode on a 2026 flagship phone [SR]. Gemma 3 270M is a base for fine-tuned routing or classification, not a coder (LCB v6 0.0 [XL]).

### Cited Findings
**Gemma 3 270M (blog dated Aug 14, 2025)**
- 270M parameters: 170M are embedding parameters (256k vocabulary) and 100M are transformer-block parameters. INT4 QAT checkpoints are provided, and both pretrained and instruction-tuned versions were released. — [Google Developers Blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
- Intended use is task-specific fine-tuning for "well-defined, high-volume tasks": sentiment analysis, entity extraction, query routing and compliance checks. It is "not designed for complex conversational use cases." Recommended tools: Hugging Face, Unsloth, JAX. — [Google Developers Blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
- Battery [SR, internal test]: on a Pixel 9 Pro SoC, the INT4 model "used just 0.75% of the battery for 25 conversations." — [Google Developers Blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
- Trained on 6T tokens (1B: 2T, 4B: 4T, 12B: 12T, 27B: 14T). Context is 32K (the 4B and larger get 128K). Knowledge cutoff is Aug 2024 and the license is the Gemma Terms of Use. IT scores [SR]: IFEval 51.2, BBH 26.7, HellaSwag 37.7. — [HF model card](https://huggingface.co/google/gemma-3-270m)
- Coding is essentially absent [XL, Meta's evaluation]: Gemma-3-270M-pt HumanEval 3.1, MBPP 2.0; Gemma-3-270m-it LCB v6 0.0. — [MobileLLM-R1 card](https://huggingface.co/facebook/MobileLLM-R1-950M)

**Gemma 3 attention (baseline for the 270M and 1B models)**
- The model interleaves 5 local layers per global layer, with a 1024-token sliding window. Global layers use RoPE base 1M and local layers 10k. At 32K context, "global only" attention costs 60% KV-memory overhead, versus "<15%" with local/global layers and 1024-token windows. The report's ablation text attaches that figure to a 1:3 ratio, while secondary sources attribute it to 5:1 (conflict). — [Gemma 3 Technical Report](https://arxiv.org/pdf/2503.19786)
- Gemma-3-1B-IT [XL]: HumanEval 41.5, MBPP 35.2 — [MobileLLM-Pro card](https://huggingface.co/facebook/MobileLLM-Pro); non-thinking LCB v5 1.8 — [Qwen3 report](https://arxiv.org/html/2505.09388v1)

**Gemma 3n (preview May 2025; full release June 26, 2025)**
- E2B/E4B have 5B/8B raw parameters but run with the memory footprint of 2B/4B models; minimum memory is 2 GB (E2B) and 3 GB (E4B). — [Gemma 3n developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/). Conflict: the HF E2B card says "raw parameters: 6B" while its size box shows 5B. — [HF card](https://huggingface.co/google/gemma-3n-E2B-it)
- MatFormer: the E2B sub-model is optimized inside E4B during training, and a pre-extracted E2B is "up to 2x faster" than E4B. Mix-n-Match builds custom sizes by changing the per-layer FFN hidden dimension (8192→16384) and skipping layers; MatFormer Lab finds the configurations. Elastic runtime switching is listed as future work. — [developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- PLE: the per-layer embeddings are loaded and computed on the CPU, so only the core transformer weights (about 2B for E2B, 4B for E4B) sit in accelerator memory. — [developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- KV-cache sharing: the keys and values of the middle layer's local and global attention are shared with the top layers, giving a "2x improvement" in prefill over Gemma 3 4B. AltUp and LAuReL are named as components (with arXiv links) but not described. — [developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- Encoders: the audio encoder is USM-based (one token per 160 ms). The vision encoder, MobileNet-V5-300M, runs at up to 60 fps on a Pixel and is 13× faster when quantized than Gemma 3's SoViT, with 46% fewer parameters and a 4× smaller memory footprint. E4B scores LMArena >1300, the first sub-10B model to do so [SR]. — [developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- About 11T training tokens (web text in 140+ languages, code, math, images, audio). Knowledge cutoff June 2024, 32K context, Gemma license. Code scores [SR], E2B / E4B IT: LCB v5 18.6 / 25.7; a second "LiveCodeBench" row shows 13.2 / 13.2; HumanEval 66.5 / 75.0; MBPP 56.6 / 63.6; Codegolf v2.2 11.0 / 16.8; AIME 2025 6.7 / 11.6. — [HF card](https://huggingface.co/google/gemma-3n-E2B-it)
- Tooling: fine-tuning with Transformers and TRL, NVIDIA NeMo, Unsloth, Axolotl; inference with llama.cpp, Ollama, MLX, LiteRT-LM, LM Studio, transformers.js, vLLM, SGLang. — [developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)

**Gemma 4 (2026): the newer small releases**
- Launched April 2, 2026 with E2B, E4B, a 26B MoE and a 31B dense model, under Apache 2.0. Gemma 3 and 3n used the Gemma Terms of Use. — [Google Developers Blog](https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/); [Wikipedia](https://en.wikipedia.org/wiki/Gemma_(language_model)) (one secondary guide says March 31).
- E2B has 2.3B effective parameters (5.1B with embeddings); E4B has 4.5B (8B). They have 35 / 42 layers, with 512-token sliding-window layers interleaved with global layers; the final layer is always global. Context is 128K and the vocabulary 262K. Encoders: ~150M vision and ~300M audio. Data cutoff is Jan 2025. Both pretrained and instruction-tuned variants were released (the repo `google/gemma-4-E2B` is presumably the pretrained one). — [HF E2B-it card](https://huggingface.co/google/gemma-4-E2B-it); [HF E2B card](https://huggingface.co/google/gemma-4-E2B)
- Technical report (arXiv 2607.02770, July 2026):
  - PLE is used "as in Gemma 3n."
  - E2B uses a 4:1 local:global ratio; the other models use 5:1.
  - KV-cache sharing covers 20 of 35 layers (E2B) and 18 of 42 (E4B).
  - Global layers use p-RoPE with p=0.25.
  - Training ran on 4,096 TPU v6e chips (E2B) and 6,144 (E4B).
  - Post-training is "similar to Gemma 3" plus a thinking mode.
  - Pretraining token counts and any distillation are not stated.
  — [Gemma 4 Technical Report](https://huggingface.co/papers/2607.02770)
- Memory [SR] (32K context, text only):
  - E2B: 4.6 GB in bf16, 0.8 GB "mobile quantized" (per-channel int2/int4 weights, int8 activations), plus 0.05 GB for an int8 KV cache.
  - E4B: 9.0 GB, 2.3 GB and +0.14 GB respectively.
  — [Gemma 4 Technical Report](https://huggingface.co/papers/2607.02770)
- Code [SR, thinking mode], E2B / E4B: LCB v6 44.0 / 52.0; Codeforces Elo 633 / 940; SciCode 21.0 / 24.0; AIME 2026 (no tools) 37.5 / 42.5. For comparison, Gemma 3 27B (non-thinking) scores LCB v6 29.1 and Elo 110. — [Gemma 4 Technical Report](https://huggingface.co/papers/2607.02770). Other labs disagree [XL]: OpenBMB measured E2B at 42.9 and E4B at 53.9 — [MiniCPM5-2B card](https://huggingface.co/openbmb/MiniCPM5-2B); Liquid lists Gemma 4 E4B at 71.00 — [LFM3-1.7B card](https://huggingface.co/LiquidAI/LFM3-1.7B).
- Thinking is switched on by putting `<|think|>` in the system prompt. E2B/E4B produce no empty thought block when it is off. — [HF card](https://huggingface.co/google/gemma-4-E2B-it)
- On-device [SR]:
  - E2B fits in under 1.5 GB on some devices (2/4-bit weights plus memory-mapped PLE).
  - Raspberry Pi 5 CPU: 133 prefill / 7.6 decode tok/s. Qualcomm Dragonwing IQ8 NPU: 3,700 / 31 tok/s.
  — [Gemma 4 dev blog](https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/)
  - Samsung S26 Ultra (LiteRT-LM, 1024 prefill / 256 decode tokens): CPU 557 prefill / about 47 decode tok/s; GPU 3,808 / 52. iPhone 17 Pro decode: 25 tok/s (CPU), 56.5 (GPU). Conflict: another version of the page lists the S26 Ultra at 195/18 (CPU) and 1,293/22 (GPU). — [LiteRT-LM Gemma 4 page](https://developers.google.com/edge/litert-lm/models/gemma-4)
  - Multi-token-prediction speculative decoding speeds decode up to 2.2× on mobile GPUs and 1.5× on mobile CPUs. — [LiteRT-LM blog](https://developers.googleblog.com/blazing-fast-on-device-genai-with-litert-lm/)
- Runtimes: LiteRT-LM (constrained decoding, 2/4-bit weights), the Android AICore Developer Preview (Gemma 4 built in), the AI Edge Gallery app (Android and iOS), WebGPU and Raspberry Pi. — [Gemma 4 dev blog](https://developers.googleblog.com/bring-state-of-the-art-agentic-skills-to-the-edge-with-gemma-4/)
- Later in 2026: Gemma 4 12B "Unified", encoder-free and multimodal, released June 3, 2026, runs in about 16 GB (LCB v6 72.0, Elo 1659 [SR]) — [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/); [HF card](https://huggingface.co/google/gemma-4-E2B-it). QAT variants arrived around June 6, 2026 [2nd]. — [WinBuzzer](https://winbuzzer.com/2026/06/06/google-releases-smaller-gemma-4-models-for-local-ai-xcxwbn/)
- Fine-tuning: Unsloth supports Gemma 4 vision, text, audio and RL fine-tuning. Its GRPO notebook uses E2B, and E2B RL fits in about 9 GB. vLLM fast inference was not supported for Gemma 4 RL (set `fast_inference=False`). — [Unsloth Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train); [Unsloth FR guide](https://unsloth.ai/docs/fr/modeles/gemma-4/train)

### Inferences
- For a coder, Gemma 4 E2B/E4B supersede Gemma 3n: higher LCB, a 4× longer context, Apache 2.0, and a far smaller KV cache. Gemma 3n remains relevant only as the origin of the MatFormer, PLE and KV-sharing ideas.
- Gemma 4 E2B's KV cost is tiny: about 1.5 KB/token in int8 [CALC from 0.05 GB / 32K], versus 28 KiB/token in bf16 for the current DeepSeek-R1-Distill-Qwen-1.5B (Section 4). That makes it the friendliest Google option for long outputs. The cost: about 5.1B parameters stored on flash (PLE memory-mapped), plus multimodal encoders a text-only coder does not need.
- Gemma 3 270M suits the proposed "decision/router" model (topic, difficulty or budget prediction), since Google explicitly targets query routing and classification. It cannot generate code.

### Gaps
- No fetched source states Gemma 3 270M / 3n / 4 distillation details (teacher, logits) or Gemma 4 pretraining token counts.
- AltUp and LAuReL mechanics are not described in the fetched Google sources.
- No neutral third-party LCB for Gemma 4 E2B; no battery numbers for Gemma 3n or Gemma 4.
- The Gemma 4 QAT release details come only from a secondary source. The S26 Ultra speed figures conflict between page versions.

## 2. Meta's MobileLLM line: design principles, MobileLLM-R1, MobileLLM-Pro and successors

### Takeaway
MobileLLM (2024) set the sub-1B design rules: deep and thin, embedding sharing, GQA, and block-wise weight sharing. MobileLLM-R1-950M shows a fully open recipe (~4.2T tokens plus 6.2M-sample reasoning SFT, no RL) reaching LCB v6 19.9, tied with DeepSeek-R1-Distill-Qwen-1.5B in Meta's evaluation, at 949M parameters. MobileLLM-Pro (1.08B) shows the phone engineering (3:1 local-global attention, 128K context, int4 QAT, 590 MB) but is not a reasoning coder. All are FAIR non-commercial, and no 2026 successor was found.

### Cited Findings
**MobileLLM (arXiv Feb 2024; ICML 2024)**
- At sub-billion scale, depth matters more than width. Combining a deep-and-thin layout, embedding sharing and GQA, the 125M/350M models beat the previous best models of the same size by 2.7%/4.3%. — [arXiv 2402.14905](https://arxiv.org/abs/2402.14905); [PMLR](https://proceedings.mlr.press/v235/liu24ce.html)
- Immediate block-wise weight sharing (MobileLLM-LS) adds another 0.7%/0.8%. Sharing weights between adjacent blocks "avoids weight movement," so the block is computed twice at little extra latency. This targets memory-bound devices. — [arXiv 2402.14905](https://arxiv.org/abs/2402.14905)

**MobileLLM-R1 (weights Sept 12, 2025; code Sept 28; paper Sept 29; ICLR 2026)**
- Sizes: 140M, 360M, 950M, each with a base and a final (post-trained) checkpoint; Meta says they are "not general-purpose chat models." The 950M has 22 layers, 24 heads, 6 KV heads, dim 1536, FFN 6144 and 949M parameters. Vocabulary is 128k with shared embeddings. Context is 4K (base) and 32K (final). License: FAIR Noncommercial Research License v1. — [HF card](https://huggingface.co/facebook/MobileLLM-R1-950M)
- Pretraining: two phases of 2T tokens each (sequence length 2k), "~2T high-quality tokens," "<5T total training tokens."
  - Phase 1 mix: FineWeb-Edu 63.75%, StarCoder 10.66%, OpenWebMath 6.93%, ArXiv 6.36%, Wiki 5.03%, StackExchange 5.03%, Algebraic Stack 2.25%.
  - Phase 2 mix: FineWeb-Edu 54.83%, OpenWebMath 23.33%, FineMath 8.01%, Nemotron math 3.01%, natural reasoning 3.18%, Cosmopedia 2.70%.
  — [HF card](https://huggingface.co/facebook/MobileLLM-R1-950M)
- Mid-training: two phases of 100B tokens (sequence length 4k), with knowledge distillation from Llama-3.1-8B-Instruct, on Dolmino, Nemotron-Pretraining-Code-v1, Nemotron-CC-Math-v1 and StarCoder. — [HF card](https://huggingface.co/facebook/MobileLLM-R1-950M)
- Post-training: a general SFT of 866K samples (Tulu-3 mixture, 2 epochs), then a reasoning SFT of 6.2M samples at 32k sequence length (4 epochs). The reasoning set is OpenMathReasoning 3.2M, OpenScienceReasoning-2 803K and OpenCodeReasoning-2 2.16M. No RL stage is listed. — [HF card](https://huggingface.co/facebook/MobileLLM-R1-950M)
- The paper claims it used 11.7% of Qwen3's 36T tokens yet matches or beats Qwen3-0.6B. — [arXiv 2509.24945](https://arxiv.org/abs/2509.24945); [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c62fb98c25c2318bf3aca24d3b0fc85e-Abstract-Conference.html)
- Results [SR; Meta ran the baselines, so those are XL]:

  | Model (post-trained) | MATH500 | GSM8K | AIME'24 | AIME'25 | LCB v6 (n=16) |
  |---|---|---|---|---|---|
  | MobileLLM-R1-950M | 74.0 | 67.5 | 15.5 | 16.3 | 19.9 |
  | Qwen3-0.6B | 73.0 | 79.2 | 11.3 | 17.0 | 14.9 |
  | DeepSeek-R1-Distill-Qwen-1.5B | 83.2 | 77.3 | 29.1 | 23.4 | 19.9 |

  Base 950M: HumanEval 46.3, MBPP 39.2 (Qwen3-0.6B-base: 30.5 / 39.0). — [HF card](https://huggingface.co/facebook/MobileLLM-R1-950M)

**MobileLLM-Pro (Oct 2025; report arXiv 2511.06719, Nov 10, 2025)**
- 1.08B parameters: 30 layers, 20 heads, 4 KV heads, dim 1280, FFN 6144. Vocabulary 202,048 with shared embeddings. Local-global attention at 3:1 with a 512-token window; 128K context. — [HF card](https://huggingface.co/facebook/MobileLLM-Pro)
- Pretrained on "<2T fully open-source tokens" (a ~1,640B datamix), with logit-based KL distillation from Llama 4-Scout. Base, instruct and int4 checkpoints are released. License: FAIR NC, gated. — [HF card](https://huggingface.co/facebook/MobileLLM-Pro)
- Quantization [SR]: int4 QAT for CPU (group size 32, int8 dynamic activations, int8 KV) loses 0.4%; int4 per-channel for accelerators loses 1.3%. QAT loses 1.5% where post-training quantization loses 34%. The int4 CPU model is 590 MB. — [HF card](https://huggingface.co/facebook/MobileLLM-Pro)
- At 8K context, versus all-global attention: prefill is 1.8× faster and the KV cache shrinks from 117 MB to 40 MB. — [HF card](https://huggingface.co/facebook/MobileLLM-Pro)
- Phone speed [SR], at 2k / 4k / 8k context:

  | Device | Prefill (s) | Decode (tok/s) |
  |---|---|---|
  | Galaxy S25 CPU | 8.9 / 24.8 / 63.5 | 33.6 / 24.8 / 19.7 |
  | Galaxy S24 HTP (NPU) | 1.96 / 3.38 / 9.82 | 31.60 / 28.95 / 22.77 |

  KV cache: 14 / 23 / 40 MB. — [HF card](https://huggingface.co/facebook/MobileLLM-Pro)
- IFT scores [SR]: HumanEval 59.8, MBPP 46.8, IFEval 62.0, versus Gemma 3 1B at 41.5 / 35.2 / 80.2 and Llama 3.2 1B at 37.8 / 39.6 / 59.5. — [HF card](https://huggingface.co/facebook/MobileLLM-Pro)

### Inferences
- MobileLLM-R1 reached its LCB with SFT alone, using about 2.16M code-reasoning samples (6.2M reasoning samples in total). The team's Stage 1 used 12k long-CoT samples, two to three orders of magnitude fewer. That is consistent with their finding that a 1.5B model trained on a little verbose CoT learns the format but not the skill.
- MobileLLM-R1's fully published data mix makes it the most reproducible "open data" template. Its non-commercial license is fine for an academic paper but blocks shipping an app.
- MobileLLM-Pro's numbers show that, even with local-global attention, phone-CPU decode falls about 41% from 2K to 8K context. Long traces cost twice: more tokens, and slower tokens.

### Gaps
- No MobileLLM-branded release in 2026 was found (latest: R1 in Sep 2025 and Pro in Oct/Nov 2025).
- No LCB for MobileLLM-Pro; no RL-trained MobileLLM variant.

## 3. Other families: LFM2/2.5/3, SmolLM2/3, Qwen3/3.5, Granite 4.0 Nano, Phi-4-mini(-flash), BitNet b1.58 2B4T, Falcon-H1/H1R/H1-Tiny, Hymba/Nemotron-Flash, and 2026 sub-2B coders

### Takeaway
By late 2026 the sub-2B coding frontier has moved from Qwen-based distills (DeepSeek-R1-Distill-Qwen-1.5B: LCB 16.9 [SR]) to heavily RL-trained, on-policy-distilled 2026 models. The leaders are LFM3-1.7B (LCB v6 73.5 [SR]) and MiniCPM5-2B (69.1 [SR], 70.5 [XL]). Next come VibeThinker-1.5B (51.1 [SR]) and Gemma 4 E2B (44.0 [SR]). Qwen3.5-2B is weak at code (20.2 [XL]). LFM2/2.5, Granite Nano, BitNet, the small Falcon-H1 models and Hymba are efficient but not competitive competitive-programming coders.

### Cited Findings
**Liquid AI: LFM2 → LFM2.5 → LFM3**
- LFM2 (tech report arXiv 2511.23404): the 1.2B model has 16 layers, 10 double-gated short-range convolution blocks plus 6 GQA blocks. Liquid claims up to 2× faster CPU prefill and decode than similar-size models [SR]. A hardware-in-the-loop search converged on this "minimal hybrid." — [lilting.ch](https://lilting.ch/en/articles/lfm-hybrid-architecture); [featherless model mirror](https://featherless.ai/models/unsloth/LFM2-1.2B); [alphaXiv overview](https://www.alphaxiv.org/overview/2511.23404)
- LFM2-8B-A1B: 8.3B total and 1.5B active parameters, 32 experts with top-4 routing. Liquid calls it "significantly faster" than Qwen3-1.7B [SR]. A secondary report says decoding is up to 5× faster on a Ryzen AI 9 HX370 and a Galaxy S24 Ultra with int4 [2nd]. It is deprecated in favour of LFM2.5-8B-A1B. — [Liquid blog](https://www.liquid.ai/blog/lfm2-8b-a1b-an-efficient-on-device-mixture-of-experts); [AIbase](https://www.aibase.com/news/21859)
- LFM2.5 (Jan 5, 2026): 1.2B Base, Instruct and JP models, plus VL-1.6B and Audio-1.5B. Pretraining grew from 10T to 28T tokens; post-training adds SFT, preference tuning and "large-scale multi-stage RL." Instruct scores [SR]: GPQA 38.89, MMLU-Pro 44.35, IFEval 86.23, AIME25 14.00, BFCLv3 49.12 (Qwen3-1.7B in instruct mode: 34.85 / 42.91 / 73.68 / 9.33 / 46.30). No coding benchmarks are reported. — [Liquid blog](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- LFM2.5 on-device [SR] (1K prefill / 100 decode tokens):

  | Device and runtime | Model | Prefill (tok/s) | Decode (tok/s) | Memory |
  |---|---|---|---|---|
  | Galaxy S25 Ultra CPU, llama.cpp Q4_0 | LFM2.5-1.2B | 335 | 70 | 719 MB |
  | Galaxy S25 Ultra CPU, llama.cpp Q4_0 | Qwen3-1.7B | 181 | 40 | 1,306 MB |
  | ROG Phone 9 Pro NPU (NexaML) | LFM2.5-1.2B | 4,391 | 82 | 0.9 GB |

  — [Liquid blog](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- LFM2.5-1.2B-Thinking runs in under 900 MB. Liquid recommends it for agentic tasks, data extraction and RAG, not for knowledge-heavy tasks or programming. — [Liquid blog](https://www.liquid.ai/en/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb); [The Batch](https://www.deeplearning.ai/the-batch/liquid-ais-small-reasoning-model-mixes-attention-with-convolutional-layers-for-efficiency)
- Other 2026 LFM2.5 releases [2nd]: LFM2.5-8B-A1B (May 28), LFM2.5-VL-3B (Aug 12), LFM2.5-24B-A2B (mid-October). — [lavx](https://news.lavx.hu/article/lfm2-5-8b-a1b-a-more-capable-on-device-mixture-of-experts-model); [emergent.sh](https://emergent.sh/news/lfm-25-vl-3b-officially-launched); [tech-insider](https://tech-insider.org/liquid-ai-lfm2-5-24b-a2b-launch-2026/)
- LFM3 (Oct 27, 2026; updated Nov 5). All models branch from LFM3-1.7B-Base.
  - Architecture: 24 layers, made of 13 gated short convolutions (kernel size raised from 3 to 7), 5 sliding-window attention layers (256-token window) and 6 global attention layers. Also head-wise sigmoid gating, partial RoPE on global layers, sandwich norm and a 128K vocabulary. It was found by hardware-in-the-loop search over 14 device/backend combinations.
  - Pretraining: 22T tokens; the last 3.5T are mid-training that extends context to 256K.
  - Post-training: 1.75T tokens of SFT → mixed RL with a length penalty → multi-domain on-policy distillation from six specialist RL teachers → agentic RL.
  — [Liquid LFM3 blog](https://www.liquid.ai/blog/lfm3.md)
- LFM3-1.7B scores [SR]: LCB v6 73.46, LCB Pro 25Q2 31.74, SciCode 28.47, SWE-bench Verified (OpenCode harness) 43.40, IFEval 96.66, BFCL V4 71.39. In the same table, Qwen3.5-9B scores LCB v6 69.86, LCB Pro 34.43, SciCode 31.60, SWE-bench Verified 51.20. — [HF card](https://huggingface.co/LiquidAI/LFM3-1.7B); [blog](https://www.liquid.ai/blog/lfm3.md)
- LFM3 efficiency [SR]: about 1 GB at 4-bit. The KV cache at 128K tokens is 1.6 GB, versus 4.3 GB for Qwen3.5-4B and 5.6 GB for MiniCPM5-2B. Decode is 1.8–2.8× faster than the next-fastest model on three devices Liquid does not name (llama.cpp Q4_0, 32K context), and about 25% faster than LFM2-1.2B at 8K. — [blog](https://www.liquid.ai/blog/lfm3.md)
- LFM3 decision models (d1-2B, d1-Omni-2B) do routing, triage, checking and classification. They answer every question in a single forward pass with calibrated probabilities (yes/no, choices, scores, pick, estimate, read, point) instead of generating text. — [blog](https://www.liquid.ai/blog/lfm3.md)
- The LFM3-1.7B card notes:
  - Reasoning is on by default; `enable_thinking=False` "can be inconsistent."
  - Recommended for local agents, tool calling, structured outputs and domain fine-tuning. Not recommended for agentic coding or "complex multi-step reasoning."
  - Fine-tuning: TRL notebooks for SFT, DPO, GRPO and continued pretraining.
  - Runtimes: vLLM, SGLang, Transformers, GGUF (llama.cpp, Ollama), MLX, ONNX, OpenVINO.
  — [HF card](https://huggingface.co/LiquidAI/LFM3-1.7B)
- License: LFM Open License v1.0. Commercial use is free only below $10M annual revenue; the threshold does not apply to qualified non-profits using the models for non-commercial or research purposes, and it does apply to derivative models. — [LFM license](https://www.liquid.ai/en/lfm-license); [Liquid docs](https://docs.liquid.ai/lfm/help/model-license)

**Hugging Face SmolLM2 / SmolLM3**
- SmolLM3 (July 8, 2025) is a 3B model.
  - Architecture: GQA with 4 groups, NoPE (no RoPE) in every 4th layer, intra-document masking, tied embeddings. Context: trained to 64K, extended to 128K with YaRN.
  - Pretraining: about 11.2T tokens in 3 stages, with code rising from 12% to 15% to 24% and math from 3% to 10% to 13%.
  - Post-training: reasoning mid-training (35B tokens × 4 epochs), then SFT on 1.8B tokens, then APO (a DPO variant), then model merging.
  - Dual-mode `/think` / `/no_think`. Base and instruct models, configs and data (SmolTalk2) are released.
  - LCB [SR]: 30.0 with thinking vs 15.2 without.
  — [HF blog](https://huggingface.co/blog/smollm3)
- SmolLM2-1.7B [XL, Meta's evaluation]: base HumanEval 0.6, MBPP 35.4; Instruct LCB v6 4.4, MATH500 19.2. — [MobileLLM-R1 card](https://huggingface.co/facebook/MobileLLM-R1-950M)

**Qwen3 (tech report May 2025) and Qwen3.5 small models (2026)**
- Qwen3: pretrained on 36T tokens; the 0.6B and 1.7B have 32K context. The small models were built by strong-to-weak distillation: off-policy distillation from teacher outputs in /think and /no_think modes, then on-policy logit KL against Qwen3-32B or 235B-A22B. Distillation took about 1/10 the GPU-hours of RL (for the 8B: 1,800 vs 17,920 GPU-h). — [Qwen3 report](https://arxiv.org/html/2505.09388v1)
- Qwen3 code scores [SR]:
  - Thinking mode, LCB v5: Qwen3-1.7B 33.2, Qwen3-0.6B 12.3, DeepSeek-R1-Distill-Qwen-1.5B 13.2 (Qwen's evaluation). AIME'24: 48.3 / 10.7 / 28.9.
  - Non-thinking LCB v5: Qwen3-1.7B 11.6, Qwen3-0.6B 3.6, Phi-4-mini 10.4.
  - Base models: EvalPlus 52.70 (Qwen3-1.7B-Base), 44.80 (Qwen2.5-1.5B), 36.23 (Qwen3-0.6B).
  — [Qwen3 report](https://arxiv.org/html/2505.09388v1)
- Qwen3-1.7B config: 28 layers, 16 query / 8 KV heads, head_dim 128, vocabulary 151,936, tied embeddings, 40,960 max positions. — [config.json](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json)
- Qwen3.5 small models are dense 0.8B, 2B, 4B and 9B, dated March 2, 2026 by one timeline [2nd]; the model-card citation says February 2026. — [timeline](https://try-qwen-ai.com/updates/release-history/); [HF Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- Qwen3.5-2B card:
  - Architecture: 24 layers, laid out as 6 × (3 Gated DeltaNet + 1 Gated Attention). Gated Attention has 8 query / 2 KV heads with head dim 256. Embedding 248,320 (tied). Trained with multi-token prediction. Native context 262,144. Includes a vision encoder.
  - Apache 2.0, with a separate Qwen3.5-2B-Base.
  - Non-thinking by default; in thinking mode the 2B is "more prone to thinking loops."
  - Recommends 81,920 output tokens for hard math and programming benchmarks.
  — [HF card](https://huggingface.co/Qwen/Qwen3.5-2B)
- Qwen3.5-2B has no published coding scores [SR]. Thinking-mode scores: MMLU-Pro 66.5, GPQA 51.6, HMMT Feb 25 22.9 (Qwen3-1.7B: 56.5 / 40.1 / 10.2) — [HF card](https://huggingface.co/Qwen/Qwen3.5-2B). OpenBMB's internal run [XL]: LCB v6 20.2, OJBench 2.6, AIME 2025 29.6 — [MiniCPM5-2B card](https://huggingface.co/openbmb/MiniCPM5-2B).
- Qwen3.5-4B: 32 layers, 8 × (3 DeltaNet + 1 attention), 4 KV heads, head dim 256; 262K native context, extensible to 1.01M with YaRN. Scores [SR]: LCB v6 55.8, OJBench 24.1; the 9B scores 65.6 / 29.2 — [HF Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Other labs [XL]: OpenBMB measured the 4B at 56.4 ([MiniCPM5 card](https://huggingface.co/openbmb/MiniCPM5-2B)); Liquid lists the 9B at 69.86 ([LFM3 card](https://huggingface.co/LiquidAI/LFM3-1.7B)).
- Independent [3P]: on the Artificial Analysis Intelligence Index, Qwen3.5-9B scores 32, 4B 27, 2B 16 and 0.8B 9. — [Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models) (from the search snippet; the page fetch failed with an SSL error)
- Later Qwen [2nd]: Qwen3.6's open weights are a 35B-A3B (Apr 16, 2026) and a 27B dense (Apr 22, 2026); Qwen3.8 is a 27B (Aug 2026). No small Qwen3.6 or 3.8 model was found. — [Miraheze wiki](https://ai.miraheze.org/wiki/Qwen_3.6); [Qwen3.8 GitHub](https://github.com/QwenLM/Qwen3.8)

**IBM Granite 4.0 Nano (Oct 28, 2025)**
- Granite-4.0-H-1B (about 1.5B, hybrid Mamba-2/transformer) and H-350M, plus plain-transformer 1B and 350M versions "for workloads where hybrid architectures may not yet have optimized support." Apache 2.0. — [itsfoss](https://itsfoss.com/news/ibm-granite-4-nano/); [HF blog](https://huggingface.co/blog/ibm-granite/granite-4-nano)
- HumanEval pass@1 [SR] (350M / H-350M / 1B / H-1B): instruct 39 / 38 / 74 / 73; base 32 / 34 / 60 / 59. IFEval for H-1B is 78.5 vs Qwen3-1.7B at 73.1. — [HF base card](https://huggingface.co/ibm-granite/granite-4.0-h-1b-base); [itsfoss](https://itsfoss.com/news/ibm-granite-4-nano/)
- IBM claims Granite 4.0-H needs over 70% less RAM for long inputs and concurrent batches than conventional transformers [SR]. Secondary coverage reports a ~9:1 Mamba-2:attention ratio. — [IBM announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)

**Microsoft Phi-4-mini / Phi-4-mini-flash-reasoning / BitNet b1.58 2B4T**
- Phi-4-mini-instruct: 3.8B, 200K vocabulary, 128K context, MIT license; HumanEval 74.4, MBPP 65.3 [SR, relayed]. — [NVIDIA NIM card](https://docs.api.nvidia.com/nim/reference/microsoft-phi-4-mini-instruct); [DataLearner](https://www.datalearner.com/en/ai-models/pretrained-models/Phi-4-mini-instruct)
- Phi-4-mini-flash-reasoning (SambaY paper, arXiv 2507.06607, July 2025): 3.8B, 64K context, 200K vocabulary.
  - SambaY is a decoder-hybrid-decoder. Its self-decoder combines Mamba, sliding-window attention and a single full-attention layer. Its cross-decoder interleaves cross-attention with Gated Memory Units, which multiply element-wise with a memory state from an earlier layer.
  - Microsoft claims up to 10× higher throughput and a 2–3× average latency reduction [SR]. That was measured with 2K prompts and 32K generation on one A100-80GB with vLLM, not on phones.
  — [Azure blog](https://azure.microsoft.com/en-us/blog/reasoning-reimagined-introducing-phi-4-mini-flash-reasoning/); [The Decoder](https://the-decoder.com/microsoft-introduces-phi-4-mini-flash-reasoning-with-up-to-10x-higher-token-throughput/); [SambaY paper](https://arxiv.org/abs/2507.06607v3); [card mirror](https://cnb.cool/ai-models/microsoft/Phi-4-mini-flash-reasoning)
- BitNet b1.58 2B4T (arXiv 2504.12285, Apr 2025): a native ternary 2B model trained on 4T tokens. Numbers [SR]:

  | Metric | BitNet 2B4T | Gemma 3 1B | Llama 3.2 1B | Qwen2.5-1.5B |
  |---|---|---|---|---|
  | Non-embedding memory | 0.4 GB | 1.4 GB | – | 2.6 GB |
  | CPU decode latency | 29 ms/token | 41 ms | 48 ms | 65 ms |
  | Energy per token (estimate) | 0.028 J | 0.186 J | – | 0.347 J |
  | HumanEval+ | 38.40 | 37.20 | 31.10 | 50.60 |

  The efficiency gains appear only with bitnet.cpp, not with standard transformers. — [BitNet report](https://arxiv.org/pdf/2504.12285); [Tom's Hardware](https://tomshardware.com/tech-industry/artificial-intelligence/microsoft-researchers-build-1-bit-ai-llm-with-2b-parameters-model-small-enough-to-run-on-some-cpus); [wavect](https://wavect.io/blog/bitnet-cpu-llm-ram-requirements/)

**TII Falcon-H1 / Falcon-H1R / Falcon-H1-Tiny**
- Falcon-H1 (launched May 2025 [2nd]; report arXiv 2507.22448) is a parallel hybrid: attention heads and Mamba-2 heads sit in the same block. The 0.5B has 36 layers; the 1.5B-Deep has 66 layers (deeper rather than wider). — [Falcon-H1 report](https://arxiv.org/html/2507.22448v1); [press release](https://www.silicon.co.uk/press-release/middle-easts-leading-ai-powerhouse-tii-launches-two-new-ai-models-falcon-arabic-the-first-arabic-model-in-the-falcon-series-falcon-h1-a-best-in-class-high-performance-model)
- Falcon-H1-0.5B-Base [SR]: HumanEval 35.98, MBPP 52.12, versus Qwen3-0.6B at 31.71 / 51.06 — [HF card](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base). A LiveCodeBench value of about 17.61 appears for the 1.5B-Instruct, but the column attribution was unclear in the snippet — [HF](https://huggingface.co/tiiuae/Falcon-H1-1.5B-Base).
- Falcon-H1R 7B (Jan 5–6, 2026): a reasoning model on the Falcon-H1-7B base. TII reports AIME-24 88.1% and up to 1,500 tok/s per GPU at batch 64, about 2× Qwen3-8B [SR]. Falcon TII License. — [Middle East AI News](https://www.middleeastainews.com/p/abu-dhabis-tii-releases-compact-reasoning); [Noqta](https://noqta.tn/en/news/uaes-tii-releases-falcon-h1r-7b-a-small-ai-model-that-outperforms-giants)
- Falcon-H1-Tiny (Jan 15, 2026): 15 models, most ≤100M, plus the R-0.6B reasoning model. Variants include Coder-90M and a tool-calling model. TII tests "anti-curriculum" training (pretraining directly on instruction, chat or reasoning data) on a hybrid Mamba+Transformer design. — [Middle East AI News](https://www.middleeastainews.com/p/tii-releases-15-tiny-falcon-language); [HF collection](https://huggingface.co/collections/tiiuae/falcon-h1-tiny)

**NVIDIA Hymba / Nemotron-Flash**
- Hymba (arXiv 2411.13676; ICLR 2025) puts attention and SSM heads in parallel inside each layer, prepends 128 learnable meta tokens, and uses cross-layer KV sharing plus a mix of global and local attention. Trained on 1.5T tokens. Against Llama-3.2-3B, Hymba-1.5B-Base reports +1.32% average accuracy, an 11.67× smaller cache and 3.49× higher throughput [SR]. — [arXiv](https://arxiv.org/abs/2411.13676); [NVIDIA Research](https://research.nvidia.com/publication/2025-04_hymba-hybrid-head-architecture-small-language-models)
- Nemotron-Flash (NeurIPS 2025) chooses its mix of attention, Mamba2, DeltaNet and FFN by evolutionary search. The 1B and 3B models keep only 2 and 3 full-attention layers. Results [SR]: the 1B scores 5.5% higher than Qwen3-0.6B at 1.9× lower latency; the 3B has 1.3–1.7× lower latency than Qwen3-1.7B and Qwen2.5-3B. Deep-and-thin wins accuracy per parameter but is not automatically on the latency frontier. — [arXiv 2511.18890](https://arxiv.org/pdf/2511.18890)

**2026 sub-2B (or ~2B) models with strong code scores, plus the current base**
- MiniCPM5-2B (OpenBMB; released Sept 7, 2026 [2nd]):
  - 2,516,756,480 total and 1,981,982,720 non-embedding parameters. A standard `LlamaForCausalLM`: 42 layers, 16 query / 2 KV heads, 131,072 context. Apache 2.0.
  - Post-training: 400B tokens of deep-thinking SFT → critic-based RL (JustRL II) → on-policy distillation from 16 expert models (full-vocabulary reverse KL).
  - Released checkpoints: Base, Midtrain, SFT and final, plus GGUF, MLX, GPTQ and a draft model.
  — [HF card](https://huggingface.co/openbmb/MiniCPM5-2B); [orcarouter](https://www.orcarouter.ai/blog/minicpm5-2b-open-weights-release)
- MiniCPM5-2B scores [SR; "reproduced internally," a few rows from Artificial Analysis]: LCB v6 69.1, LCB-Pro 25Q2 Easy 68.0, OJBench 32.5, SWE-bench Verified 46.4, AIME 2025 86.5, MATH-500 94.6. Qwen3.5-4B in the same table: 56.4 / 58.3 / 24.8 / 33.6 / 78.8 / 99.0 — [HF card](https://huggingface.co/openbmb/MiniCPM5-2B). Liquid's table lists MiniCPM5-2B at LCB v6 70.52 [XL] — [LFM3 card](https://huggingface.co/LiquidAI/LFM3-1.7B).
- VibeThinker-1.5B (WeiboAI; report Nov 9, 2025):
  - Fine-tuned from Qwen2.5-Math-1.5B with the "Spectrum-to-Signal Principle": SFT that explores diverse solutions, then RL that reinforces the correct ones. Total cost $7,800. MIT license.
  - Scores [SR]: AIME24 80.3, AIME25 74.4, HMMT25 50.4, LCB v5 55.9, LCB v6 51.1.
  - Recommended settings: up to 40,960 tokens, temperature 0.6 or 1.0, top_p 0.95. Intended only for competitive math and coding, in English.
  — [HF card](https://huggingface.co/WeiboAI/VibeThinker-1.5B); [arXiv 2511.06221](https://arxiv.org/pdf/2511.06221). The paper's table reportedly lists DeepSeek-R1-Distill-Qwen-1.5B at LCB v6 12.8 (from the search snippet).
- DeepSeek-R1-Distill-Qwen-1.5B (the current base): MIT license, built from Qwen2.5-Math-1.5B. Scores [SR]: AIME24 28.9, MATH-500 83.9, LCB 16.9, Codeforces rating 954. Evaluated with 32,768-token generation at temperature 0.6. — [HF card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B). Config: 28 layers, hidden size 1536, 12 query / 2 KV heads, untied embeddings, vocabulary 151,936, rope_theta 10,000. — [config.json](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/config.json)

### Inferences
- The 2026 jump (LFM3, MiniCPM5) comes from post-training scale: 0.4–1.75T SFT tokens, multi-stage RL, and on-policy distillation from many expert teachers. The team's "ample compute" is best spent on RL with test-case rewards and a length penalty on top of such a base, not on more small SFT sets.
- The current base is now outperformed at the same size and architecture: VibeThinker-1.5B posts about 3× its LCB, on the same Qwen2.5-Math-1.5B lineage.
- Vendors themselves say LFM2/LFM2.5, Granite Nano and BitNet are not coding models. Hymba and the small Falcon-H1 models report no competitive LCB scores.

### Gaps
- Falcon-E (the BitNet-style Falcon) was not researched; no source was fetched.
- No LCB for Hymba, Granite Nano or LFM2.5; vendors did not report one.
- SmolLM2/3 licenses and SmolLM2 token counts are not confirmed in the fetched pages.
- An independent Artificial Analysis comparison of LFM2.5-Thinking vs Qwen3-1.7B appeared only in search snippets and was not verified.
- Axolotl support was confirmed only for Gemma 3n.

## 4. Which architecture choices matter most for phone latency and memory with long outputs (numbers)

### Takeaway
For long chain-of-thought on a phone, the biggest cost driver is how many tokens the model generates. Next comes KV-cache bytes per token, which equals the number of layers keeping full-length K/V × KV heads × head_dim. Third is weight bytes (4-bit QAT gives about 0.6–1.4 GB for 1–2.5B models). Hybrids that keep only 2–8 full-attention layers cut KV 3–10× or more against dense-GQA transformers: Gemma 4 E2B, LFM3, Qwen3.5 and Nemotron-Flash. Speculative decoding via multi-token prediction adds 1.5–2.2×. MoE helps compute but is limited by phone RAM.

### Cited Findings
- [CALC] KV cache, bf16 (= 2 × full-KV layers × KV heads × head_dim × 2 bytes); int8 halves it. Configs come from the cards and configs cited in Section 3.

  | Model | Full-KV layers × KV heads × head_dim | KV/token | 8K | 32K | 128K |
  |---|---|---|---|---|---|
  | DeepSeek-R1-Distill-Qwen-1.5B (current) | 28 × 2 × 128 | 28 KiB | 0.23 GB | 0.94 GB | – |
  | Qwen3-1.7B | 28 × 8 × 128 | 112 KiB | 0.94 GB | 3.76 GB | (40,960 max) |
  | MobileLLM-R1-950M (head_dim assumed 1536/24 = 64) | 22 × 6 × 64 | 33 KiB | 0.28 GB | 1.11 GB | (32K max) |
  | MiniCPM5-2B (head_dim 128 inferred) | 42 × 2 × 128 | 42 KiB | 0.35 GB | 1.41 GB | 5.64 GB (Liquid reports 5.6 GB) |
  | Qwen3.5-2B (+ fixed DeltaNet state, ~9–19 MB) | 6 × 2 × 256 | 12 KiB | 0.10 GB | 0.40 GB | 1.61 GB |
  | Qwen3.5-4B | 8 × 4 × 256 | 32 KiB | 0.27 GB | 1.07 GB | 4.29 GB (Liquid reports 4.3 GB) |
  | LFM3-1.7B | reported | – | – | – | 1.6 GB [SR] |
  | Gemma 4 E2B | reported, int8 | ~1.5 KB | – | 0.05 GB [SR] | – |

  Sources for the reported values: [LFM3 blog](https://www.liquid.ai/blog/lfm3.md); [Gemma 4 report](https://huggingface.co/papers/2607.02770)
- Grouped-query attention: the current base uses 12 query / 2 KV heads, while Qwen3-1.7B uses 16 / 8. With the same 28 layers, Qwen3-1.7B therefore carries 4× more KV per token. — [config](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/config.json); [config](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json)
- Sliding-window and local-global attention:
  - Gemma 3: KV overhead at 32K drops from 60% ("global only") to under 15%. — [Gemma 3 report](https://arxiv.org/pdf/2503.19786)
  - MobileLLM-Pro (3:1 ratio, 512 window) at 8K: KV goes from 117 MB to 40 MB and prefill is 1.8× faster. Yet S25 CPU decode still falls from 33.6 tok/s (2K) to 19.7 (8K). — [MobileLLM-Pro card](https://huggingface.co/facebook/MobileLLM-Pro)
- Hybrid convolution + attention [SR]:
  - LFM2.5-1.2B vs Qwen3-1.7B on a Galaxy S25 Ultra CPU (Q4_0): decode 70 vs 40 tok/s, prefill 335 vs 181, memory 719 vs 1,306 MB. — [LFM2.5 blog](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
  - LFM3 decodes 1.8–2.8× faster than the next-fastest model, and about 25% faster than LFM2 at 8K. — [LFM3 blog](https://www.liquid.ai/blog/lfm3.md)
- Hybrid SSM + attention [SR]:
  - Hymba: 11.67× smaller cache and 3.49× throughput vs Llama-3.2-3B. — [Hymba](https://arxiv.org/abs/2411.13676)
  - Granite 4.0-H: over 70% less RAM. — [IBM](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)
  - Phi-4-mini-flash: up to 10× throughput and 2–3× lower latency at 32K generation (on an A100). — [Azure blog](https://azure.microsoft.com/en-us/blog/reasoning-reimagined-introducing-phi-4-mini-flash-reasoning/)
  - Nemotron-Flash-1B: 1.9× lower latency than Qwen3-0.6B, with depth mattering most for small-batch latency. — [Nemotron-Flash](https://arxiv.org/pdf/2511.18890)
  - Qwen3.5 uses a 3:1 Gated DeltaNet : attention layout plus multi-token prediction. — [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)
- KV sharing across layers: Gemma 3n reports 2× faster prefill than Gemma 3 4B — [Gemma 3n guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/). Gemma 4 shares KV in 20/35 (E2B) and 18/42 (E4B) layers — [Gemma 4 report](https://huggingface.co/papers/2607.02770).
- Quantization: MobileLLM-Pro's int4 QAT loses 0.4–1.3% (vs 34% for post-training quantization) at 590 MB — [card](https://huggingface.co/facebook/MobileLLM-Pro). Gemma 4 E2B drops from 4.6 GB to 0.8 GB with int2/int4 weights — [report](https://huggingface.co/papers/2607.02770). BitNet runs at 0.4 GB and 29 ms/token on CPU, but only with bitnet.cpp — [BitNet](https://arxiv.org/pdf/2504.12285).
- Speculative decoding (multi-token prediction): Gemma 4 decodes up to 2.2× faster on mobile GPUs and 1.5× on mobile CPUs. — [LiteRT-LM blog](https://developers.googleblog.com/blazing-fast-on-device-genai-with-litert-lm/)
- Mixture of experts with few active parameters:
  - LFM2-8B-A1B (8.3B total, 1.5B active) decodes "up to 5×" faster than Qwen3-1.7B [2nd]. — [AIbase](https://www.aibase.com/news/21859)
  - gpt-oss-20b has 3.6B active but a 12.8 GiB checkpoint and needs at least 16 GB of memory. — [gpt-oss card](https://arxiv.org/html/2508.10925v1)
- Weight sharing: MobileLLM's adjacent-block sharing avoids weight movement, the bottleneck on memory-bound phones. — [MobileLLM](https://arxiv.org/abs/2402.14905)

### Inferences
- Lever ranking for competitive programming on a phone:
  1. Output length. At 20–40 tok/s, an 8K-token trace takes 3.4–6.8 minutes and a 32K trace 14–27 minutes, while a 2K blueprint-style output takes 50–100 s [CALC]. Shortening traces gains more than any architecture change; this supports the team's router + blueprint idea and RL with a length penalty (as in LFM3).
  2. KV per token: few global layers, few KV heads, sliding windows, KV sharing, int8 KV.
  3. 4-bit QAT weights.
  4. Multi-token-prediction or speculative decoding.
  5. Convolution/SSM hybrids for CPU prefill and decode.
- The current base already has lean KV (2 KV heads). Its phone problem is verbosity, not attention design. Moving to Qwen3-1.7B would make KV 4× worse; moving to MiniCPM5-2B, 1.5× worse.
- Small-active MoE is the wrong fit for the phone memory budget: every expert must be resident in RAM.

### Gaps
- No neutral same-device phone benchmark compares the candidates. Decode slowdown at 16–32K on phones is unmeasured for the hybrids.
- Phone-runtime support for DeltaNet/convolution kernels (llama.cpp, ExecuTorch) was not checked.
- Battery data exists only for Gemma 3 270M.

## 5. OpenAI gpt-oss-20b/120b: training, format, code results, phone feasibility, and the strongest small open alternatives

### Takeaway
gpt-oss models (Aug 2025, Apache 2.0) are MoE models with MXFP4-quantized experts, trained by distillation plus o3-style chain-of-thought RL. They use the harmony format with low/medium/high reasoning effort. gpt-oss-20b scores Codeforces Elo 2230 without tools at high effort [SR] and LCB v6 74.6 [XL]. It can technically run on 16 GB+ Snapdragon flagships but is "barely usable" there [3P]. Its best use here is as an Apache-2.0 teacher, with effort levels that control trace length. At 2B, the 2026 small models (LFM3-1.7B, MiniCPM5-2B) now self-report gpt-oss-20b-level LCB v6.

### Cited Findings
- The models were trained with "large-scale distillation and reinforcement learning," and are released under Apache 2.0 (weights, inference code, tool environments, tokenizers). The model card was submitted to arXiv on Aug 8, 2025. — [arXiv abstract](https://arxiv.org/abs/2508.10925)
- Architecture — [model card](https://arxiv.org/html/2508.10925v1):

  | | gpt-oss-120b | gpt-oss-20b |
  |---|---|---|
  | Total / active parameters | 116.83B / 5.13B | 20.91B / 3.61B |
  | Layers | 36 | 24 |
  | Experts (top-4 routing) | 128 | 32 |
  | Checkpoint size | 60.8 GiB | 12.8 GiB |

  Shared by both: residual dimension 2880; attention alternating between a 128-token banded window and dense; 64 query heads of dimension 64; GQA with 8 KV heads; a learned attention-sink bias; 131,072 context via YaRN; tokenizer o200k_harmony (201,088 tokens).
- MXFP4 (about 4.25 bits per parameter) is applied to the MoE weights, which are over 90% of parameters. The 120b fits on one 80 GB GPU and the 20b in 16 GB. — [model card](https://arxiv.org/html/2508.10925v1)
- Pretraining: text-only, trillions of tokens focused on STEM, coding and general knowledge; knowledge cutoff June 2024. The 120b took 2.1M H100-hours; the 20b almost 10× fewer. — [model card](https://arxiv.org/html/2508.10925v1)
- Post-training: chain-of-thought RL similar to o3. The harmony format has channels for analysis (CoT), commentary (function calls) and final (answers). Reasoning effort (low/medium/high) is set in the system prompt. Tools: browsing, Python and developer-defined functions. — [model card](https://arxiv.org/html/2508.10925v1)
- Results [SR], low / medium / high effort — [model card](https://arxiv.org/html/2508.10925v1):

  | Benchmark | gpt-oss-120b | gpt-oss-20b |
  |---|---|---|
  | Codeforces Elo, no tools | 1595 / 2205 / 2463 | 1366 / 1998 / 2230 |
  | Codeforces Elo, with tools | 1653 / 2365 / 2622 | 1251 / 2064 / 2516 |
  | SWE-bench Verified (%) | 47.9 / 52.6 / 62.4 | 37.4 / 53.2 / 60.7 |
  | Aider Polyglot (%) | 24.0 / 34.2 / 44.4 | 16.6 / 26.6 / 34.2 |
  | AIME 2025, no tools (%) | 50.4 / 80.0 / 92.5 | 37.1 / 72.1 / 91.7 |

- LCB v6 [XL, Qwen's table]: gpt-oss-120b 82.7, gpt-oss-20b 74.6; OJBench 41.5 / 36.3. — [Qwen3.5-4B card](https://huggingface.co/Qwen/Qwen3.5-4B)
- Running on phones:
  - Qualcomm says the model runs on-device on flagship Snapdragon devices (aimed at developers) [SR]. — [Qualcomm](https://www.qualcomm.com/news/onq/2025/08/openai-model-on-device-snapdragon)
  - Android Authority's test on a ROG phone with Snapdragon 8 Elite (Aug 2025): it "can just about run... barely usable," with tokens arriving "slower than I can read" [3P]. — [Android Authority](https://www.androidauthority.com/i-installed-gpt-oss-on-my-phone-3584439/)
  - Another Android Authority piece says most smartphones cannot run it. — [Android Authority](https://www.androidauthority.com/openai-gpt-oss-20b-qualcomm-snapdragon-launch-3584207/)
  - A Nexa AI demo (Oct 2025) ran it in Nexa Studio on a Snapdragon "Gen 5" phone. It needs at least 16 GB of RAM, so the 12 GB iPhone 17 Pro Max is presumably not enough. — [Simon Willison](https://simonwillison.net/2025/Oct/10/gpt-oss-20b-snapdragon)
- Strongest small open alternatives, by LCB v6:
  - LFM3-1.7B: 73.46 [SR]
  - MiniCPM5-2B: 69.1 [SR] / 70.52 [XL]
  - Qwen3.5-9B: 65.6 [SR] / 69.86 [XL]
  - Qwen3.5-4B: 55.8 [SR] / 56.4 [XL]
  - Gemma 4 E4B: 52.0 [SR] / 53.9 [XL]
  - VibeThinker-1.5B: 51.1 [SR]
  - Gemma 4 E2B: 44.0 [SR] / 42.9 [XL]

  On SWE-bench Verified: MiniCPM5-2B 46.4 [SR] and LFM3-1.7B 43.40 [SR], against gpt-oss-20b at 60.7 (high effort). — [LFM3 card](https://huggingface.co/LiquidAI/LFM3-1.7B); [MiniCPM5 card](https://huggingface.co/openbmb/MiniCPM5-2B); [Qwen3.5-4B card](https://huggingface.co/Qwen/Qwen3.5-4B); [Gemma 4 report](https://huggingface.co/papers/2607.02770); [VibeThinker](https://huggingface.co/WeiboAI/VibeThinker-1.5B)

### Inferences
- gpt-oss-20b is not a practical phone competitive-programming model: its 12.8 GiB of weights need a 16–24 GB phone, and its long high-effort traces run slowly. It is an attractive Apache-2.0 teacher. Low effort gives shorter traces (Elo 1366) and high effort longer ones (2230), so it can generate trace-length-graded distillation data. Harmony's analysis/final split is a ready template for blueprint-then-code outputs.
- The 2026 2B models' claims of gpt-oss-level LCB need independent re-testing on problems after their data cutoffs, at phone-realistic token budgets.

### Gaps
- No measured tok/s for gpt-oss-20b on a phone; no LCB from OpenAI; no estimate of harmony's token overhead.

## 6. Which bases (≤2B) are best for a phone-deployable competitive-programming coder? Comparison and ranked shortlist

### Takeaway
On current evidence, the best ≤2B bases are:
- **MiniCPM5-2B**: Apache 2.0, plain Llama architecture, a full Base → Midtrain → SFT → RL checkpoint ladder, LCB v6 about 69–70 as reported by two labs.
- **LFM3-1.7B**: the highest self-reported LCB v6 (73.5) and the best phone efficiency, with a base checkpoint. Caveats: the LFM license and its "not for complex multi-step reasoning" disclaimer.

**Gemma 4 E2B** is the safest deployment-first choice (Apache 2.0, LiteRT-LM/AICore, tiny KV), with mid-tier code (about 43–44). **VibeThinker-1.5B** is the best same-architecture drop-in replacement for the current DeepSeek-R1-Distill-Qwen-1.5B (LCB v6 51.1, MIT). For the router, Gemma 3 270M or LFM3's d1 decision models fit.

### Cited Findings
| Model (release) | Params | Context / vocab | License | Base ckpt | Code scores | Phone notes | Tuning support | Sources |
|---|---|---|---|---|---|---|---|---|
| LFM3-1.7B (Oct 27, 2026) | 1.73B | 256K / 128K | LFM Open v1.0 (<$10M revenue; non-profit research exempt) | Yes | LCB v6 73.46, LCB Pro 31.74 [SR] | ~1 GB 4-bit; KV 1.6 GB at 128K [SR] | TRL SFT/DPO/GRPO; vLLM, llama.cpp, MLX | [card](https://huggingface.co/LiquidAI/LFM3-1.7B), [blog](https://www.liquid.ai/blog/lfm3.md), [license](https://www.liquid.ai/en/lfm-license) |
| MiniCPM5-2B (Sept 2026) | 2.52B (1.98B non-embedding) | 131K / n.s. | Apache 2.0 | Yes (Base, Midtrain, SFT) | LCB v6 69.1 [SR], 70.52 [XL]; OJBench 32.5 | KV 5.6 GB at 128K [XL]; no phone tok/s | LlamaForCausalLM; GGUF/MLX/GPTQ | [card](https://huggingface.co/openbmb/MiniCPM5-2B) |
| Gemma 4 E2B (Apr 2, 2026) | 2.3B effective / 5.1B | 128K / 262K | Apache 2.0 | Yes (PT + IT) | LCB v6 44.0 [SR], 42.9 [XL]; Codeforces 633 | 0.8 GB + 0.05 GB KV at 32K; S26 Ultra GPU 52 tok/s | Unsloth SFT + GRPO (~9 GB); LiteRT-LM | [card](https://huggingface.co/google/gemma-4-E2B-it), [report](https://huggingface.co/papers/2607.02770), [Unsloth](https://unsloth.ai/docs/models/gemma-4/train) |
| VibeThinker-1.5B (Nov 2025) | 1.5B (Qwen2.5-Math-1.5B) | 40,960-token generation recommended | MIT | No (it is a fine-tune) | LCB v5 55.9, v6 51.1 [SR] | Same architecture as current base (28 KiB/token KV [CALC]) | Any tool supporting Qwen2.5 | [card](https://huggingface.co/WeiboAI/VibeThinker-1.5B) |
| Qwen3-1.7B (2025) | 1.7B | 32K / 151,936 | Apache 2.0 | Yes | LCB v5 33.2 thinking [SR] | 112 KiB/token KV [CALC]; S25 Ultra 40 tok/s, 1,306 MB [XL] | Mature | [Qwen3 report](https://arxiv.org/html/2505.09388v1), [LFM2.5 blog](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai) |
| Qwen3.5-2B (Feb/Mar 2026) | 2B | 262K / 248,320 | Apache 2.0 | Yes | No code scores [SR]; LCB v6 20.2 [XL]; thinking loops | 12 KiB/token KV [CALC] | Unsloth (vision notebooks) | [card](https://huggingface.co/Qwen/Qwen3.5-2B), [MiniCPM5 card](https://huggingface.co/openbmb/MiniCPM5-2B) |
| Qwen3.5-4B (stretch) | 4B | 262K / 248,320 | Apache 2.0 | Yes | LCB v6 55.8 [SR], 56.4 [XL] | KV 4.3 GB at 128K | Unsloth GRPO (vision) | [card](https://huggingface.co/Qwen/Qwen3.5-4B) |
| MobileLLM-R1-950M (Sept 2025) | 949M | 32K / 128K | FAIR NC | Yes | LCB v6 19.9 [SR] | 33 KiB/token [CALC] | Training code open | [card](https://huggingface.co/facebook/MobileLLM-R1-950M) |
| DeepSeek-R1-Distill-Qwen-1.5B (current) | 1.5B | 32K eval / 151,936 | MIT | (Qwen2.5-Math-1.5B) | LCB 16.9, Codeforces 954 [SR]; v6 19.9 [XL Meta]; v5 13.2 [XL Qwen] | 28 KiB/token | Any | [card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
| MobileLLM-Pro (Oct 2025) | 1.08B | 128K / 202,048 | FAIR NC | Yes | HumanEval 59.8, MBPP 46.8 [SR] | 590 MB int4; 33.6→19.7 tok/s (2K→8K) | – | [card](https://huggingface.co/facebook/MobileLLM-Pro) |
| Gemma 3n E2B (Jun 2025) | 2B effective / 5–6B raw | 32K | Gemma Terms of Use | PT benchmarks reported | LCB v5 18.6, HumanEval 66.5 [SR] | 2 GB minimum | TRL, Unsloth, Axolotl, NeMo | [card](https://huggingface.co/google/gemma-3n-E2B-it) |
| Granite-4.0-H-1B (Oct 2025) | ~1.5B | n.s. | Apache 2.0 | Yes | HumanEval 73 [SR] | Hybrid Mamba-2 | – | [HF](https://huggingface.co/ibm-granite/granite-4.0-h-1b-base) |
| Gemma 3 270M (Aug 2025) | 270M | 32K / 256K | Gemma Terms of Use | Yes | LCB v6 0.0 [XL] | 0.75% battery per 25 conversations | HF, Unsloth, JAX | [blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/) |

- Reported scores assume long generation budgets: Qwen3.5 recommends 81,920 output tokens for hard programming — [card](https://huggingface.co/Qwen/Qwen3.5-2B); VibeThinker 40,960 — [card](https://huggingface.co/WeiboAI/VibeThinker-1.5B); DeepSeek evaluated at 32,768 — [card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).

### Inferences
#### Ranked shortlist: main coder base
1. **MiniCPM5-2B.**
   - Why: best mix of score (LCB v6 69–70, from two labs), permissive license (Apache 2.0), plain-Llama compatibility with every tuning and serving tool (Unsloth, TRL, Axolotl, vLLM rollouts for GRPO, llama.cpp), and released Base, Midtrain and SFT checkpoints. Those checkpoints allow clean ablations, e.g. "our blueprint SFT + RL applied to Base vs SFT."
   - Costs: 2.5B total parameters (about 1.4 GB at 4-bit [CALC]) and heavier KV (42 KiB/token). Use int8 KV and short traces.
2. **LFM3-1.7B.**
   - Why: best phone efficiency (convolution hybrid, 1.6 GB KV at 128K, fastest decode) and the highest self-reported LCB. A base checkpoint, TRL GRPO notebooks, and a sibling d1 decision model for routing.
   - Risks: only three weeks old; no neutral verification yet; the card disclaims complex multi-step reasoning; the non-OSI LFM license (fine for university research, but check terms before redistributing derivatives).
3. **Gemma 4 E2B.**
   - Why: the deployment-first pick, with Apache 2.0, Google's LiteRT-LM/AICore stack, a 0.8 GB footprint, near-zero KV growth and Unsloth GRPO support.
   - Cost: LCB is about 25 points below options 1–2.
4. **VibeThinker-1.5B.**
   - Why: the strongest true-1.5B model and an architecture match with the current base. The paper can swap bases without touching tokenizer or LoRA code and report a like-for-like comparison.
   - Costs: very verbose (40K-token budget) and narrow, so it needs brevity training.
5. **Qwen3.5-4B** if a ~4B model is acceptable on the target phone (LCB v6 about 56, Apache 2.0, hybrid DeltaNet with modest KV). **Qwen3-1.7B** only for continuity and tooling maturity. Avoid Qwen3.5-2B for code (LCB 20.2, thinking loops).
6. **MobileLLM-R1-950M**: research only (open data, NC license), useful for sub-1B ablations.

**Not recommended as the coder base:**
- Gemma 3 270M and Gemma 3n (superseded).
- LFM2/LFM2.5: the vendor says they are not for programming.
- Granite Nano: no LCB.
- BitNet 2B4T: tied to its own runtime.
- Falcon-H1 small and Tiny: weak or unknown LCB.
- Hymba: no code results.
- Phi-4-mini(-flash): 3.8B.
- gpt-oss-20b: not phone-practical.

#### Router / decision model
- Gemma 3 270M: Google's intended use includes query routing; INT4 QAT; very low battery cost.
- LFM3 d1-2B: purpose-built for routing, scoring every option with calibrated probabilities in one forward pass. But it is coder-sized, so it only pays off if it replaces the coder's thinking.
- Falcon-H1-Tiny (~90M).
- Cheapest option: fine-tune the existing MiniLM retriever as a topic classifier, or add a classification head or first-token prompt to the coder, which needs no second model on the phone.

#### Teachers for distillation data
- gpt-oss-120b/20b (Apache 2.0; LCB v6 82.7 / 74.6 [XL]; effort levels control trace length).
- Qwen3.5-9B (Apache 2.0; 65.6).
- The chosen base's own RL-tuned sibling.

#### Evaluation advice
- Report Pass@1 against output-token budget (e.g. 1K, 2K, 4K, 8K), since phone viability depends on it.
- Re-test all vendor LCB claims on problems released after each model's data cutoff.

### Gaps
- No neutral evaluation of LFM3-1.7B or MiniCPM5-2B, and none at phone-realistic token budgets.
- No measured phone tok/s for MiniCPM5-2B or VibeThinker-1.5B.
- Axolotl and Unsloth support for LFM3 and MiniCPM5 is not verified; Unsloth support for LFM3 is only an ecosystem-list mention.
- Python tokenization efficiency across the tokenizers (151K / 248K / 262K / 128K) was not measured.
- Contamination of LCB v6 (2024–25 problems) in 2026 models' RL data is unknown.
- Gemma 3n base-checkpoint naming was not verified.
