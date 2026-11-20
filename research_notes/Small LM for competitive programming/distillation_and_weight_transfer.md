# Knowledge Distillation and Weight Transfer for a ~1.5B Qwen-Family Competitive-Programming Student (state as of 2026-11-19)

Conventions used below: **[CODE]** = public implementation found/confirmed; **[NO CODE CONFIRMED]** = I could not confirm a released implementation; **[SUPERSEDED]** = method/tool has a newer replacement or is no longer maintained. "LCB" = LiveCodeBench; version windows differ between papers (v5 = 2024-08..2025-02, v6 = 2024-08..2025-05), so cross-paper LCB numbers are only roughly comparable. All numbers are as reported by the cited source.

---

## 1. Distillation objectives: sequence-level KD, forward vs reverse KL (MiniLLM), GKD / on-policy distillation (Thinking Machines, Qwen3), DistiLLM / DistiLLM-2, TAID, skew KL, SKD: when each is best, compute cost, reported gains for ~1–2B reasoning/code students

### Takeaway
In 2025–2026 the standard recipe for small reasoning students is **off-policy SFT on teacher traces first, then on-policy distillation**. In on-policy distillation the student samples and the teacher scores every token, usually with per-token reverse KL. On Qwen3-8B, on-policy distillation beat RL by +6.8 AIME'24 and +7.4 LiveCodeBench points while using about 1/10 of the GPU hours. Thinking Machines reproduced a 9–30x cost advantage over scaling SFT. Refinements of the logit loss (skew KL, DistiLLM-2's contrastive loss, TAID's interpolated teacher) add only about 1–3 points for 1.5B code/math students.

### Cited Findings

**Sequence-level KD (SFT on teacher outputs)**
- Sequence-level KD trains the student on the teacher's generated output sequences rather than on per-token teacher distributions. It is the classic "SFT on teacher outputs" formulation. **[CODE]** It is trivially implemented with any SFT trainer. — [Kim & Rush 2016](https://arxiv.org/abs/1606.07947)
- DeepSeek-R1 released six dense distilled students (1.5B–70B, Qwen2.5 and Llama3 bases). They were made by SFT on R1-generated data and showed that distillation from a strong reasoner works for small dense models. — [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948)
- Reference point for the team's current base: DeepSeek-R1-Distill-Qwen-1.5B scores 16.9 on LCB (v5, 8/1/24–2/1/25), Codeforces rating 615 (1.9th percentile) and HumanEval+ 58.3, in Agentica's evaluation. — [DeepCoder-1.5B-Preview model card](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)
- Qwen3 calls this stage "off-policy distillation". Students are trained on teacher outputs generated in both /think and /no_think modes, to build basic reasoning and mode-switching ability, before the on-policy phase. — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388)

**Forward vs reverse KL; MiniLLM**
- MiniLLM replaces forward KL with reverse KL for generative LMs, so the student does not overestimate low-probability regions of the teacher. It optimizes the sequence-level reverse KL with a policy-gradient-style method (ICLR 2024). **[CODE]** microsoft/LMOps (minillm). — [MiniLLM](https://arxiv.org/abs/2306.08543)
- "Rethinking KL divergence in KD for LLMs" argues that forward KL and reverse KL share the same optimal solution and converge after enough training. Early in training, forward KL focuses on the head of the distribution and reverse KL on the tail. The paper proposes Adaptive KL (AKL). — [Wu et al.](https://arxiv.org/abs/2404.02657)
- Thinking Machines describes reverse KL as mode-seeking: it converges on one teacher behavior rather than spreading probability over alternatives. It is "unhackable" in the sense that low reverse KL reliably means the teacher assigns high probability to the student's tokens. — [Thinking Machines, "On-Policy Distillation" (Oct 27, 2025)](https://thinkingmachines.ai/blog/on-policy-distillation/)

**GKD / on-policy distillation**
- GKD (ICLR 2024) trains the student on its own sampled sequences with teacher feedback. It allows alternative divergences such as generalized JSD and integrates with RL fine-tuning. **[CODE]** — [Agarwal et al., GKD](https://arxiv.org/abs/2306.13649)
- TRL's GKDTrainer exposes `lmbda`, the fraction of student-generated (on-policy) data, and `beta`, the generalized-JSD interpolation (beta=0 gives forward KL, beta=1 reverse KL). It also has a `seq_kd` option. **[CODE]** — [TRL GKD docs](https://huggingface.co/docs/trl/gkd_trainer)
- **Thinking Machines setup.** The student is Qwen3-8B-Base, warm-started by SFT on 400K OpenThoughts-3 prompts (QwQ-32B-generated traces), which gives 60% AIME'24. The loss is per-token reverse KL with discount factor 0, implemented as RL with advantage = −reverse KL, so no reward model is needed. The default configuration is 64 prompts per batch × 4 samples per prompt. The post names Qwen3-32B as teacher but says Qwen3-8B was used in the runs because it performed slightly better. — [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **Thinking Machines results and cost.**
  - On-policy distillation reached 70% AIME'24 in about 150 steps (about 77K prompts). It used 8.4×10¹⁹ teacher FLOPs plus 8.2×10¹⁹ student FLOPs.
  - An extrapolated SFT-2M run would also reach about 70%, but would need 3.4×10²¹ teacher FLOPs plus 1.5×10²¹ student FLOPs.
  - That makes on-policy distillation 9–30x cheaper (about 18x in GPU-hours). RL reached 68%. — [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **Thinking Machines caveats.** On-policy distillation "works best when the student's initialization already places reasonable probability on the teacher's behavior"; otherwise much larger batches are needed. The experiments distilled only on sampled tokens, not top-k logits. Combining per-token distillation rewards with sequence-level environment rewards is left as future work. **[CODE]** Tinker cookbook `tinker_cookbook/recipes/distillation`. A June 2026 update moved the recipe to Qwen3.5-9B-Base (student) and Qwen3.5-9B (teacher). — [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **Thinking Machines, LoRA.** In the post's LoRA experiment (rank 32), the gap to full fine-tuning shrank from 13% after SFT to 6% after on-policy distillation. — [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **Qwen3 "Strong-to-Weak Distillation".**
  - Teachers: Qwen3-32B or Qwen3-235B-A22B.
  - Students: Qwen3-0.6B, 1.7B, 4B, 8B, 14B (dense) and Qwen3-30B-A3B (MoE).
  - Two phases: off-policy, then on-policy, where the student samples in /think or /no_think mode and its logits are aligned to the teacher's by minimizing KL.
  - Claimed cost: about 1/10 of the GPU hours of the four-stage pipeline. — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388)
- **Qwen3 Table 21 (Qwen3-8B)**, with pass@64 in parentheses. — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388)

  | Method | AIME'24 | AIME'25 | MATH500 | LCB v5 | GPQA-D | GPU hours |
  |---|---|---|---|---|---|---|
  | Off-policy distillation | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 55.6 | n/a |
  | + RL | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 61.3 | 17,920 |
  | + On-policy distillation | 74.4 (93.3) | 65.5 (86.7) | 97.0 | 60.3 | 63.3 | 1,800 |

  On-policy distillation raised pass@64, which the authors read as an expanded exploration space. RL left pass@64 unchanged.
- **HF H4 / TRL on-policy study (Oct 29, 2025), same tokenizer, Countdown task.**
  - Distillation recovered over 80% of teacher performance.
  - Fully on-policy (λ=1) beat lower λ values and SFT.
  - Qwen2.5-1.5B-Instruct rose from 0.076 to 0.6515 with a Qwen3-4B-Instruct-2507 teacher.
  - KD scored about 2x GRPO with matched tokenizers. — [HF "Unlocking On-Policy Distillation for Any Model Family"](https://huggingfaceh4-on-policy-distillation.hf.space/)
- **Same study, code domain.**
  - Codeforces SFT moved Qwen3-4B from LCB 35.1 to 40.29 but dropped IFEval from 83.4 to 79.48.
  - A following GKD step on Tulu-3 prompts restored IFEval to 82.8 with LCB 39.8.
  - So on-policy distillation can undo the forgetting caused by SFT. — [HF H4](https://huggingfaceh4-on-policy-distillation.hf.space/)
- **Speculative KD (SKD, ICLR 2025).** The student drafts tokens and the teacher replaces those outside its top-K. Accepting all teacher tokens recovers supervised KD; accepting all student tokens recovers on-policy KD. SKD behaves like supervised KD early in training and like on-policy KD later. It is reported to beat both on translation, summarization, math and instruction following. **[CODE]** (per the search summary; repository URL not checked) — [SKD arXiv](https://arxiv.org/abs/2410.11325v2); [ICLR poster](https://iclr.cc/virtual/2025/poster/30392)

**Skew KL / DistiLLM / DistiLLM-2**
- DistiLLM (ICML 2024) introduced a skew KL divergence loss and an adaptive off-policy scheme for reusing student-generated outputs efficiently. It reports up to 4.3x speedup over recent KD methods. **[CODE]** github.com/jongwooko/distillm. **[SUPERSEDED by DistiLLM-2]** — [DistiLLM](https://arxiv.org/abs/2402.03898)
- DistiLLM-2 (ICML 2025 Spotlight) is contrastive. It applies skew KL on teacher responses, raising their likelihood, and skew reverse KL on student responses, lowering theirs. **[CODE]** — [DistiLLM-2 abstract](https://arxiv.org/abs/2503.07067); [code](https://github.com/jongwooko/distillm-2)
- **DistiLLM-2 code results** (pass@1, HumanEval / MBPP / average). — [DistiLLM-2 paper](https://arxiv.org/html/2503.07067)
  - Qwen2.5-Coder-7B-Inst → Qwen2.5-Coder-1.5B: DistiLLM-2 42.24 / 62.70 / 52.47; GKD 51.38 average; DistiLLM 50.90 average.
  - DS-Coder-6.9B → DS-Coder-1.3B: DistiLLM-2 67.79 average vs GKD 64.61.
- **DistiLLM-2 math results** (GSM8K / MATH average). — [DistiLLM-2 paper](https://arxiv.org/html/2503.07067)
  - Qwen2.5-Math-7B-Inst → Qwen2.5-Math-1.5B: DistiLLM-2 62.07 (81.20 / 42.94); GKD 60.38; DistiLLM 61.10.

**TAID (Sakana AI)**
- TAID (ICLR 2025 Spotlight) distils toward an intermediate distribution that gradually shifts from the student's initial distribution to the teacher's. It targets the capacity gap, mode averaging and mode collapse. — [TAID arXiv](https://arxiv.org/abs/2501.16937)
- TAID-LLM-1.5B (English): teacher Qwen2-72B-Instruct, student Qwen2-1.5B-Instruct. TinySwallow-1.5B (Japanese): teacher Qwen2.5-32B-Instruct, student Qwen2.5-1.5B-Instruct, released Jan 30, 2025. Sakana describes the method as an "intermediate teacher". — [Sakana TAID page](https://sakana.ai/taid/); [TAID-LLM-1.5B card](https://huggingface.co/SakanaAI/TAID-LLM-1.5B); [TinySwallow-1.5B card](https://huggingface.co/SakanaAI/TinySwallow-1.5B)

**Compute-saving infrastructure for logit KD**
- Gemma 3 samples 256 logits per token, weighted by teacher probability. The student matches the renormalized teacher distribution over those samples with cross-entropy; non-sampled logits get zero target probability. — [Gemma 3 Technical Report](https://arxiv.org/html/2503.19786)
- Arcee DistillKit supports online and offline distillation. Offline mode compresses logits (polynomial approximation + quantization + bit-packing) and samples them with vLLM. Losses include KL, JSD, TVD and hidden-state alignment (MSE/cosine). Arcee distilled Llama-3.1-405B into a 70B model from offline logits (SuperNova). **[CODE]** — [DistillKit GitHub](https://github.com/arcee-ai/DistillKit); [Arcee DistillKit v0.1 post](https://blog.arcee.ai/distillkit-v0-1-by-arcee-ai)
- torchtune, which had a KD recipe, carries a README warning that it is no longer actively maintained; development wound down in 2025. **[SUPERSEDED]** — [torchtune README](https://raw.githubusercontent.com/meta-pytorch/torchtune/main/README.md)

### Inferences
- **When each objective is best (synthesis):**
  - **SFT / sequence-level KD.** Use it to install format, long-CoT behavior and broad coverage when the student starts far from the teacher, and whenever the teacher is closed or uses a different tokenizer. Thinking Machines notes on-policy distillation needs the student to already put reasonable mass on teacher behavior. It is cheap per step but scales poorly: AIME gains are log-linear in data.
  - **Forward-KL logit KD (dense teacher distributions on fixed data).** Use it for pretraining and pruning recovery (Minitron, Gemma, Llama 3.2; Section 4). It needs the same tokenizer and has teacher-forward cost on every token, which can be cached as top-k or sampled logits.
  - **On-policy reverse KL (GKD with λ=1, TM-style, Qwen3 stage 2).** Use it after SFT to fix the exposure bias the student shows on its own trajectories. It gives the best accuracy per GPU-hour in every 2025 comparison found. It needs a same-tokenizer teacher, or GOLD (Section 2).
  - **DistiLLM-2, TAID, skew KL, SKD.** These are drop-in loss refinements on the logit-KD axis. Reported gains over GKD at 1.5B are +1.1 (Qwen2.5-Coder-1.5B) to +1.7 (Qwen2.5-Math-1.5B) points. They are worth an ablation, not a pipeline redesign.
  - **MiniLLM's sequence-level policy-gradient reverse KL.** It has effectively been superseded in practice by token-level on-policy reverse KL (TM, Qwen3), which needs no high-variance sequence-level estimator.
- **Compute ordering.** Per unit of gain: on-policy distillation is cheaper than RL, which is cheaper than SFT scaling (TM: 9–30x; Qwen3: about 10x vs RL). Per training step: SFT is cheapest. Logit KD adds a teacher forward. On-policy adds student sampling plus a teacher forward; teacher prefill is cheap relative to decoding. SKD adds teacher queries during sampling.
- **For a 1.5B student the absolute gains will be smaller than at 8B.** The capacity-gap evidence in Section 6 supports this. A same-family mid-size teacher may beat the largest one: TM preferred Qwen3-8B over Qwen3-32B for an 8B student.
- **The team's Stage 1/Stage 2 is pure sequence-level KD with LoRA.** The largest untested lever is adding an on-policy distillation stage from a same-tokenizer Qwen teacher. That stage needs only problem prompts, not verified solutions.

### Gaps
- I found no public head-to-head of GKD vs DistiLLM-2 vs TAID vs SKD on **competitive programming with long CoT at 1.5B**. DistiLLM-2's code numbers are HumanEval/MBPP with short outputs.
- The TAID GitHub repository URL was not verified (model weights are on HF). MiniLLM and GKD headline numbers were not re-extracted in this session.
- The Thinking Machines statement that Qwen3-8B, not Qwen3-32B, was the teacher actually used should be double-checked in the post's footnotes.
- Whether TRL's GKD has moved into `trl.experimental` in the latest releases was not checked. GOLD is documented as experimental (Section 2).

---

## 2. Cross-tokenizer distillation (teacher and student with different vocabularies): ULD, DSKD, MultiLevelOT, ALM, GOLD; practical recipes and limits; when to prefer a same-tokenizer (Qwen-family) teacher

### Takeaway
Cross-tokenizer logit distillation now works: GOLD for on-policy use, ALM for offline. Its gains over plain SFT on teacher text are small (about 1–3 points in ALM's large-to-small math test), and it degrades sharply as tokenizer overlap falls. A Gemma-3 teacher with a 262k SentencePiece vocabulary feeding a Qwen BPE student is a low-overlap pair. Use Gemma (or any non-Qwen) teacher outputs as **text for SFT**, which is tokenizer-agnostic. Use a **same-tokenizer Qwen teacher** for any logit or on-policy stage.

### Cited Findings
- **ULD (Universal Logit Distillation)** uses an optimal-transport-style loss, comparing sorted probability vectors, to distill across architectures and tokenizers. **[CODE]** **[SUPERSEDED by GOLD (on-policy) and ALM (offline)]** — [ULD](https://arxiv.org/abs/2402.12030)
- **DSKD (EMNLP 2024)** unifies the two models' output spaces. Its cross-model attention aligns representations of models with different vocabularies. **[CODE]** github.com/songmzhang/DSKD. — [DSKD](https://arxiv.org/abs/2406.17328)
- **MultiLevelOT (AAAI 2025)** aligns teacher and student logit distributions at token and sequence level with optimal transport (Sinkhorn distance), without a shared vocabulary. It reports consistent gains over prior cross-tokenizer KD on extractive QA, generative QA and summarization; reasoning and code were not reported. The authors say code and models are on GitHub. — [MultiLevelOT arXiv](https://arxiv.org/abs/2412.14528v1); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34543)
- **ALM (Approximate Likelihood Matching, NeurIPS 2025)** is presented as the first method to distill effectively across fundamentally different tokenizers. Use cases: tokenizer transfer as self-distillation (including subword→byte), large-math-specialist→small-general distillation, and training hypernetworks for zero-shot tokenizer transfer. **[CODE]** github.com/bminixhofer/tokenkit. — [ALM arXiv](https://arxiv.org/abs/2503.20083); [ALM paper](https://arxiv.org/html/2503.20083)
- **ALM, large-to-small math result.** Teacher OpenMath2-Llama3.1-8B (GSM8K / MATH 88.9 / 60.2) into student Gemma2-2B-IT (8.7 average before training). — [ALM paper](https://arxiv.org/html/2503.20083)
  - SFT: 67.2 / 36.2 (average 51.7). DSKD: 50.3. MinED: 49.8.
  - ALM: 68.5 / 36.7 (52.6). ALM+SFT: 70.2 / 36.4 (53.3).
  - The authors note SFT is competitive here and beats MinED and DSKD.
- **ALM, tokenizer transfer to Qwen2's tokenizer** (benchmark averages). — [ALM paper](https://arxiv.org/html/2503.20083)
  - Gemma2-2B-IT: original 58.0, SFT 51.6, DSKD 51.6, MinED 53.0, ALM 55.1.
  - Llama3.2-3B-IT: original 59.3, SFT 55.7, DSKD 51.1, MinED 57.3, ALM 58.6.
  - Gemma3-12B-IT: original 71.2, SFT 64.6, ALM 69.2.
- **ALM limitations (stated by the authors).** About 0.6B training tokens only; only supervised KD was tested, not on-policy; a hidden-state loss slightly hurt subword-to-subword transfer; byte-level transfer always kept a gap; ULD was not evaluated. — [ALM paper](https://arxiv.org/html/2503.20083)
- **GOLD (General Online Logit Distillation) in TRL** extends ULD to on-policy, cross-tokenizer distillation. — [HF H4 blog/space](https://huggingfaceh4-on-policy-distillation.hf.space/); [TRL GOLD docs (mirror)](https://hf-2025.pages.dev/docs/trl/en/gold_trainer.md)
  - **Sequence alignment:** instead of ULD's truncation, it merges token spans using the product rule over continuation tokens. A June 2026 update switched to a UTF-8 byte-offset walker (TRL #5885) to handle BOS mismatches.
  - **Vocabulary alignment:** shared tokens with different IDs are matched one-to-one and trained with the GKD loss; unmatched tokens fall back to ULD sorting.
  - **Loss:** a hybrid L = w1·L_GKD + w2·L_ULD.
  - **Status:** lives in `trl.experimental.gold`, so the API may change.
- **GOLD results (Countdown).** — [HF H4](https://huggingfaceh4-on-policy-distillation.hf.space/)
  - Same tokenizer: over 80% of teacher performance recovered.
  - Qwen3-4B → Llama-3.2-1B: GOLD improved the student's starting score by 25% and recovered 60% of teacher performance; ULD managed +5% and 10%. GOLD was about 20% better than GRPO and 15% better than SFT.
  - Qwen3-4B → Gemma-3-1B: only 0.0305, with a tokenizer-similarity score of 0.063. Llama's similarity is 0.64 under token-content mapping; both score 0 under strict ID matching.
  - Clear gains for Llama appeared only after about 1,000 steps.
  - Recipe: lr 1e-7, effective batch 64, max completion 2048, temperature 1.0, lmbda 0.25, beta 0.0.
- Older token-alignment approaches include MinED (minimum-edit-distance alignment, from FuseLLM). ALM outperformed both MinED and DSKD in its experiments. **[SUPERSEDED by ALM/GOLD for these pairs]** — [ALM paper](https://arxiv.org/html/2503.20083)
- **Zero-shot tokenizer transfer (ZeTT, NeurIPS 2024)** trains a hypernetwork that predicts embeddings for an arbitrary new tokenizer. It is another route to putting teacher and student on one vocabulary. **[CODE]** (tokenkit lineage) — [ZeTT](https://arxiv.org/abs/2405.07883)
- **Vocabulary facts relevant to the brief.**
  - Gemma 3 uses the Gemini 2.0 SentencePiece tokenizer with 262k entries. — [Gemma 3 report](https://arxiv.org/html/2503.19786)
  - Qwen3 uses Qwen's byte-level BPE with 151,669 vocabulary entries. — [Qwen3 report](https://arxiv.org/html/2505.09388)
  - Qwen3.5-2B and Qwen3.6-27B have a 248,320 (padded) token embedding, a new vocabulary that differs from the ~151k Qwen2.5/Qwen3 one. — [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B); [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B)

### Inferences
- **The Stage-2 Gemma-3-27B "blueprint" data was used as text SFT, so the tokenizer mismatch did not matter there.** It matters only if the team wants Gemma's logits, or Gemma as an on-policy teacher.
  - Gemma (262k SentencePiece) and Qwen (151k byte-level BPE) are likely a low-overlap pair. The GOLD result, where Gemma-3-1B as a student of a Qwen teacher scored only 0.063 similarity, suggests very little benefit beyond SFT. This is an inference: the pair direction and models differ.
- **Rule of thumb.**
  1. If any acceptable teacher shares the student's tokenizer, use it for logit and on-policy stages. Same-tokenizer recovery is above 80% vs about 60% cross-tokenizer in the GOLD study.
  2. Use cross-family frontier models (DeepSeek, Kimi, GLM, gpt-oss, Gemma) only as **text generators** for SFT data.
  3. Use GOLD only when the best on-policy teacher is cross-family and tokenizer similarity is reasonably high (Llama-like 0.64, not Gemma-like 0.063).
- **Qwen-family tokenizer split (important, 2026).** The 151k-vocab family covers Qwen2.5, Qwen3, QwQ-32B, R1-Distill-Qwen-* and presumably Qwen3-Coder. The 248k-vocab family covers Qwen3.5 and Qwen3.6.
  - A student on DeepSeek-R1-Distill-Qwen-1.5B (Qwen2.5 lineage) is same-tokenizer with QwQ-32B, Qwen3-32B and Qwen3-235B-A22B.
  - It is cross-tokenizer with Qwen3.5/3.6 teachers, though probably with high token-content overlap. GOLD's token-content mapping is designed for exactly this case.
  - DeepSeek's distill checkpoints re-map special tokens. Before plain GKD, diff the tokenizer files; GOLD handles "same token, different ID" natively.

### Gaps
- No published cross-tokenizer distillation result on **long-CoT competitive programming** was found. ALM covered math (GSM8K/MATH) and general benchmarks; GOLD covered Countdown and a small Codeforces/IFEval experiment.
- MultiLevelOT's direct comparisons to DSKD/ULD were not extracted.
- I did not verify that the Qwen2.5 and Qwen3 regular-token vocabularies are byte-identical. It is likely, since both are "Qwen's tokenizer", 151,669 entries in Qwen3. Verify by diffing tokenizer.json.
- The overlap between the 248k Qwen3.5 vocabulary and the 151k Qwen3 vocabulary was not measured.

---

## 3. Hidden-state / attention / feature distillation for decoder LLMs; Low-Rank Clone; weight selection / inheritance ("Initializing Models with Larger Ones"), subcloning, Inheritune, learngene; model growth (LiGO)

### Takeaway
Using a bigger model's weights to initialize a smaller one clearly saves pretraining tokens. LRC claims to match Qwen3-1.7B with 20B vs 36T tokens, and Minitron/Sheared-LLaMA show similar savings (Section 4). However, none of these methods report competitive-programming or long-CoT results. For a team starting from an already strong 1.5B Qwen reasoning checkpoint, they matter mainly if a **custom, smaller phone architecture** (e.g., about 1B) is needed. Feature/hidden-state losses are optional extras with mixed evidence; logit KD is the workhorse.

### Cited Findings
**Low-Rank Clone (LRC; NeurIPS 2025 Spotlight)** **[CODE]**
- LRC trains low-rank projection matrices that (a) compress teacher weights into student weights ("soft pruning") and (b) "activation clone" by aligning student activations, including FFN signals, with the teacher's. No separate alignment modules are needed. The paper claims it matches or surpasses SOTA models trained on trillions of tokens using only 20B tokens (over 1,000x efficiency). — [LRC arXiv](https://arxiv.org/abs/2505.12781); [code](https://github.com/CURRENTF/LowRankClone)
- **LRC models** (averages are zero-shot over ARC-E/C, LogiQA, CSQA, PIQA, WinoGrande, BoolQ, SciQ, MMLU). — [LRC paper v5](https://arxiv.org/html/2505.12781v5)

  | Model | Teacher | Tokens | Average | Baselines |
  |---|---|---|---|---|
  | LRC-1.5B | Llama-3.2-3B-Instruct | 10B | 63.48 | |
  | LRC-1.7B | Qwen2.5-3B-Instruct | 20B | 64.98 | Qwen3-1.7B 63.17 (36T); SmolLM2-1.7B 60.50 (11T); InternLM2-1.8B 62.60 (2T); MiniCPM-1.2B 60.42 (1T) |
  | LRC-4B | Qwen2.5-7B-Instruct | 18B (text says 10B) | 70.30 | Qwen3-4B 70.29 (36T); Gemma3-4B 69.43 (4T); Minitron-4B 68.31 (94B) |

- **LRC loss and limits.** The loss is L_KL + L_LM + α·L_clone. The clone term is an MSE on attention/FFN outputs plus q, k, v, up and gate intermediates. Only the projections and RMSNorm weights are trained (KL temperature 40). There are **no reasoning, math or code benchmarks**. The student keeps the teacher's FFN intermediate dimension, and the ceiling at larger scale is unexplored. — [LRC paper v5](https://arxiv.org/html/2505.12781v5)

**Weight selection / subcloning / inheritance**
- **Weight selection ("Initializing Models with Larger Ones", ICLR 2024)** initializes a smaller model by selecting a subset of a pretrained larger model's weights (layer selection, component mapping, element selection). It is evaluated mainly on vision models and is compatible with KD. **[CODE]** github.com/OscarXZQ/weight-selection. — [Xu et al.](https://arxiv.org/abs/2311.18823)
- **Weight subcloning (Apple)** directly initializes a smaller transformer from a larger pretrained one by removing neurons ranked by importance and removing blocks, then rescaling. It reports about 4x faster training for ViTs and next-token LMs. **[NO CODE CONFIRMED]** — [Samragh et al.](https://arxiv.org/abs/2312.09299)
- **Inheritune (TMLR 02/2026)** inherits the early layers of a larger pretrained LM (zero-shot initialization), then retrains and progressively grows the student. Motivation: deeper layers degenerate into "lazy layers" whose attention collapses to a single column. A 16-layer GPT-2-medium variant matches the 24-layer model. **[CODE]** github.com/sanyalsunny111/LLM-Inheritune. A third-party summary's "89% of larger-model accuracy with 0.1% data" is not confirmed from the paper. — [Inheritune arXiv](https://arxiv.org/abs/2404.08634v3); [PDF v4](https://arxiv.org/pdf/2404.08634v4)
- **Learngene** extracts a compact "gene" (a few condensed layers) from an ancestry model to initialize descendant models of varying sizes. It is vision-centric; no LLM reasoning results were found. — [Learngene](https://arxiv.org/abs/2305.02279)

**Model growth (small → large; the opposite direction)**
- **LiGO (ICLR 2023)** learns linear operators that map a smaller pretrained model's parameters into a larger model's initialization, saving up to about 50% of from-scratch training compute (BERT/RoBERTa/GPT-2/ViT). **[CODE]** github.com/VITA-Group/LiGO. — [LiGO](https://arxiv.org/abs/2303.00980)

**Hidden-state / feature losses in practice**
- NVIDIA's Llama-3.1-Minitron / MN-Minitron recipe used **logit-only forward KL**: no hidden-state loss, and the LM cross-entropy is ignored. — [Minitron in practice](https://arxiv.org/html/2408.11796)
- In ALM, adding a hidden-state loss slightly **hurt** subword-to-subword transfer. — [ALM paper](https://arxiv.org/html/2503.20083)
- DistillKit offers hidden-state alignment losses (MSE/cosine) alongside KL/JSD/TVD, but no public numbers were found in this session. — [DistillKit](https://github.com/arcee-ai/DistillKit)

### Inferences
- **Relevance ranking for the team.**
  1. Logit or on-policy KD (Sections 1 and 6) is highest value.
  2. Prune-then-distill or LRC is relevant only if they want a new, smaller architecture than 1.5B, or a 1.5B carved from a better code teacher (e.g., Qwen3-4B or Qwen3.5-4B).
  3. Weight selection, subcloning and learngene are mostly vision-era or pretraining-era methods without reasoning evidence.
  4. LiGO and growth point the wrong way unless they want to grow a 0.5B into a 1.5B.
- **An LRC-style approach could plausibly be pointed at a code/reasoning teacher** (e.g., clone Qwen3-4B into about 1.5B, then do long-CoT SFT and on-policy KD). This is untested. LRC's only evaluations are commonsense/MMLU, and it keeps the teacher's FFN width, which limits how small the student can get.
- Feature-matching losses require dimension projections between different widths. For a pruned-from-teacher student (same width family), Minitron shows logit-only KD is sufficient, so feature losses are low priority.

### Gaps
- No LRC, weight-selection or Inheritune results on GSM8K/MATH/LCB/Codeforces were found.
- Exact weight-selection numbers (vision) were not extracted.
- Whether LRC's released checkpoints include a Qwen3-based clone was not checked.

---

## 4. Prune-then-distill pipelines: NVIDIA Minitron (Llama-3.1-Minitron, MN-Minitron, Nemotron-Nano), Sheared-LLaMA, Llama 3.2 1B/3B, Gemma 2/3, DistilQwen: size ratios, token budgets, recovered accuracy

### Takeaway
Structured pruning from a 1.5–2x larger parent, followed by logit KD, recovers most quality with 50–400B tokens. That is 40–150x fewer tokens than training the parent (Minitron). Width pruning beats depth pruning at equal size. "Teacher correction" (briefly fine-tuning the teacher on the distillation corpus) matters. Every major small open model family (Llama 3.2, Gemma 2/3, Qwen3 small models, Nemotron Nano) is built this way. For the team this is a pretraining-scale recipe (tens of billions of tokens), so prefer it only if a custom-size phone model is required.

### Cited Findings
- **Minitron (original).** Pruned Nemotron-4 15B to 8B and 4B, then retrained with distillation. Up to 40x fewer training tokens per model than from scratch, 1.8x compute savings for the whole family, and up to 16% better MMLU than training from scratch. — [Minitron](https://arxiv.org/abs/2407.14679)
- **Llama-3.1-Minitron / MN-Minitron ("Minitron approach in practice").** — [arXiv HTML](https://arxiv.org/html/2408.11796)
  - Llama 3.1 8B → 4B (depth or width), distilled on 94B tokens. Mistral NeMo 12B → 8B, distilled on 380B tokens.
  - Teacher correction: about 100B tokens of teacher fine-tuning on the distillation set, reducing LM validation loss by over 6%; it can run in parallel with distillation.
  - Logit-only forward-KL loss. Ran on 32 DGX H100 nodes.
- **Minitron-in-practice results.** — [arXiv HTML](https://arxiv.org/html/2408.11796)
  - Base MMLU (5-shot): teacher Llama 3.1 8B 65.3; 4B-Width 60.5; 4B-Depth 58.7; MN-Minitron-8B 69.5 (teacher Mistral NeMo 12B 69.0).
  - Base GSM8K: 4B-Depth 16.8, 4B-Width 41.2, teacher 48.6. Depth pruning hurt math reasoning much more.
  - MN-Minitron-8B beat its teacher on GSM8K (55.7 → 58.5) and HumanEval (23.8 → 36.2).
  - Aligned 4B models beat similar-size models except on HumanEval/MBPP.
  - Width pruning (keep attention heads, prune MLP and embedding channels) beat depth pruning. For depth pruning, contiguous layer removal worked better.
  - Speedups: about 2.7x (depth-pruned 4B) and about 1.7–1.8x (width-pruned 4B) over Llama 3.1 8B.
- **NVIDIA Nemotron Nano 2.** Nemotron-Nano-9B-v2 is pruned (Minitron-style) from a 12B hybrid Mamba-Transformer base pretrained on 20T tokens in FP8. It reports up to 6x reasoning throughput vs Qwen3-8B and 128k context on one A10G. — [Nemotron Nano 2 report](https://arxiv.org/html/2508.14444v2); [hyper.ai summary](https://hyper.ai/papers/2508.14444)
  - A secondary summary gives the staged KD budget: depth-prune KD about 60B tokens, then width-prune KD of 50B (8k context) + 25B (49k) + 1B (262k), with a 70/30 post-training/pretraining data blend for reasoning distillation. Treat as unverified. — [Emergent Mind summary](https://api.emergentmind.com/papers/2508.14444)
- **Sheared-LLaMA (ICLR 2024).** Targeted structured pruning plus dynamic batch loading turned LLaMA2-7B into 1.3B and 2.7B models with 50B tokens, about 3% of from-scratch compute. **[CODE]** princeton-nlp/LLM-Shearing. **[SUPERSEDED in quality by Minitron/LRC-era models]**: LRC reports Sheared-Llama-2.7B at 52.55 average vs LRC-4B 70.30. — [Sheared-LLaMA](https://arxiv.org/abs/2310.06694); [LRC comparison](https://arxiv.org/html/2505.12781v5)
- **Llama 3.2 1B/3B.** "Structured pruning in a single shot manner from the Llama 3.1 8B". Logits from Llama 3.1 8B and 70B were "used as token-level targets" during pretraining, with KD after pruning to recover performance. Post-training used SFT, rejection sampling and DPO rounds. Day-one Qualcomm/MediaTek support, optimized for Arm. — [Meta Llama 3.2 blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- **Gemma 2 and 3.** Gemma 2's 2B and 9B were trained with KD instead of next-token prediction. — [Gemma 2 report](https://arxiv.org/abs/2408.00118). All Gemma 3 models are trained with KD (256 sampled logits per token, as in Section 1), and post-training distils from a large instruction-tuned teacher before RL. — [Gemma 3 report](https://arxiv.org/html/2503.19786)
- **Qwen3 small models** come from strong-to-weak distillation of Qwen3-32B or 235B-A22B (Section 1). — [Qwen3 report](https://arxiv.org/html/2505.09388). One blog claims the **Qwen3.5 small series (0.8B/2B/4B/9B, Mar 2026)** was distilled from the 397B flagship, and that the Qwen team found distillation more effective than direct RL at that scale. This is unverified; the official model card does not mention it. — [logdew blog](https://logdew.pages.dev/en/posts/qwen35-small-series-local-ai-revolution/); [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)
- **DistilQwen / OmniThought (Alibaba PAI).** OmniThought holds about 2M CoT traces from multiple large reasoning models, each annotated with a Reasoning Verbosity score (RV, 0–9) and a Cognitive Difficulty score (CD). The premise is that "overly complicated reasoning steps... may not be suitable for... smaller models". (ACL 2026) — [OmniThought arXiv](https://arxiv.org/html/2505.10937v1); [ACL 2026 PDF](https://aclanthology.org/2026.acl-long.382.pdf)
  - DistilQwen-ThoughtX-7B (vendor-reported): LCB 56.8 vs 48.4 for R1-Distill-Qwen-7B; AIME24 56.7 vs 57.3. DistilQwen-ThoughtY is Qwen3-based with a DeepSeek-R1-0528 teacher. Recipes are in the EasyDistill toolkit. **[CODE]** (EasyDistill; only a mirror was reached) — [DistilQwen-ThoughtX card (proxy)](https://huggingface-proxy-6ci.pages.dev/alibaba-pai/DistilQwen-ThoughtX-7B); [EasyDistill mirror](https://code.rizlum.com/lphatnguyen/distillation)
- **Tooling.** Arcee DistillKit (online/offline logit KD, hidden-state losses) — [GitHub](https://github.com/arcee-ai/DistillKit). The torchtune KD recipe is unmaintained — [README](https://raw.githubusercontent.com/meta-pytorch/torchtune/main/README.md).

### Inferences
- **Parent:student ratios in successful pipelines** are about 1.5–2x (Minitron 12B→8B, 8B→4B; Nemotron 12B→9B; LRC 3B→1.5–1.7B). Larger jumps (Sheared 7B→1.3B; Llama 8B→1B) also used KD from larger teachers (70B logits for Llama 3.2).
- For a ~1.5B student, a Minitron-style option would be width-pruning Qwen3-4B (151k vocab) or Qwen3.5-4B (248k vocab) to about 2B. Then:
  1. Teacher-correct the parent on a code/reasoning-heavy corpus.
  2. Run logit KD on roughly 50–100B tokens.
  3. Do reasoning SFT and on-policy KD.

  Since Qwen already ships 1.7B/2B models distilled from far larger teachers on trillions of tokens, this is only worth it for (a) a sub-1.5B phone model, or (b) a paper contribution about pruning a reasoning model. These are inferences from the token budgets above.
- **Depth pruning hurts math much more than MMLU** (GSM8K 16.8 vs 41.2). Avoid aggressive layer dropping for a reasoning student.
- **OmniThought's RV/CD annotation directly targets the "yapping" failure.** Score traces for verbosity and filter or mix toward lower-verbosity traces for a 1.5B student (Section 6).

### Gaps
- Exact Llama 3.2 token counts were not on the Meta blog page fetched. The widely quoted "up to 9T tokens" is not confirmed here.
- Nemotron Nano 2's per-stage token budgets come from a secondary summary.
- No prune-then-distill study specifically on a **reasoning** (long-CoT) parent → 1–2B student with LCB numbers was found.
- DistilQwen2.5-R1 small-model (1.5B/3B) numbers were not extracted.

---

## 5. Hypernetworks and weight generation (Text-to-LoRA, Drag-and-Drop LLMs, other LoRA generators); model merging (TIES, DARE, soups, evolutionary merge). Could a large model generate or tune topic-specific LoRAs for a small coder?

### Takeaway
Prompt-to-LoRA hypernetworks (T2L, DnD) are real and have released code. They generalize zero-shot to unseen tasks at 2B–8B, but there is no evidence for long-CoT competitive programming at 1.5B. Model merging for long-to-short reasoning **fails to transfer long-CoT ability at 1.5B**. The practical, phone-proven version of "topic adapters" is to train each topic LoRA by distillation from the big teacher, then route and hot-swap them, as Apple does with rank-16 adapters on its ~3B on-device model. A hypernetwork is a research extension on top.

### Cited Findings
- **Text-to-LoRA (T2L, ICML 2025; Sakana AI).** A hypernetwork builds a LoRA from a natural-language task description in one forward pass. It was trained on 9 pretrained LoRA adapters (e.g., GSM8K, ARC). Reconstructed LoRAs match the task-specific adapters, it zero-shot generalizes to unseen tasks, and it can compress hundreds of LoRAs. **[CODE]** — [T2L arXiv](https://arxiv.org/abs/2506.06105); [code](https://github.com/SakanaAI/text-to-lora)
- **Drag-and-Drop LLMs (DnD).** A text encoder turns a batch of task prompts into condition embeddings, and a cascaded hyper-convolutional decoder outputs a full set of LoRA matrices. It is trained on many prompt-checkpoint pairs. Claims: up to 12,000x lower overhead than full fine-tuning, and average gains of up to 30% over the strongest training LoRAs on unseen common-sense, math, coding and multimodal benchmarks. Preprint, June 2025. **[CODE]** via project site (repository not separately verified). — [DnD arXiv](https://arxiv.org/abs/2506.16406); [project](https://jerryliang24.github.io/DnD)
- **Transformer² (Sakana).** Self-adaptive LLMs: Singular Value Fine-tuning trains "expert vectors", and a two-pass inference first identifies the task and then mixes experts. Conceptually this is a router plus topic adapters. — [Transformer²](https://arxiv.org/abs/2501.06252)
- **LoraHub** composes LoRA modules trained on different tasks to adapt to unseen tasks, gradient-free from a few examples. — [LoraHub](https://arxiv.org/abs/2307.13269)
- **Apple on-device precedent.** Adapters are small weight sets on a shared base, loaded and swapped at runtime for task specialization. For the ~3B on-device model a rank-16 adapter is tens of MB. The base uses mixed 2/4-bit quantization (reported average 3.7 bits per weight in the search summary of Apple's post). Accuracy-recovery adapters on the quantized base serve as the starting point for feature adapters. — [Apple ML Research](https://machinelearning.apple.com/research/introducing-apple-foundation-models); [Trail of Bits analysis](https://blog.trailofbits.com/2024/06/14/understanding-apples-on-device-and-server-foundations-model-release/)
- **Merging methods (all [CODE] via mergekit):**
  - Model soups: averaging fine-tuned weights. — [Wortsman et al.](https://arxiv.org/abs/2203.05482)
  - Task arithmetic. — [Ilharco et al.](https://arxiv.org/abs/2212.04089)
  - TIES: trim small deltas, resolve sign conflicts, merge only agreeing parameters. — [Yadav et al.](https://arxiv.org/abs/2306.01708)
  - DARE: drop delta parameters at rate p and rescale by 1/(1−p). — [Yu et al.](https://arxiv.org/abs/2311.03099)
  - Evolutionary model merge: evolutionary search over parameter-space and data-flow merges. — [Akiba et al.](https://arxiv.org/abs/2403.13187)
  - mergekit toolkit. — [mergekit](https://arxiv.org/abs/2403.13257)
- **Long-to-short via merging (System-1 + System-2 models).** — [Wu et al.](https://arxiv.org/abs/2503.20641); [paper HTML](https://arxiv.org/html/2503.20641v2); [code](https://github.com/hahahawu/Long-to-Short-via-Model-Merging)
  - **7B** (Qwen2.5-Math-7B + R1-Distill-Qwen-7B; baseline 54.5 average): TIES 54.8 with 53.0% shorter outputs; Sens-Merging 56.4 with −49.8% length; DARE only 47.3; Average Merging 48.6.
  - **1.5B** (GSM8K / MATH500 / AIME24; R1-1.5B baseline 76.8 / 69.6 / 20.0): TA 74.5 / 62.6 / 10.0; TIES 75.9 / 64.2 / 10.0; Average Merging 26.6 / 12.2 / 0.0; Sens-Merging 81.3 / 69.8 / 16.7.
  - The authors conclude that "smaller models struggle to learn long CoT reasoning ability through model merging" and that the learned reflections are "false reflections".
  - At 14B/32B, length reductions are smaller.

### Inferences
- **"Could a large model generate or tune topic-specific LoRAs for a small coder?"** Three tiers, in order of evidence:
  1. **Distilled topic LoRAs (proven pattern).** For each algorithm family (DP, graphs, greedy, math/number theory, strings, data structures, ...), run on-policy distillation from the same-tokenizer teacher on topic-specific prompts into a LoRA on the shared student. Serve with a router and runtime adapter swap; Apple's rank-16 adapters show tens of MB per adapter is feasible on phones. Here the large model "tunes" the adapters as the distillation teacher.
  2. **Hypernetwork-generated LoRAs (T2L/DnD).** Train a generator on the library from tier 1 (prompt or problem statement → LoRA). This is promising as a paper contribution, but no published evidence covers long-CoT code at 1.5B, and T2L's training set was 9 LoRAs.
  3. **Merging topic LoRAs, or long-to-short merging.** Weak at 1.5B per Wu et al. Not recommended except as a cheap baseline.
- **Risk.** Splitting capacity across topic adapters may not beat one well-distilled full model, because competitive problems often mix topics and the router can misclassify. A fair ablation is single full-FT model vs router plus topic LoRAs at equal total training compute.

### Gaps
- T2L's and DnD's base-model sizes and per-benchmark numbers (especially DnD's "coding" results) were not extracted.
- No study was found that routes per-topic LoRAs for competitive programming, or that measures topic-router accuracy on Codeforces tags.
- Apple's exact bits-per-weight figure was taken from a search summary of its post; the primary page was not re-opened.

---

## 6. Reasoning distillation for small students: learnability gap and Mix Distillation, long vs short CoT, LIMO / s1, data quality vs quantity, teacher choice, distillation scaling laws, teacher hacking, and how the strongest open ~1.5B models are built

### Takeaway
Small models (≤3B) are hurt by **small** amounts of long-CoT data: Long CoT Degradation causes losses of up to 75% with 8k examples, and in the 7.5k-problem MATH study, long CoT scored 7.1 points below short CoT at 1.5B. They are **helped** by massive verified long-CoT corpora (580K traces give rStar-Coder-1.5B 40.1 LCB; 5M traces give OpenReasoning-Nemotron-1.5B 28.6 LCB v6) and by mixing in short traces. The team's 12k-sample LoRA Stage 1 sits exactly in the "valley". "Less is more" (LIMO, s1) holds for 32B-class models, not 1.5B. The best teacher is not necessarily the strongest model (QwQ-32B beat DeepSeek-R1 as a teacher), and capacity-gap and teacher-hacking effects argue for moderate teachers and on-policy data.

### Cited Findings
**Learnability gap and Mix Distillation**
- Students ≤3B "don't consistently benefit" from long CoT or larger teachers ("Small Model Learnability Gap"). Mixing long and short CoT, or large and small teachers ("Mix Distillation"), works better than either alone. (arXiv v3, Nov 2025; project page small-model-gap.github.io) — [Li et al.](https://arxiv.org/abs/2502.12143)
- **Setup.** Trained on 7.5k MATH problems; long-CoT teacher QwQ-32B-Preview, short-CoT teacher Qwen2.5-32B-Instruct; full-parameter SFT for models under 14B (lr 1e-5, 2 epochs); 16k max generation. — [Li et al. HTML](https://arxiv.org/html/2502.12143v3)
- **Long-CoT gap (ΔLong = long-CoT score minus short-CoT score, five-benchmark math average).** — [Li et al. HTML](https://arxiv.org/html/2502.12143v3)
  - Negative (short CoT better): Qwen2.5-0.5B −4.7, Qwen2.5-1.5B −7.1 (27.0 vs 34.2), Qwen2.5-3B −3.1, Llama-3.2-1B −3.7, Llama-3.2-3B −0.6.
  - Positive (long CoT better): Qwen2.5-7B +1.7, Qwen2.5-14B +4.9, Qwen2.5-32B +13.7.
- **Large-teacher gap (Qwen2.5-72B-Instruct vs Qwen2.5-3B-Instruct as teacher).** — [Li et al. HTML](https://arxiv.org/html/2502.12143v3)
  - At 0.5B: −3.5. At 1.5B: −0.8. At 3B: +0.3. At 7B: +6.6.
- **Mix Distillation (Qwen2.5-3B-Instruct, average).** — [Li et al. HTML](https://arxiv.org/html/2502.12143v3)
  - Mix-Long (long:short = 1:4, α=0.2 best): 45.9, vs 40.3 long-only and 43.4 short-only. MATH rose 56.2 → 64.7.
  - Mix-Large: 45.8. DeepSeek-R1-32B long CoT alone: 33.5.
  - Average response length: long 3,385 tokens, short 576, Mix-Long 1,249.

**Long CoT Degradation**
- ≤3B models trained on limited long-CoT data degrade: with 8k examples, some lose "up to 75% of their original performance". For some very small models, even 220k examples did not recover the pre-fine-tuning level. — [Luo et al., "Through the Valley"](https://arxiv.org/abs/2506.07712)
  - Models tested: Qwen2.5, LLaMA3 and Gemma3 families.
  - Cause: error accumulation in longer outputs.
  - The degradation can hurt downstream RL and is alleviated by sufficiently scaled SFT.

**"Less is more" results (large students)**
- LIMO reports 63.3% AIME24 and 95.6% MATH500 using about 1% of the training data of prior approaches. Its hypothesis: when pretraining has encoded the domain knowledge, a few "cognitive template" demonstrations suffice. — [LIMO](https://arxiv.org/abs/2502.03387)
- s1: 1,000 curated questions (s1K) plus "budget forcing" (test-time control of thinking length). s1-32B exceeds o1-preview on competition math by up to 27% (MATH, AIME24), and budget forcing extrapolates AIME24 from 50% to 57%. — [s1](https://arxiv.org/abs/2501.19393)

**Data scale and recipe for small reasoning models**
- **Phi-4-mini-reasoning (3.8B)** uses a four-stage recipe: (1) mid-training on large-scale distilled long-CoT, (2) SFT on high-quality long-CoT, (3) Rollout DPO, (4) RL with verifiable rewards. It beats R1-Distill-Qwen-7B by 3.2 and R1-Distill-Llama-8B by 7.7 points on MATH-500. — [Phi-4-mini-reasoning](https://arxiv.org/abs/2504.21233)
- **OpenThoughts (ICLR 2026).** — [OpenThoughts arXiv](https://arxiv.org/abs/2506.04178); [PDF](https://arxiv.org/pdf/2506.04178); [ICLR 2026 oral](https://iclr.cc/virtual/2026/oral/10011247)
  - More than 1,000 controlled data-recipe experiments.
  - OpenThoughts3-1.2M with a QwQ-32B teacher gives OpenThoughts3-7B: AIME25 53%, LCB (06/24–01/25) 51%, GPQA-D 54%, which is +15.3 / +17.2 / +20.5 points over R1-Distill-Qwen-7B.
  - **QwQ-32B was a better teacher than DeepSeek-R1** (+1.9% code, +2.6% math), even though R1 outscores QwQ by 9% on CodeElo, 8% on GPQA-D and 23% on JEEBench.
- **rStar-Coder (competitive programming).** — [rStar-Coder](https://arxiv.org/html/2505.21297)
  - Data: 418,314 verified problems (37,754 expert seeds + 380,560 synthetic) and about 580K solutions. QwQ-32B sampled 16 long-CoT solutions per problem; for seed problems only test-passing solutions were kept (all kept for very hard problems where QwQ failed).
  - Tests: GPT-4o wrote input generators and validators (CYaRon). Outputs were labelled by mutual verification: at least 50 inputs and ≥60% agreement, or 40% for CF >1600. Labelling accuracy was 96.8% vs 12.7% for direct GPT-4o.
  - Training: 6 epochs, 16k max length, batch 96, lr 4e-5, 8 MI300X for 1.5B.
- **rStar-Coder results.** — [rStar-Coder](https://arxiv.org/html/2505.21297)
  - **rStar-Coder-1.5B: LCB 40.1, HumanEval+ 81.7, MBPP+ 60.8.** For comparison, R1-Distill-Qwen-1.5B scores 16.9 / 61.8 / 48.5, and the Qwen2.5-Coder-1.5B base is listed at 6.5 LCB.
  - 7B: 57.3 (vs R1-Distill-7B 37.6). 14B: 62.5.
  - OCR-Qwen-7B/14B-Instruct: 51.3 / 59.4.
- **OpenCodeReasoning-II:** about 2.5M question-solution-critique triples over about 35K questions. Two-stage fine-tuning (generation, then joint generation + critique) of Qwen2.5-Instruct models; self-critique gives test-time scaling gains; LCB extended to C++. — [OCR-II](https://arxiv.org/abs/2507.09075)

**Strongest open ~1.5B reasoning/code models and how they were made**
- **OpenReasoning-Nemotron-1.5B (Jul 16, 2025; CC-BY-4.0).**
  - Built on Qwen2.5-1.5B with **SFT only** on about 5M DeepSeek-R1-0528 traces: math (OpenMathReasoning), code (OpenCodeReasoning I+II) and science, plus instruction-following and tool data.
  - Results: AIME24 55.5, AIME25 45.6, HMMT-Feb25 31.5, **LCB v6 28.6**, GPQA 31.6, MMLU-Pro 47.5.
  - GenSelect / majority@64: AIME24 76.7, AIME25 70.0. — [model card](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B)
- **DeepCoder-1.5B-Preview.** RL (GRPO+, iterative context lengthening, 32K training context) from R1-Distill-Qwen-1.5B on 24K problem-test pairs (TACO-Verified, PrimeIntellect SYNTHETIC-1, LCB v5). LCB 16.9 → 25.1; Codeforces 615 → 963 (28.5th percentile); HumanEval+ 58.3 → 73.0. — [model card](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)
- **Nemotron-Research-Reasoning-Qwen-1.5B (ProRL, prolonged RL from R1-Distill-Qwen-1.5B).** A mirrored card table gives Codeforces pass@1 34.50 vs 14.13 for the base (DeepCoder-1.5B 21.70), and LCB 23.81 vs 16.80. The paper headline is about +14.4% average code pass@1 (a later version says 13.9%). Unverified third-party mirror. — [mirror card](https://hf-mirror.492719920.workers.dev/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B); [ProRL paper](https://arxiv.org/html/2505.24864v1)
- **Qwen3-1.7B** was built by Qwen3's off-policy then on-policy strong-to-weak distillation (Section 1). — [Qwen3 report](https://arxiv.org/html/2505.09388)
- **Qwen3.5-2B (Feb/Mar 2026; Apache-2.0).** — [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B); [Artificial Analysis](https://artificialanalysis.ai/es/articles/qwen3-5-small-models)
  - Architecture: 24-layer hybrid, 6 × (3 × Gated DeltaNet + 1 × Gated Attention), 248k vocabulary, 262k context, vision encoder, thinking and non-thinking modes.
  - The card warns the 2B is "more prone to entering thinking loops".
  - Results: MMLU-Pro 66.5 (thinking), GPQA 51.6, HMMT-Feb25 22.9; no LCB listed.
  - Artificial Analysis Intelligence Index: 2B 16, 4B 27, 9B 32. Under 2GB at 4-bit.

**Teacher choice (2025–2026 open models)**
- **Strong-model lineup** (Qwen family, 248k vocabulary).
  - Qwen3.6-27B (dense hybrid, April 2026): LCB v6 83.9, SWE-bench Verified 77.2. — [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B)
  - Qwen3.6-35B-A3B (MoE, April 16, 2026). — [secondary coverage](https://www.noze.it/en/insights/qwen-3-6/)
  - Qwen3.5-397B-A17B (flagship). — [Qwen3.5 small coverage](https://artificialanalysis.ai/es/articles/qwen3-5-small-models)
- **DeepSeek V4** (late April 2026; Pro 1.6T MoE, Flash 284B; 1M context). Reported V4-Pro LCB 93.5 and Codeforces 3206, with Kimi K2.6 at 89.6 LCB. All figures come from **secondary** sources; the license is reported inconsistently (MIT vs Apache). — [DataNorth](https://datanorth.ai/news/deepseek-releases-v4-open-source-1m-context); [HandyAI](https://handyai.substack.com/p/model-drop-deepseek-v4)
- **Teachers used by the strongest small-model pipelines:** DeepSeek-R1-0528 (OpenReasoning-Nemotron; DistilQwen-ThoughtY), QwQ-32B (OpenThoughts3; rStar-Coder), Qwen3-32B/235B (Qwen3 small models). — [OR-Nemotron card](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B); [OpenThoughts](https://arxiv.org/pdf/2506.04178); [rStar-Coder](https://arxiv.org/html/2505.21297); [Qwen3](https://arxiv.org/html/2505.09388)

**Distillation scaling laws and teacher hacking**
- **Busbridge et al. (ICML 2025).** — [arXiv](https://arxiv.org/abs/2502.08606); [HTML](https://arxiv.org/html/2502.08606v2); [Apple ML](https://machinelearning.apple.com/research/distillation-scaling-laws)
  - Student loss can be predicted from teacher cross-entropy, student size and distillation tokens.
  - Distillation beats supervised learning when a teacher already exists or many students are planned, but only up to a compute level that grows with student size. With one student and a teacher that must be trained, supervised learning is generally preferable.
  - **Capacity gap:** a stronger teacher can yield a worse student, because the gap is in learning capacity rather than raw size.
  - Scope: fitted on 143M–12.6B models with up to 512B tokens, on students with cross-entropy above 2.3.
- **Teacher hacking.** The student over-optimizes toward an imperfect teacher, and true quality degrades. It occurs with **fixed offline datasets**. Online data generation mitigates it, with data diversity the key factor. It can be detected by deviation from polynomial convergence laws. (arXiv preprint) — [Tiapkin et al.](https://arxiv.org/abs/2502.02671)

### Inferences
- **Diagnosis of the team's Stage 1 (12k long-CoT, LoRA, about 0–1% Pass@1).** This matches Long CoT Degradation (8k examples → up to −75%) and the learnability gap (−7.1 at 1.5B). It is probably compounded by LoRA's gap vs full fine-tuning (13% in TM's experiment) and by a starting checkpoint already tuned on 800k R1 traces.
  - The fix is not shorter data alone. Either scale verified long-CoT to hundreds of thousands of examples with full fine-tuning (rStar-Coder-1.5B: 580K, 6 epochs → 40.1 LCB), or mix about 20% short or structured traces (Mix-Long α=0.2), or both.
  - The Stage-2 "blueprint" gain (8%) is consistent with the short-CoT advantage for 1.5B students.
- **Reconciling "short is better" with the long-CoT successes:** the two lines of evidence differ by data scale (7.5k problems vs 0.58–5M traces). At scale, long CoT works for 1.5B students, and the short-CoT advantage is a small-data phenomenon. This is an inference consistent with "Through the Valley".
- **Teacher choice for a 1.5B student.**
  - Prefer a moderate, same-tokenizer teacher for logit and on-policy stages: QwQ-32B or Qwen3-8B/14B/32B for the 151k family; Qwen3.6-27B for the 248k family.
  - Use the strongest available models (DeepSeek-R1-0528/V4, Qwen3.5-397B) mainly as **solution and test generators** for SFT, filtered by execution.
  - OpenThoughts' result (QwQ beat R1 as teacher), the capacity-gap law and the 1.5B large-teacher gap (−0.8) all argue against assuming "bigger teacher is better". Pick teachers empirically with a small student pilot.
- **Teacher hacking implies:** after offline SFT, add an on-policy stage, diversify prompts (hundreds of thousands of distinct problems from rStar-Coder or OCR-II rather than repeated epochs on 12k), and use execution-based verification as the ground truth.

### Gaps
- LIMO's exact sample count and base model were not on the fetched abstract page (commonly cited as 817 samples on Qwen2.5-32B-Instruct).
- rStar-Coder's LCB version window was not captured, so its 40.1 vs OR-Nemotron's 28.6 (v6) is not apples-to-apples.
- Whether rStar-Coder model weights (not just data) are released was not checked.
- No primary-source LCB numbers for DeepSeek V4, Kimi K2.x or GLM-5.x were obtained. No public study was found of gpt-oss-120b or Kimi as teachers for ≤2B code students.
- Qwen3.5-2B's LCB score and training (distillation vs RL) are not disclosed on its card.
- The ProRL-1.5B numbers come from a mirror.

---

## 7. Recommended distillation pipeline for a ~1.5B Qwen-family competitive-programming student (unlimited compute): teacher choice, objective, data size, stages, expected gains

### Takeaway
Recommended pipeline:
1. Large-scale verified SFT with full fine-tuning on a long+short mix: 0.5–1M execution-verified traces, about 80% long, about 20% structured/short.
2. On-policy reverse-KL distillation from a same-tokenizer Qwen teacher.
3. Optional RL with unit-test rewards and length control.
4. Quantization-aware recovery for the phone.

Public reference points suggest LCB moving from 16.9 (current base) into roughly the 30–45 range. On the team's 200-problem set this plausibly means a multiple of the current 8% Pass@1, though that mapping is uncertain.

### Cited Findings
Reference points that anchor the plan and its expected gains (all reported elsewhere above):

| Model / step | Recipe | Code result | Source |
|---|---|---|---|
| R1-Distill-Qwen-1.5B (team's base) | SFT on R1 data | LCB v5 16.9; CF 615 | [DeepCoder card](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview) |
| DeepCoder-1.5B | RL (24K problems) on that base | LCB 25.1; CF 963 | [DeepCoder card](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview) |
| ProRL-1.5B | Prolonged RL on that base | LCB 23.81 (mirror) | [mirror](https://hf-mirror.492719920.workers.dev/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B) |
| OpenReasoning-Nemotron-1.5B | SFT-only, about 5M R1-0528 traces | LCB v6 28.6 | [card](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B) |
| rStar-Coder-1.5B | SFT on about 580K QwQ-32B verified CP traces | LCB 40.1 | [rStar-Coder](https://arxiv.org/html/2505.21297) |
| Qwen3-8B, off-policy → on-policy KD | Strong-to-weak | LCB v5 42.0 → 60.3 (RL: 52.9), 1,800 vs 17,920 GPU h | [Qwen3](https://arxiv.org/html/2505.09388) |
| Qwen3-4B, CF SFT → GKD | HF/TRL | LCB 35.1 → 40.29; IFEval restored 79.5 → 82.8 | [HF H4](https://huggingfaceh4-on-policy-distillation.hf.space/) |

- On-policy distillation needs a reasonable SFT initialization; reaching +10 AIME points took about 150 steps × 64 prompts × 4 samples. — [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Mix-Long 1:4 (α=0.2) is the best long/short ratio found for a ≤3B student. — [Li et al.](https://arxiv.org/html/2502.12143v3)
- Execution-verified test generation (mutual verification, 96.8% accurate) is available in rStar-Coder; OCR-II provides 2.5M triples. — [rStar-Coder](https://arxiv.org/html/2505.21297); [OCR-II](https://arxiv.org/abs/2507.09075)
- Qwen3 exposes a thinking-budget mechanism. — [Qwen3 abstract](https://arxiv.org/abs/2505.09388). L1 (LCPO) trains a 1.5B reasoning model to respect a user-specified token budget via RL. — [L1](https://arxiv.org/abs/2503.04697)
- Qwen3.5-2B runs in under 2GB at 4-bit. Its card warns of thinking loops, and its 248k vocabulary pairs natively with Qwen3.6-27B (LCB v6 83.9). — [Artificial Analysis](https://artificialanalysis.ai/es/articles/qwen3-5-small-models); [Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B); [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B)
- Tooling with code:
  - TRL GKDTrainer (same tokenizer). — [TRL GKD](https://huggingface.co/docs/trl/gkd_trainer)
  - TRL GOLD (cross-tokenizer, experimental). — [HF H4](https://huggingfaceh4-on-policy-distillation.hf.space/)
  - Tinker cookbook distillation recipe. — [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
  - DistillKit (offline compressed logits). — [DistillKit](https://github.com/arcee-ai/DistillKit)
  - DistiLLM-2. — [code](https://github.com/jongwooko/distillm-2)
  - tokenkit/ALM. — [code](https://github.com/bminixhofer/tokenkit)

### Inferences
**The concrete recommended pipeline (synthesis; numbers are proposals, not measured results):**

**Step 0: Pick the tokenizer lineage first.** The vocabulary determines which teachers can give logits.
- **Track A (lowest risk, keeps the paper's lineage): the 151k-vocab Qwen family.**
  - Student candidates: DeepSeek-R1-Distill-Qwen-1.5B (current); OpenReasoning-Nemotron-1.5B (a stronger SFT start, LCB v6 28.6); or Qwen3-1.7B (native thinking-budget control).
  - Logit/on-policy teachers: QwQ-32B and Qwen3-8B/14B/32B (and Qwen3-235B-A22B for a size sweep).
  - Text-only solution generators: DeepSeek-R1-0528, DeepSeek V4, Qwen3.5/3.6.
- **Track B (newer and more phone-friendly): the 248k-vocab Qwen3.5 family.**
  - Student: Qwen3.5-2B. Its hybrid Gated DeltaNet keeps a mostly constant-size state over long traces, and it is under 2GB at 4-bit.
  - Same-tokenizer teachers: Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.5-397B-A17B. The Thinking Machines cookbook already runs on Qwen3.5.
  - Risks: the card's "thinking loops" warning (which may be exactly the yapping failure), the unused vision encoder, and less mature tooling for on-device kernels.

**Step 1: Data (about 1–2 weeks of generation; mostly reuse).**
- Start from rStar-Coder (418K problems, about 580K verified solutions) and OCR/OCR-II (2.5M triples over about 35K questions).
- Optionally regenerate with the chosen teacher: about 4–16 samples per problem (rStar-Coder used 16), kept only if they pass execution tests.
- Decontaminate against the evaluation set by date and problem ID.
- Score each trace for length and verbosity (OmniThought-style RV). Build a **Mix-Long corpus**: about 80% verified long-CoT plus about 20% short/structured traces. The structured share can be the existing [GOAL]/[LOGIC]/[PSEUDOCODE]/[CORNER CASES] blueprint format, regenerated by a Qwen teacher so the blueprints are also on-policy-distillable later.
- Target scale: **≥500K traces for code**, optionally plus math (as OpenReasoning-Nemotron did with 5M multi-domain traces).

**Step 2: Off-policy SFT (sequence-level KD), full fine-tuning, not LoRA.**
- Settings: 16–32k context, 3–6 epochs (rStar-Coder: 6 epochs, lr 4e-5, batch 96, 16k), curriculum by length as in the current Stage 1.
- Rough cost: 600K traces × about 8K tokens ≈ 4.8B tokens per epoch. At 6·N·D ≈ 4×10¹⁹ FLOPs per epoch for 1.5B, that is tens of H100-hours per epoch excluding attention overhead (my estimate).
- **Expected:** LCB in the high-20s to about 40, from 16.9 (OR-Nemotron 28.6 v6; rStar-Coder-1.5B 40.1).

**Step 3: On-policy distillation (the key new stage).**
- Objective: per-token reverse KL (TRL GKD with lmbda=1, beta=1; or the Tinker recipe), from a **same-tokenizer** teacher on a diverse set of competitive-programming prompts. Prompts only; no labels needed.
- Run a teacher sweep: {Qwen3-8B, QwQ-32B, Qwen3-32B} for Track A, or {Qwen3.6-27B, Qwen3.6-35B-A3B} for Track B.
- Budget: start at about 150–500 steps × 64 prompts × 4 samples (TM), at a 16–32k generation cap. Student sampling dominates cost; teacher scoring is prefill-only.
- Optional loss ablations: DistiLLM-2 contrastive loss, TAID interpolation, SKD interleaving (+1–3 points at 1.5B in published code/math results).
- **Expected:** a further gain on the order of what Qwen3-8B saw (+18 LCB points), scaled down for a 1.5B capacity gap. Even about half of that would be +5–10 LCB points. Pass@k should also rise (Qwen3 saw pass@64 gains).

**Step 4: RL with verifiable rewards (optional).**
- GRPO-style RL on unit-test pass rate with iterative context lengthening (DeepCoder) or prolonged RL (ProRL), plus a length or budget penalty (L1/LCPO) to curb yapping.
- Keep a KL anchor to the distilled model, or alternate with short on-policy distillation phases; TM lists combining the two signals as open work.
- **Expected:** +5–8 LCB points, judging by DeepCoder (+8.2) and ProRL (+7.0) on the R1-Distill-1.5B base. Gains may be smaller on top of a stronger SFT/OPD model.

**Step 5: Phone packaging.**
- 4-bit quantization (GGUF/MLC), then a short **self-distillation recovery** stage: the bf16 model teaches the quantized model, or an accuracy-recovery LoRA is trained on top (the Apple pattern).
- Measure Pass@1 at the phone's realistic generation budget (e.g., 4–8k tokens), not 32k.

**Step 6 (optional, paper-novel): router plus topic adapters.**
- Distill a tiny classifier (e.g., Qwen3-0.6B or a small encoder) on teacher-generated labels per problem: algorithm topic, difficulty bucket, and the teacher's own successful reasoning length as the "thinking budget" label.
- The router selects (a) a budget for budget-conditioned generation (Qwen3-style budget / L1) and (b) optionally a topic LoRA trained by on-policy distillation on that topic's prompts, hot-swapped at runtime.
- A T2L/DnD hypernetwork trained on the topic-LoRA library is a research extension.
- Retrieval of hand-written blueprints (the team's M3) can stay as a test-time add-on. Distilling the teacher's blueprints at scale should subsume the 7-blueprint library.
- Compare against the single full-FT model at equal compute. Expect modest or uncertain gains, since there is no direct evidence.

**Ablations that would make the paper convincing:**
- LoRA vs full fine-tuning
- 12k vs 100k vs 500k traces
- long vs short vs Mix-Long
- SFT vs +OPD vs +RL
- teacher-size sweep
- Gemma-text SFT vs Qwen-teacher OPD (same- vs cross-tokenizer)
- GOLD (Gemma teacher) vs GKD (Qwen teacher)
- budget-router on vs off

**What not to prioritize:**
- Cross-tokenizer logit KD from Gemma (small gains; low overlap).
- Long-to-short model merging at 1.5B (fails).
- Re-pruning or LRC-cloning a new 1.5B (pretraining-scale cost; no reasoning evidence), unless a sub-1B phone model is a goal.
- Torchtune (unmaintained).

### Gaps
- How LCB maps onto the team's 200-problem open-r1/verifiable-coding-problems-python Pass@1 is unknown. The expected-gain ranges above are extrapolations from public 1.5B results (different LCB windows, eval settings and bases) and from an 8B on-policy result.
- No public result combines large-scale CP SFT, on-policy distillation and RL at 1.5B, so the stage gains may not add up.
- Qwen3.5-2B's code performance and susceptibility to long-trace looping after distillation are unmeasured publicly. On-device kernel support for Gated DeltaNet was not checked.
- The optimal on-policy teacher size for a 1.5B student is untested publicly; TM's preference for Qwen3-8B was for an 8B student.
