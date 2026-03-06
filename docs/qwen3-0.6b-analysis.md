# Qwen3-0.6B: Architecture Analysis & Finetuning Strategy

## 1. Architecture Overview

| Spec | Value |
|------|-------|
| Parameters | 0.6B total, **0.44B non-embedding** |
| Layers | 28 |
| Hidden size | 1,024 |
| FFN size | 3,072 (SwiGLU, 3 matrices) |
| Attention | GQA: 16 query heads, 8 KV heads |
| Head dim | 128 |
| Vocab | 151,936 (BPE, 119 languages) |
| Context | 32,768 tokens |
| Position encoding | RoPE (theta=1,000,000) |
| Normalization | RMSNorm (pre-norm) |
| Embedding | Tied (input = output) |
| Special | Dual-mode: `<think>` reasoning / `/no_think` direct |

---

## 2. Strengths

### 2.1 Extractive copying
The model's attention mechanism is well-suited for **copying verbatim text** from input to output. With 16 query heads attending over the source, it can locate and reproduce exact names, numbers, dates from the document. This is near identity mapping — the model just needs to "point" at the right tokens.

### 2.2 Short-input performance
When input is short (< 2K tokens), all 16 query heads have enough capacity to attend to every token. The model produces coherent, factually consistent output on short passages. Benchmarks show sub-1B models achieve **93-97% factual consistency** on short news articles.

### 2.3 Concise output
Small models naturally produce shorter outputs (50-70 words). For summarization, this is a feature — less room for hallucination, less redundancy.

### 2.4 Inference speed
0.6B parameters with Q4_K_M quantization → ~400MB GGUF. Runs on CPU or 2GB GPU. With Metal on M-series Mac: **~200 tokens/sec** generation. Enables real-time serving without expensive hardware.

### 2.5 Dual-mode thinking
`/no_think` mode skips reasoning overhead, giving faster and more predictable outputs for extractive tasks. `/think` mode available when reasoning is needed but costs extra tokens.

---

## 3. Weaknesses

### 3.1 Primacy bias (lost-in-the-middle)

**Problem**: The model exhibits a U-shaped attention curve — it attends strongly to the **beginning** and **end** of the input, but **misses information in the middle**. Research shows that when relevant information is placed in the middle of the context, performance drops below closed-book performance (the model does worse WITH the document than WITHOUT it).

**Root cause**:
- Causal masking creates inherent bias toward early tokens
- Softmax normalization forces residual attention to concentrate somewhere — early tokens become "attention sinks"
- With only 16 query heads (8 KV heads), there aren't enough parallel channels to cover beginning, middle, AND end simultaneously

**Impact on summarization**: For a 5,000-word government report, facts in paragraphs 10-30 (the middle) are systematically underrepresented in the output. The model "forgets" the middle of the document.

### 3.2 Limited representation capacity

**Problem**: Hidden size of 1,024 means each position is represented by a 1,024-dimensional vector. This limits how many linguistic features (syntax, semantics, entities, discourse structure) can be simultaneously encoded at each position.

**Root cause**:
- FFN intermediate size is only 3,072 — limits complex feature transformations
- Tied embeddings force the same 1,024-dim space to serve both input encoding and output prediction
- 0.44B non-embedding parameters store far less world knowledge than larger models

**Impact on summarization**: The model cannot maintain a rich "mental model" of a long document. It can track ~5-10 entities before representations start colliding. Complex relationships between entities (e.g., "Agency X funded Program Y which was administered by Department Z") get lost.

### 3.3 Poor abstractive generation

**Problem**: The model struggles to **paraphrase** — it either copies verbatim or generates incoherent text. With copy mechanisms, novel word generation drops from 6.6% to 2.2% (vs. 14.8% in human summaries).

**Root cause**:
- Abstractive generation requires understanding semantics deeply enough to express the same idea in different words
- With limited hidden size, the model doesn't build deep enough semantic representations to support paraphrasing
- Training on 36T tokens gives broad coverage but shallow depth per concept at 0.6B scale

**Impact on summarization**: Summaries either read like a cut-and-paste of source sentences, or contain hallucinated/incoherent phrases when the model attempts to generate novel text.

### 3.4 Prompt sensitivity

**Problem**: Detailed/complex prompts actually **degrade** performance. Research shows Qwen2-0.5B-Instruct dropped from 93% to 78.2% factual consistency when given elaborate instructions.

**Root cause**:
- Long prompts consume attention capacity that should be spent on the document
- The model cannot simultaneously follow complex instructions AND process document content
- Instruction-following competes with content-processing for the same limited attention heads

**Impact on summarization**: We cannot compensate for model limitations by writing better prompts. More instructions = worse output.

### 3.5 Hallucination under pressure

**Problem**: When asked to generate more text than it can faithfully support, the model fills gaps with plausible-sounding but factually incorrect content. Faithfulness scores: **0.75-0.77** for 0.5B vs **0.84-0.86** for 1.5B.

**Root cause**:
- Limited knowledge capacity means the model has less "ground truth" to draw from
- When the representation of the source document is lossy (due to 3.2), the model "invents" details to maintain fluency
- Autoregressive generation amplifies small errors — one hallucinated token biases subsequent tokens

**Impact on summarization**: Summaries may contain invented statistics, wrong attributions, or conflated events — especially for less common topics or longer documents.

---

## 4. Finetuning Strategy to Address Weaknesses

### 4.1 Pipeline design: exploit strengths, avoid weaknesses

Instead of asking the model to do one hard thing (read long doc → write abstract summary), we decompose into two easy things:

```
Pass 1: extract_facts (STRENGTH: extractive copying)
  Input:  full document
  Output: bullet-point facts ("- fact1\n- fact2\n...")

Pass 2: summarize_facts (MANAGEABLE: short-input abstractive)
  Input:  only the extracted facts (NOT full document)
  Output: prose summary
```

**Why this works:**
| Weakness | How pipeline addresses it |
|----------|--------------------------|
| 3.1 Primacy bias | Pass 1 prompt says "Cover beginning, middle, and end" — SFT trains model to attend to full doc |
| 3.2 Limited representation | Pass 2 sees only ~500 tokens of facts instead of 5,000+ tokens of doc — 10x more attention per fact |
| 3.3 Poor abstractive | Pass 2 input is already distilled — combining 10 short bullet points into prose is much easier than abstracting a 5K-word report |
| 3.4 Prompt sensitivity | Both prompts are short and simple (5-6 lines). No complex instructions |
| 3.5 Hallucination | Pass 2 can only reference extracted facts — no access to full doc means no opportunity to hallucinate from lossy document representation |

### 4.2 SFT with QLoRA: what we teach the model

**Task A: Extract facts** (addresses weakness 3.1, 3.2)

The base model already knows how to copy text. SFT teaches it:
- **WHERE to look**: attend to the full document, not just the beginning
- **WHAT to extract**: important facts (names, numbers, dates, events), not filler
- **HOW to format**: one fact per bullet point, starting with "- "

Gold data: generated by Opus from `(report, reference_summary)` pairs — "extract facts from this report that support this summary". This ensures extracted facts are the RIGHT facts for producing a good summary.

**Task B: Summarize facts** (addresses weakness 3.3, 3.5)

The base model struggles with abstraction on long inputs. But summarizing 10-20 short bullet points into prose is tractable. SFT teaches it:
- **Combine** related facts into sentences
- **Use all** facts (no omission)
- **Add nothing** beyond what's in the facts (no hallucination)
- **Match target length** (~80-170 words depending on doc)

Gold data: reference summary from GovReport directly — the model learns to produce summaries that match human quality.

### 4.3 QLoRA configuration

| Param | Value | Rationale |
|-------|-------|-----------|
| Method | QLoRA (4-bit base + LoRA) | 0.6B fits in 4-bit; LoRA avoids catastrophic forgetting |
| Rank | 32 | Sufficient for 2 simple tasks; higher rank risks overfitting at 0.6B scale |
| Alpha | 32 | alpha=rank for stable training |
| Target modules | All (q,k,v,o,gate,up,down) | Full coverage for both attention patterns and FFN features |
| Learning rate | 2e-5 | Standard SFT rate; lower than GRPO to avoid instability |
| Epochs | 3 | Enough passes for the model to learn format + extraction patterns |
| Batch size | 2 × 4 accumulation = 8 effective | Balance between stability and speed |
| Max seq length | 4,096 | Fits truncated docs + completions |

### 4.4 Merge + export

After SFT:
1. **Merge** LoRA adapters into base weights → single full-precision model
2. **Export** Q4_K_M GGUF → single file for serving

No separate adapter loading at inference. The serving `model_loader.py` just loads the merged GGUF — same code, better weights.

### 4.5 Why SFT is sufficient (no RLHF/DPO needed)

For our two tasks:
- **Task A (extract)** has a near-deterministic correct answer — the right facts to copy. SFT with gold extraction data directly teaches this mapping.
- **Task B (summarize)** has a clear reference summary. SFT teaches the model to produce summaries matching human quality.

DPO/RLHF would help if we needed the model to learn **preferences** (e.g., "prefer concise over verbose"). But our pipeline already constrains output through:
- Token budget limiting (max_new_tokens scales with doc length)
- Prompt instructions ("Write a {target_words}-word summary")
- Pipeline structure (Pass 2 only sees facts, can't hallucinate from doc)

Adding DPO/RLHF would increase complexity without proportional benefit for these well-constrained tasks.

---

## 5. Expected Improvements After Finetuning

| Metric | Base model (current) | After SFT (expected) | Why |
|--------|---------------------|---------------------|-----|
| ROUGE-1 | 0.28 | 0.35-0.40 | Better fact extraction → higher unigram recall |
| ROUGE-2 | 0.15 | 0.20-0.25 | Facts align with reference → more bigram overlap |
| ROUGE-L | 0.21 | 0.28-0.33 | Coherent summary from good facts → longer matching subsequences |
| Factual consistency | ~75% | ~90%+ | Pass 2 only sees facts, no hallucination from doc |
| Coverage (middle of doc) | Poor | Good | SFT trains extraction from beginning, middle, end |

These projections are based on:
- TrueBrief findings: SFT improves faithfulness from 0.75 → 0.84 at similar scale
- The pipeline structurally eliminates the worst failure mode (hallucination from long-doc processing)
- Gold extraction data from Opus ensures the model learns to extract the RIGHT facts

---

## References

- [Qwen3 Technical Report (arXiv 2505.09388)](https://arxiv.org/html/2505.09388v1)
- [Liu et al., "Lost in the Middle" (TACL 2024)](https://aclanthology.org/2024.tacl-1.9/)
- [When Attention Sink Emerges (ICLR 2025)](https://arxiv.org/html/2410.10781v2)
- [Evaluating Small Language Models for News Summarization](https://arxiv.org/html/2502.00641v2)
- [TrueBrief: Faithful Abstractive Summarization](https://arxiv.org/html/2601.04212)
- [MobileLLM: Optimizing Sub-Billion Parameter LMs](https://arxiv.org/abs/2402.14905)
- [Pre-trained Summarization Distillation](https://arxiv.org/pdf/2010.13002)
