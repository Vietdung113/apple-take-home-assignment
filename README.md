# GovReport Summarization — Technical Report

> Building a summarization adapter for a small language model (Qwen3-0.6B), capable of summarizing U.S. government reports up to hundreds of thousands of words long, deployed on consumer-grade hardware.

> **Setup & run guide:** [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)

## Table of Contents

- [1. Data](#1-data)
  - [1.1 Dataset](#11-dataset)
  - [1.2 Statistics](#12-statistics)
  - [1.3 Distribution by Context Length](#13-distribution-by-context-length)
  - [1.4 Key Characteristics Influencing Design](#14-key-characteristics-influencing-design)
- [2. Approach](#2-approach)
  - [2.1 Overall Strategy](#21-overall-strategy)
  - [2.2 Fine-tuning: QLoRA on Qwen3-0.6B](#22-fine-tuning-qlora-on-qwen3-06b)
  - [2.3 Agentic Pipeline: Handling Documents Beyond Context Window](#23-agentic-pipeline-handling-documents-beyond-context-window)
- [3. Evaluation](#3-evaluation)
  - [3.1 Evaluation Design](#31-evaluation-design)
  - [3.2 Test Set](#32-test-set)
  - [3.3 Metrics](#33-metrics)
  - [3.4 Results](#34-results)
- [4. Error Analysis](#4-error-analysis)
  - [4.1 Error Detection Flow](#41-error-detection-flow)
  - [4.2 Analysis Results](#42-analysis-results)
  - [4.3 Observations & Improvement Directions](#43-observations--improvement-directions)
- [5. System Architecture](#5-system-architecture)
  - [5.1 Deployment Architecture](#51-deployment-architecture)
  - [5.2 Monitoring: Phoenix + OpenTelemetry](#52-monitoring-phoenix--opentelemetry)

---

## 1. Data

### 1.1 Dataset

The project uses **GovReport** (`ccdv/govreport-summarization` on HuggingFace) — a dataset of U.S. government report summaries.

| Attribute | Value |
|-----------|-------|
| Domain | U.S. government reports |
| Document types | GAO (Government Accountability Office), CRS (Congressional Research Service), CBO (Congressional Budget Office) |
| Language | English |
| Total samples | 18,490 (train: 17,517 — validation: 973) |

### 1.2 Statistics

> Detailed analysis: [`finetuning/data_analysis/full_dataset_analysis.ipynb`](finetuning/data_analysis/full_dataset_analysis.ipynb)

Analysis of all 18,490 samples using the Qwen3-0.6B tokenizer:

| Attribute | Document | Summary |
|-----------|----------|---------|
| Characters (min / max) | 320 / 1,323,870 | 127 / 13,652 |
| Characters (mean / median) | 51,194 / 42,768 | 3,235 / 3,306 |
| Words (mean / median) | 7,750 / 6,480 | 483 / 496 |
| Tokens (mean / median) | 9,988 / 8,294 | 628 / 648 |
| Tokens (P95 / max) | 22,367 / 338,822 | — / 2,388 |

Compression ratio: mean **18.3x**, median **14.0x** (by word count).

### 1.3 Distribution by Context Length

```
≤ 8K tokens:    8,215 samples (44.4%)  — Short
8K – 16K:       7,677 samples (41.5%)  — Medium
16K – 32K:      2,278 samples (12.3%)  — Long
> 32K:            320 samples  (1.7%)  — Very long
                ─────────────────────
Cumulative ≤32K: 98.3% of total dataset
```

### 1.4 Key Characteristics Influencing Design

1. **Long-context:** Average ~10K tokens/document, P95 reaches ~22K tokens, max 339K tokens — requires a model that supports long context.
2. **Heavy-tail distribution:** 98.3% of samples are ≤ 32K tokens, but 1.7% (320 samples) far exceed this threshold.
3. **High compression ratio:** 18x compression — summarization needs to capture key points from a large volume of text.
4. **Structured domain:** Government reports have clear structure (context -> findings -> implications).

---

## 2. Approach

### 2.1 Overall Strategy

From the data characteristics above, two main challenges emerge:

| Challenge | Solution |
|-----------|----------|
| Documents average ~10K tokens, requiring a model that understands the government report domain | **Fine-tuning** (QLoRA) on GovReport so the model learns to summarize within this domain |
| Model context window = 32K tokens, but 1.7% of documents exceed 32K (max 339K) | **Agentic pipeline** splits documents -> summarizes each part -> synthesizes |

Additionally, the project uses **prompt engineering** combined with fine-tuning: specialized system prompts (expert persona, guidelines, structured output) are centrally managed in `prompts.yaml`, shared across training -> serving -> evaluation.

### 2.2 Fine-tuning: QLoRA on Qwen3-0.6B

#### Why QLoRA?

| Method | VRAM | Adapter size | Trade-off |
|--------|------|-------------|-----------|
| Full fine-tuning | 16GB+ | — | Catastrophic forgetting, heavy |
| LoRA (FP16) | 8GB+ | ~68MB | Good but VRAM-intensive |
| **QLoRA (4-bit)** | **~6GB** | **~34MB** | Balances quality / efficiency |
| Prompt engineering only | ~2GB | 0 | Insufficient for specialized domain |

#### Adapter Configuration

```
Qwen3-0.6B (0.6B params, frozen 4-bit NF4)
    └── QLoRA Adapters (~34MB, trainable)
        ├── Attention: q_proj, k_proj, v_proj, o_proj
        └── FFN: gate_proj, up_proj, down_proj
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 32 | Balances capacity/efficiency for a 0.6B model |
| Alpha | 64 | 2 x rank |
| Dropout | 0 | Required by Unsloth fast patching |
| Target modules | 7 linear layers | All attention + FFN layers |
| Max seq length | 32,768 tokens | Covers 98.3% of the dataset |

#### Training Data Processing

- **Filtering:** Removed 320 samples > 32K tokens (exceeds context window -> silent truncation)
- **Train/Val split:** 16,641 train / 1,849 validation (after filtering)

Training data distribution (after filtering):

```
≤ 8K:    ~44%  (short docs)
8K–16K:  ~42%  (medium docs)
16K–32K: ~14%  (long docs)
```

#### Prompt Template

Each training sample is formatted using `tokenizer.apply_chat_template()` with a custom template that removes Qwen3's default `<think>` blocks (unnecessary for summarization and wastes tokens):

```
<|im_start|>system
You are an expert summarizing government reports.

Guidelines:
• Read the entire document before summarizing
• Include all major findings and their significance
• Use only facts from the document - never invent numbers, dates, or names
• Organize clearly: context → findings → implications
• Match summary length to document complexity<|im_end|>
<|im_start|>user
Summarize the following government report.

Government Report:
{document}

Now write a summary covering: (1) what was examined and why,
(2) key findings and conclusions, (3) implications or significance.
Include specific numbers and details when available. Do not repeat information.<|im_end|>
<|im_start|>assistant
{summary}<|im_end|>
```

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 2.0e-4 |
| Weight decay | 0.15 |
| Warmup ratio | 0.03 |
| Epochs | 4 |
| Batch size | 4 (per device) x 8 (grad accum) = **32 effective** |
| LR scheduler | Cosine |
| Optimizer | AdamW 8-bit |
| Precision | BF16 |

Total: ~2,080 steps, ~28 hours on A100 40GB.

#### GRPO (Experimental — not yet deployed)

> Complete code implementation (`train_grpo.py`) but not yet tested due to time constraints. Idea: optimize the model using rewards from an LLM judge instead of just imitating the reference summary.

### 2.3 Agentic Pipeline: Handling Documents Beyond Context Window

For the 1.7% of documents exceeding 32K tokens (and in production where any document may be received), the project implements an **agentic pipeline** using LangGraph:

```
Input document
       │
       ▼
  chunk_document (measure length, routing)
       │
       ├── < 120K characters (~97% of cases)
       │   └── direct_summarize
       │       Calls LLM once with the entire document
       │       → Final result
       │
       └── ≥ 120K characters (~3% of cases)
           └── summarize_chunks (hierarchical)
               ├── Level 1: Split document into N chunks (~25K characters/chunk)
               │             8-sentence overlap between chunks
               │             Summarize each chunk → N section summaries
               │
               └── Level 2: Synthesize N section summaries
                             → 1 final summary
```

**Routing design:**
- Threshold `120K characters` (~30K tokens): based on data analysis, 97% of GovReport documents fall below this threshold
- Direct path is optimized for the majority of cases (1 LLM call), only activating hierarchical when truly necessary
- Chunking at **sentence boundaries** to avoid splitting mid-thought

---

## 3. Evaluation

### 3.1 Evaluation Design

Comparing **3 methods** on the same test set:

| # | Method | Description |
|---|--------|-------------|
| 1 | **Base model** | Original Qwen3-0.6B + prompt engineering, no fine-tuning |
| 2 | **Fine-tuned** | Qwen3-0.6B + QLoRA adapter (checkpoint-300) |
| 3 | **Agent pipeline** | Fine-tuned model + agentic chunking (not yet evaluated) |

### 3.2 Test Set

**100 samples**, stratified by length from the GovReport test split:

| Category | Token range | Sample count | Proportion |
|----------|-----------|--------------|------------|
| Short (8K) | ≤ 8,192 | 50 | 50% |
| Medium (16K) | 8,193 – 16,384 | 30 | 30% |
| Long (32K) | 16,385 – 32,768 | 20 | 20% |

Stratification ensures evaluation of performance across all lengths, avoiding bias from the majority category.

### 3.3 Metrics

#### A. ROUGE — Lexical overlap

- **ROUGE-1** (unigram F1), **ROUGE-2** (bigram F1), **ROUGE-L** (Longest Common Subsequence)
- **Pros:** Fast, reproducible, correlates with human judgment
- **Cons:** Penalizes paraphrasing (correct meaning but different words -> low score), does not capture semantics

#### B. Embedding Similarity — Semantic

- **Model:** BAAI/bge-m3 (1024-dim, supports long documents)
- **Metric:** Cosine similarity between reference and generated summary embeddings
- **Pros:** Robust to rephrasing, captures semantic similarity
- **Cons:** Cannot distinguish hallucination if semantics are similar

#### C. LLM-as-Judge — Multi-dimensional quality

- **Model:** NVIDIA Nemotron Nano 9B (free via NVIDIA API)
- **4 evaluation dimensions (1-5 score):**

| Dimension | Evaluates | Weight (GRPO) |
|-----------|-----------|---------------|
| **Coverage** | Are all key points covered? | 0.3 |
| **Specificity** | Are there specific numbers, dates, names? | 0.2 |
| **Consistency** | Are facts accurate? No hallucination? | 0.4 |
| **Conciseness** | Concise, well-structured? | 0.1 |

- **Pros:** Multi-dimensional evaluation close to human judgment, can detect hallucination
- **Cons:** Depends on judge model quality, API cost

### 3.4 Results

**Base Model** (Qwen3-0.6B + prompt engineering, no fine-tuning, 95 samples):

| Metric | 8K (n=47) | 16K (n=28) | 32K (n=20) | Overall |
|--------|-----------|------------|------------|---------|
| ROUGE-1 | 0.355 | 0.410 | 0.406 | **0.382** |
| ROUGE-L | 0.158 | 0.183 | 0.167 | **0.167** |
| Embedding | 0.806 | 0.827 | 0.817 | **0.814** |
| Coverage | 3.49 | 3.50 | 3.60 | **3.52** |
| Specificity | 2.87 | 2.86 | 3.15 | **2.93** |
| Consistency | 4.34 | 4.75 | 4.65 | **4.53** |
| Conciseness | 4.49 | 4.50 | 4.50 | **4.49** |
| Avg words | 429 | 512 | 594 | **488** |

**Fine-tuned** (QLoRA checkpoint-300, 98 samples):

| Metric | 8K (n=49) | 16K (n=30) | 32K (n=19) | Overall |
|--------|-----------|------------|------------|---------|
| ROUGE-1 | 0.436 | 0.482 | 0.502 | **0.463** |
| ROUGE-L | 0.187 | 0.203 | 0.202 | **0.195** |
| Embedding | 0.840 | 0.857 | 0.858 | **0.849** |
| Coverage | 2.88 | 3.43 | 3.58 | **3.18** |
| Specificity | 2.47 | 3.17 | 3.21 | **2.83** |
| Consistency | 3.43 | 4.10 | 4.63 | **3.87** |
| Conciseness | 2.76 | 3.70 | 4.26 | **3.34** |
| Avg words | 810 | 742 | 816 | **791** |

**Overall comparison:**

| Metric | Base | Fine-tuned | Δ |
|--------|------|-----------|---|
| ROUGE-1 | 0.382 | 0.463 | **+21.2%** |
| ROUGE-L | 0.167 | 0.195 | **+16.8%** |
| Embedding | 0.814 | 0.849 | **+4.3%** |
| Coverage | 3.52 | 3.18 | **-9.7%** |
| Specificity | 2.93 | 2.83 | **-3.4%** |
| Consistency | 4.53 | 3.87 | **-14.6%** |
| Conciseness | 4.49 | 3.34 | **-25.6%** |
| Avg words | 488 | 791 | **+62.1%** |

---

## 4. Error Analysis

> Details: [`finetuning/data_analysis/error_analysis.ipynb`](finetuning/data_analysis/error_analysis.ipynb)

### 4.1 Error Detection Flow

```
Evaluation pipeline
       │
       ▼
  Run 100 samples through model → generate summaries
       │
       ▼
  Compute ROUGE + Embedding + LLM Judge
       │
       ▼
  Filter low-score samples:
  ├── Judge avg < 4.0 ?
  ├── Coverage < 3 ?
  ├── Specificity < 3 ?
  ├── Consistency < 3 ?
  └── Conciseness < 3 ?
       │
       ▼
  Export → low_score_samples.jsonl
       │
       ▼
  Error Analysis notebook:
  ├── Side-by-side comparison: Document / Reference / Generated
  ├── Read judge explanation for each sample
  └── Classify error patterns
```

### 4.2 Analysis Results

**Base model** — 9 low-score samples / 95 total (~9.5%):

| Error type | Count | Description |
|------------|-------|-------------|
| Low Specificity (≤ 2) | 8 | Missing specific figures (amounts, percentages, dates) — model summarizes too generically |
| Low Consistency (≤ 2) | 2 | Hallucination — fabricates information not in the document |
| Multiple issues (≥ 2 dimensions ≤ 2) | 1 | Both specificity and consistency are low |

Mean scores for the low-score group: Coverage 3.22, Specificity **2.11**, Consistency 4.22, Conciseness 3.33.

**Key observations:**
- **Specificity is the biggest weakness** — the model tends to omit specific numbers, dates, and organization names
- Common judge explanation: *"omits specific details like numbers (e.g., 11% vacancy rate), names, and dates"*

**Fine-tuned model** — Different issues:

| Observation | Details |
|-------------|---------|
| **Verbose** | Generates 791 words vs 488 words (base) — 62% longer |
| **Repetition** | Repeats the same text passages multiple times within a single summary |
| **Low Conciseness (8K)** | Judge score 2.76/5 for short docs — model rambles |
| **Early checkpoint** | Checkpoint-300 (~14% of training) — model has not converged |

Typical example (fine-tuned, sample_id=0): the model repeats the passage *"Addressing these challenges will require active leadership support..."* **more than 10 times** within a single summary.

### 4.3 Observations & Improvement Directions

| Issue | Root cause | Improvement direction |
|-------|-----------|----------------------|
| Lack of specificity | Prompt does not sufficiently enforce extracting specific figures | Add few-shot examples, add emphasis in prompt |
| Repetition (fine-tuned) | Early checkpoint, model has not learned when to stop | Train for full epochs, increase repetition_penalty |
| Verbose output | Fine-tuning on long references biases the model | GRPO with conciseness reward, length penalty |
| Hallucination (2 samples) | Small model (0.6B) is prone to confabulation | High consistency weight in GRPO (0.4) |

---

## 5. System Architecture

### 5.1 Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│  Browser — http://localhost:7860                 │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  Gradio Frontend (Docker, :7860)                │
│  Text input → POST /summarize → display results │
└───────────────────┬─────────────────────────────┘
                    │ POST /summarize
                    ▼
┌─────────────────────────────────────────────────┐
│  FastAPI + LangGraph (Docker, :8001)            │
│  ├── chunk_document → route by length           │
│  ├── direct_summarize (< 120K chars)            │
│  ├── summarize_chunks (≥ 120K chars, 2-level)   │
│  └── OpenTelemetry tracing                      │
└───────────────────┬─────────────────────────────┘
                    │ /v1/chat/completions (OpenAI API)
                    ▼
┌─────────────────────────────────────────────────┐
│  llama.cpp (Native, :8080)                      │
│  Qwen3-0.6B GGUF (Q4_K_M, ~400MB)              │
│  Metal GPU / CUDA — 32K context                 │
└─────────────────────────────────────────────────┘
```

> **Note:** llama.cpp runs natively (not in Docker) to leverage Metal GPU on Mac. Docker Desktop runs a Linux VM, which cannot access Metal.

### 5.2 Monitoring: Phoenix + OpenTelemetry

```
LangGraph Pipeline
    │
    ├── OpenTelemetry spans (each node = 1 span)
    │
    └── gRPC (:4317) → Phoenix (:6006)
                            │
                            └── Dashboard:
                                ├── Trace timeline (per node)
                                ├── Latency breakdown
                                ├── Prompt / Response pairs
                                └── Token usage per request
```

**Phoenix** (Arize, port 6006) enables:
- **Pipeline debugging:** View input/output of each node (chunk_document, direct_summarize, summarize_chunks)
- **Performance monitoring:** Processing time per step, bottleneck detection
- **Quality inspection:** Read prompt/response pairs for real-time quality checking
