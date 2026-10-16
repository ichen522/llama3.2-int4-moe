# Efficient Llama 3.2-1B Inference: INT4 Quantization & Mixture-of-Experts

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)

This project looks at two ways to make inference cheaper for Meta's **Llama 3.2-1B**, implemented from scratch in PyTorch:

1. **INT4 weight-only quantization.** Per-group (g = 128) asymmetric round-to-nearest quantization, with two 4-bit weights packed into each byte. It replaces every `nn.Linear` in the model.
2. **Mixture-of-Experts (MoE) conversion.** Each dense SwiGLU FFN becomes a top-2-of-4 routed MoE layer. I built two initialization strategies: *slicing* the pretrained FFN into experts, and keeping the FFN frozen with routed *LoRA adapters* as experts.

Final project for **EE 508: Hardware Foundations of Machine Learning** at the University of Southern California (Spring 2026). All experiments ran on a single NVIDIA A40 (48 GB).

---

## Key results

| | Result |
|---|---|
| **INT4 model size** | 2,858 MB → **1,127 MB** (−60.6 %). Output stays coherent |
| **INT4 runtime** | 5.7–7.5× *slower* than FP16 with the naive dequantize-then-matmul path (analysis [below](#why-is-naive-int4-slower-and-larger-at-runtime)) |
| **LoRA-MoE quality** | Held-out perplexity **9.59** vs. 9.66 for the dense model. Only **0.15 %** of parameters are trainable |
| **Slice-MoE quality** | Perplexity rises to 6,089 after a 200-sample fine-tune. Splitting the FFN destroys pretrained knowledge that a small dataset can't restore |

---

## What I implemented

The base model, tokenizer, generation loop and experiment drivers come from the course starter code, which is adapted from Meta's [`llama3`](https://github.com/meta-llama/llama3) reference implementation. My work is in these two files:

| File | Components |
|---|---|
| [`llama/quantize.py`](llama/quantize.py) | `QuantizedLinear` (quantize, pack, unpack, dequantize, forward), `quantize_model()` |
| [`llama/moe.py`](llama/moe.py) | `MoEFeedForward` (slice mode), `LoRAExpert`, `LoRAMoEFeedForward` (LoRA mode), `convert_to_moe()` |

Both pass every check in [`check_student.py`](check_student.py): shapes, quantization round-trip error, size reduction, routing weights summing to 1, exact parameter counts, zero-init identity for LoRA-MoE, and frozen/trainable parameter split.

---

## Part 1: INT4 weight-only quantization

### Method

For each row of a weight matrix, every group of 128 consecutive input weights gets its own scale and zero-point:

```
scale      = (w_max − w_min) / 15
zero_point = clamp(round(−w_min / scale), 0, 15)
w_int4     = clamp(round(w / scale) + zero_point, 0, 15)
packed     = w_int4[even] | (w_int4[odd] << 4)        # 2 weights per uint8
```

At inference time `QuantizedLinear.forward()` unpacks the nibbles, computes `(w_int4 − zero_point) × scale` to rebuild an FP16 weight matrix, and calls `F.linear`. Groups where every weight is equal (scale = 0) are handled so the code never divides by zero. `quantize_model()` walks `named_modules()` and uses `setattr` on each parent to swap every `nn.Linear` for a `QuantizedLinear`. That covers the attention projections, the FFN weights and the LM head.

### Model size

| | Size |
|---|---|
| FP16 model | 2,858.13 MB |
| INT4 model | **1,127.21 MB** (−60.6 %) |
| ↳ INT4 buffers (packed weights + FP16 scales and zero-points) | 626.08 MB |
| ↳ FP16 parameters left unquantized (token embedding, RMSNorm) | 501.13 MB |

The token embedding table (128,256 × 2,048) is an `nn.Embedding`, not an `nn.Linear`, so it stays in FP16. That table is why the total saving is about 60 % instead of close to 75 %.

### Benchmark (input = 256 tokens, output = 32 tokens)

| | batch = 1 | batch = 8 | batch = 16 |
|---|---|---|---|
| **FP16** peak memory (MB) | 3,072 | 4,495 | 6,134 |
| **FP16** runtime (s) | 0.69 | 0.55 | 0.65 |
| **FP16** throughput (tok/s) | 46.1 | 466.7 | 782.1 |
| **INT4** peak memory (MB) | 5,301 | 6,269 | 7,382 |
| **INT4** runtime (s) | 3.91 | 4.11 | 4.30 |
| **INT4** throughput (tok/s) | 8.2 | 62.3 | 119.0 |

### Sample output (few-shot translation)

```
Translate English to French:
    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese =>
```

| FP16 | INT4 |
|---|---|
| `fromage`, then `onion => oignon`, `potato => pomme de terre`, … | `fromage`, then `car => voiture`, `toot => trombone`, … |

Both models get the target word right and keep the few-shot format. After that their continuations differ, but the INT4 model still produces coherent text.

### Why is naive INT4 slower and larger at runtime?

Decoding is memory-bandwidth bound: at small batch sizes, every weight is read from HBM for roughly one multiply-add. INT4 should therefore help in principle. This implementation, however, **materializes the whole dequantized FP16 weight in HBM before every matmul**. As a result:

- **Memory traffic goes up, not down.** The GPU reads the INT4 weights, writes an FP16 copy, and then reads that copy again for the GEMM. Each step is a separate, unfused PyTorch kernel with its own launch overhead.
- **Peak memory goes up.** The packed buffer and the temporary dequantized weight are alive at the same time. `dequantize_packed()` also goes through FP32 intermediates. For the 128,256 × 2,048 LM head, each FP32 temporary is about 1 GB, so these temporaries account for much of the extra peak memory. The overhead shrinks relative to the total as batch size grows (+73 % at batch 1, +20 % at batch 16) because activations take up a growing share of memory.

A **fused dequantize + GEMM kernel** fixes both problems. It loads INT4 tiles, dequantizes them in registers or shared memory, and multiplies them right away, so the FP16 weight never touches HBM. Weight traffic then drops about 4×, which is where the expected decode speedup comes from. Libraries such as bitsandbytes and CUTLASS INT4 GEMM work this way. The full report also discusses GPTQ, AWQ, GGUF K-quants and KV-cache quantization.

---

## Part 2: Mixture-of-Experts conversion

Each of the 16 `FeedForward` layers (dim 2,048, SwiGLU hidden size 8,192) is replaced with a routed MoE layer using **N = 4 experts and top-K = 2**. The router is `softmax(W_router · x)`. The top-2 weights are renormalized so they sum to 1.

### Slice mode (`MoEFeedForward`)

The pretrained `w1`, `w3` (rows) and `w2` (columns) are split along the hidden dimension into 4 non-overlapping slices. Each expert is a small SwiGLU FFN with hidden size 2,048. By default only the router is trained.

### LoRA mode (`LoRAMoEFeedForward`)

The original FFN stays **frozen**, and the experts are low-rank adapters added on top:

```
y = FFN_frozen(x) + Σ_{k ∈ top-2} g_k · B_k A_k x · (α / r)        r = 8, α = 16
```

`B` starts at zero, so the converted model **produces exactly the same logits as the dense model** before any training. The adapters and router run in FP32 for numerical stability: inputs are cast up and outputs cast back to FP16. Each expert adds only 2 × 2,048 × 8 = 32,768 parameters. A full slice expert has about 12.6 M.

### Results

| Metric | Dense | MoE-slice (N=4, K=2) | MoE-LoRA (N=4, K=2, r=8) |
|---|---|---|---|
| Total parameters | 1,498,482,688 | 1,498,613,760 | 1,500,710,912 |
| Trainable parameters | — | 131,072 (0.01 %) | 2,228,224 (0.15 %) |
| Peak memory (MB) | 3,072 | ~5,300 | ~5,300 |
| Inference time (s), batch 1, 256 in / 32 out | 0.65 | ~1.0 | ~1.0 |
| **Perplexity (held-out) ↓** | 9.66 | 6,088.74 | **9.59** |
| **Next-token accuracy (held-out) ↑** | 51.30 % | 3.57 % | **51.53 %** |

### Training loss

![Training loss curves](figures/phase3_loss_curves.png)

- **Slice:** loss starts at 11.66 and falls to 8.76 over 2 epochs, but stays far above dense-level loss. The router is learning from scratch, and each token now sees only half of the original FFN capacity.
- **LoRA:** loss starts at the dense level (about 2.2) and stays flat (about 2.2–2.4). The frozen base keeps pretrained quality, and the adapters only learn a small correction.

### Expert load balance (first 4 layers, share of top-2 assignments)

| Layer | Slice: E0 / E1 / E2 / E3 | LoRA: E0 / E1 / E2 / E3 |
|---|---|---|
| 0 | 13 / 11 / 30 / 46 % | 25 / 21 / 34 / 21 % |
| 1 | 42 / 34 / 13 / 12 % | 30 / 11 / 21 / 38 % |
| 2 | 9 / 13 / 46 / 32 % | 17 / 33 / 20 / 30 % |
| 3 | 21 / 38 / 16 / 24 % | 31 / 28 / 31 / 10 % |

No expert collapses in either mode. Slice mode is noticeably more skewed, with one expert taking up to 46 % of the load. Neither mode uses an auxiliary load-balancing loss.

### Sample generations (after fine-tuning)

| Prompt | Dense | MoE-slice | MoE-LoRA |
|---|---|---|---|
| *Compose a short poem about the beauty of mathematics.* | "The poem must be between 20 and 30 lines long…" | "in in in in in down down down…" | "You can use any word in the poem, but you should choose a word that has a mathematical meaning…" |
| *If a train travels at 60 mph for 2.5 hours, how far does it go?* | "Answer: 150 miles…" | `" " " " " "…` | "1. 1,800 miles 2. 1,500 miles 3. 1,000 miles…" |

### Takeaways

- **Slice mode** assumes the dense FFN already contains separable sub-functions along its hidden dimension. In practice, routing a token to only 2 of 4 slices removes half the capacity it was trained with. A router trained on 200 samples can't recover from that.
- **LoRA mode** treats MoE as a small perturbation on a model that already works. Zero-initialized `B` guarantees the result is no worse than dense at step 0, and freezing the base prevents catastrophic forgetting. That makes it the right choice for small-data conversion.
- **Wall-clock overhead** comes from the Python-level dispatch: a loop over experts and top-k slots, with boolean masks, that launches up to 8 small GEMMs per layer. Production MoE systems sort tokens by expert and run a single grouped or batched GEMM instead.

### Training setup

| | |
|---|---|
| Data | Alpaca: samples 0–199 for training, 200–299 held out for evaluation. MT-bench prompts for generation |
| Epochs / sequence length | 2 epochs, 64 tokens |
| Optimizer | SGD (momentum 0.9), gradient clipping at 1.0 |
| Learning rate | 1e-3 for the router, 1e-4 for the LoRA adapters / experts |
| Hardware | NVIDIA A40 48 GB, PyTorch 2.6 + CUDA 12.4 |

---

## Repository structure

```
├── llama/
│   ├── model.py              # Llama 3.2-1B architecture (GQA, RoPE, RMSNorm, SwiGLU)
│   ├── generation.py         # Autoregressive generation with KV cache
│   ├── tokenizer.py          # tiktoken-based tokenizer
│   ├── quantize.py           # ★ INT4 QuantizedLinear + quantize_model
│   └── moe.py                # ★ Slice-MoE, LoRA-MoE, convert_to_moe
├── inference.py              # Smoke test: sample generations
├── benchmark_inference.py    # Single-config FP16 timing baseline
├── run_benchmark.py          # Full INT4 + MoE benchmark sweep
├── train_moe.py              # MoE fine-tuning (--init-mode slice | lora)
├── eval_moe.py               # Perplexity / next-token accuracy / tok/s evaluation
├── check_student.py          # Correctness checks for quantize.py and moe.py
├── prepare_data.py           # Download Alpaca + MT-bench prompts
├── figures/                  # Loss curves and expert-load results
└── Efficient_LLM_Inference_Project.md   # Full write-up with all analysis
```

---

## Running it

**1. Install dependencies** (PyTorch 2.6 + CUDA 12.4):

```bash
pip install -r requirements.txt
pip install datasets   # only needed for prepare_data.py
```

**2. Get the model weights.** These aren't included in this repo. Request access to Llama 3.2-1B from Meta, then:

```bash
pip install llama-stack
llama model download --source meta --model-id Llama3.2-1B
```

**3. Prepare data:**

```bash
python prepare_data.py --output-dir ./data
```

**4. Update paths.** The scripts point to the course server (`/project2/saifhash_1190/...`). Change `checkpoint_dir` and the data paths at the top of `inference.py`, `benchmark_inference.py`, `run_benchmark.py`, `train_moe.py`, `eval_moe.py` and `check_student.py` to your local copies.

**5. Run:**

```bash
python check_student.py                    # correctness checks
python inference.py                        # smoke test

python run_benchmark.py                    # INT4 vs FP16 and MoE benchmark tables

python train_moe.py --init-mode lora       # fine-tune LoRA-MoE → checkpoints/moe_finetuned.pt
python eval_moe.py  --init-mode lora       # perplexity / accuracy / speed

python train_moe.py --init-mode slice
python eval_moe.py  --init-mode slice
```

---

## Future work

- A **fused INT4 dequantize + GEMM kernel** (e.g. in Triton) so dequantized weights never reach HBM. Even without a custom kernel, dequantizing straight to FP16 instead of through FP32 would cut peak memory.
- **Calibration-based quantization** (GPTQ / AWQ) to compare against round-to-nearest at the same bit-width.
- **KV-cache quantization**, which complements weight quantization for long-context decoding.
- **Grouped-GEMM expert dispatch** and an **auxiliary load-balancing loss** for the MoE layers.

---

## Full report

[`Efficient_LLM_Inference_Project.md`](Efficient_LLM_Inference_Project.md) has the complete write-up, including background Q&A on transformer and LLM fundamentals and the full discussion of every analysis question.

## Acknowledgments

- Course: EE 508, Hardware Foundations of Machine Learning, USC (Spring 2026). Instructor: Arash Saifhashemi. The starter code and project specification were provided by the course staff.
- Model code adapted from Meta's [`llama3`](https://github.com/meta-llama/llama3) reference implementation. Use of Llama 3.2 is subject to the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE).
