# Sarcasm-in-the-Browser Fine-Tuning Challenge

## Executive Summary
> **Proof-of-concept (PoC):** Feel free to skip non-essential parts—focus on demonstrating the core flow.

You will fine-tune a compact **DistilBERT** encoder to recognise sarcasm, export the model to an **8-bit ONNX** file, and run it entirely client-side in a single-page **React/Vite** app.  
Public sarcasm datasets (≈ 10 k sentences) fit on a laptop, and **INT8 quantisation** shrinks the model to ≈ 18 MB, achieving sub-200 ms latency with WebGPU on mainstream hardware.

The assignment tests your skills in:
* data curation  
* Hugging Face **Trainer**  
* optimisation with **Optimum**  
* front-end deployment using **Transformers.js** *or* **ONNX Runtime Web**

---

## 1  Challenge Overview

### 1.1  Objective
Build a **browser-only sarcasm detector** that, given an English sentence, returns:
* a confidence score (`0 = sincere`, `1 = sarcastic`)
* a friendly emoji 😊 / 🙃

The entire pipeline—tokeniser, model, post-processing—must run locally in the user’s tab; **no server calls allowed**.

### 1.2  Why Sarcasm?
Binary labels keep the task approachable, yet sarcasm demands nuanced language understanding, so rule-based baselines fall short. This lets us judge your modelling chops without huge compute.

### 1.3  Recommended Ingredients

| Component           | Default Pick                                                            | Rationale                                                         | Key Spec                          |
|---------------------|--------------------------------------------------------------------------|-------------------------------------------------------------------|-----------------------------------|
| **Base model**      | `distilbert-base-uncased`                                               | 66 M params, 6 layers, 97 % BERT-style accuracy at 40 % size      | ~250 MB FP32 → ≤ 20 MB INT8       |
| **Datasets**        | Sarcasm Corpus V2, iSarcasm, News-Headlines                             | Balanced labels, permissive licences                             | ~9 k–20 k samples                 |
| **Exporter**        | `optimum-cli export onnx --quantize dynamic`                            | One-command ONNX + quantisation                                   | 18–22 MB ONNX                     |
| **Browser runtime** | (A) **Transformers.js** (WASM/WebGPU)<br>(B) **ONNX Runtime Web** WebGPU | Both support INT8; WebGPU halves latency on modern laptops        | < 150 ms on desktop               |
| **Frontend**        | **React + Vite**                                                        | Instant HMR dev-server, zero-config static build                  | Bundle ≤ 1 MB gz                  |

---

## 2  Deliverables
* **Public Git repo** with reproducible code and a `README.md` detailing:  
  * dataset(s) used and licence links  
  * training hyper-parameters  
  * final metrics (F1, accuracy)  
* `model.onnx` ≤ 25 MB and `tokenizer.json` committed under `/public/model`.
* **Browser demo** at `<your-url>/sarcasm/` featuring:  
  * textarea input  
  * sarcasm probability bar ± emoji  
  * latency log in console  
* Short screencast (< 2 min, GIF or MP4) *or* live URL proving offline mode—reload with Wi-Fi disabled.

---

## 3  Step-by-Step Guide

### 3.1  Environment
```bash
conda create -n sarcasm python=3.10
conda activate sarcasm
pip install "transformers>=4.40" datasets evaluate accelerate \
           "optimum[onnxruntime,gpu]" onnxruntime-web==1.17.0 \
           @huggingface/transformers     # for JS side
```
> ONNX Runtime Web ≥ 1.17 adds official WebGPU support.

### 3.2  Load Data
```python
from datasets import load_dataset
ds = load_dataset("Orbay/sarcasm_corpus_v2", "default")  # or your choice
```
Filter/clean as needed; keep a **10 % hold-out** test split.

### 3.3  Fine-Tune with Trainer
Follow the Hugging Face sequence-classification recipe; **2 epochs**, learning rate `2e-5`, batch `16` on a single GPU hits **F1 ≈ 0.82**.

### 3.4  Checkpoint & Metrics
Save the best checkpoint (`save_total_limit=1`) and log F1/accuracy with `evaluate` for reproducibility.

### 3.5  Export & Quantise
```bash
optimum-cli export onnx --model path/to/ckpt onnx/ --quantize dynamic
```
Dynamic INT8 squeezes DistilBERT to ≈ 18 MB with negligible accuracy loss.

### 3.6  Browser Deployment

#### Option A — Transformers.js
```js
import { pipeline } from '@huggingface/transformers';

const clf = await pipeline('text-classification', '/public/model', { quantized: true });
const { label, score } = (await clf(userInput))[0];
```
Transformers.js auto-detects WebGPU when enabled.

#### Option B — ONNX Runtime Web
```js
import * as ort from 'onnxruntime-web';

const sess = await ort.InferenceSession.create('/model.onnx', {
  executionProviders: ['webgpu', 'wasm']
});
```
WebGPU cuts inference nearly in half vs pure WASM.

