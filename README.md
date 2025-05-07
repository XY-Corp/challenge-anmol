# Fake-News Detection in the Browser – Fine-Tuning Challenge

You will fine-tune a compact **DistilBERT** encoder to decide whether a news article is *fake* or *real*, export the model to an **8-bit ONNX** file, and run it fully client-side in a single-page **React / Vite** app.  
The popular *Fake and Real News* corpus (~44 k articles) fits on a laptop, and **INT8** quantisation shrinks the model to ≈ 18 MB, keeping browser latency below 200 ms with WebGPU.

---

## 1. Challenge Overview

### 1.1 Objective  
Build a **browser-only fake-news detector** that takes an article (headline + text) and returns  

* a probability (`0 = real`, `1 = fake`)
* a quick verdict badge

All inference must remain in the user’s tab—**no server calls**.

### 1.2 Why Fake vs Real?  
Misinformation detection is topical, involves longer inputs than tweets, and requires more than surface-level cues—perfect for demonstrating practical NLP skills without huge compute.

### 1.3 Recommended Ingredients  

| Component        | Default Pick                                                                  | Rationale                                                     | Key Spec                     |
|------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------|------------------------------|
| **Dataset**      | *Fake and Real News* (Kaggle)                                                 | 23 k fake + 21 k real; permissive CC0-like licence            | CSV (~35 MB)                 |
| **Base model**   | `distilbert-base-uncased`                                                     | 66 M params, 6 layers, ≈ 97 % of BERT at 40 % size            | 250 MB FP32 → ≤ 20 MB INT8   |
| **Exporter**     | `optimum-cli export onnx --quantize dynamic`                                  | One-command ONNX + quantisation                              | 18–22 MB ONNX                |
| **Browser EP**   | (A) **Transformers.js** (WASM/WebGPU) <br> (B) **ONNX Runtime Web** (WebGPU)  | Both support INT8; WebGPU halves latency                      | < 200 ms on desktop          |
| **Frontend**     | **React + Vite**                                                              | Instant HMR dev-server, zero-config static build             | Bundle ≤ 1 MB gz             |

---

## 2. Deliverables

1. **Public Git repo** with a concise `README.md` describing  
   * dataset provenance & licence link  
   * training hyper-parameters  
   * evaluation metrics (F1, accuracy)
2. `model.onnx` ≤ 25 MB and `tokenizer.json` committed under `/public/model/`.
3. **Browser demo** at `<your-url>/fakenews/` featuring  
   * textarea (or drop-zone) for article text  
   * probability bar ± verdict badge  
   * console log of inference latency
4. Short screencast (< 2 min) *or* live URL proving the app works **offline** (reload with Wi-Fi disabled).

---

## 3. Step-by-Step Guide

### 3.1 Environment
```bash
conda create -n fakenews python=3.10
conda activate fakenews
pip install "transformers>=4.40" datasets evaluate accelerate \
           "optimum[onnxruntime,gpu]" onnxruntime-web==1.17.0 \
           @huggingface/transformers      # JS side
```
> ONNX Runtime Web ≥ 1.17 adds official WebGPU support.

### 3.2 Load Data
```python
from datasets import load_dataset
# The Kaggle CSV can be loaded directly once downloaded locally
news = load_dataset(
    "csv",
    data_files={"train": "train.csv", "test": "test.csv"}
)
```
Split 10 % of training into validation; truncate/clean text as desired.

### 3.3 Fine-Tune with `Trainer`
* Task: sequence classification  
* Epochs: **2**  
* Learning rate: **2 e-5**  
* `max_length`: **384**  
* Batch size: **8–16** (GPU-dependent)  
Monitor F1 on the validation set.

### 3.4 Checkpoint & Metrics
Keep the best checkpoint (`save_total_limit=1`) and log metrics with `evaluate`.

### 3.5 Export & Quantise
```bash
optimum-cli export onnx --model path/to/best onnx/ --quantize dynamic
```
Expect an ~18 MB INT8 model.

### 3.6 Browser Deployment

#### Option A — Transformers.js
```js
import { pipeline } from "@huggingface/transformers";

const clf = await pipeline(
  "text-classification",
  "/public/model",
  { quantized: true }
);
const { label, score } = (await clf(articleText))[0];
```

#### Option B — ONNX Runtime Web
```js
import * as ort from "onnxruntime-web";

const session = await ort.InferenceSession.create("/model.onnx", {
  executionProviders: ["webgpu", "wasm"]
});
```

---

Good luck—help us separate **facts** from **fiction**!
