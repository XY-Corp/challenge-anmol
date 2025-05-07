# Sarcasm‑in‑the‑Browser Fine‑Tuning Challenge

## Executive Summary

Please note: This challenge is a proof‑of‑concept (PoC). It's perfectly fine if you don't complete every section—focus on demonstrating the core flow.

You will fine‑tune a compact DistilBERT encoder to recognise sarcasm, export the model to an 8‑bit ONNX file and run it entirely client‑side in a single‑page React/Vite app. Public sarcasm datasets (~10 k sentences) fit on a laptop, and INT8 quantisation shrinks the model to ≈ 18 MB, achieving sub‑200 ms latency with WebGPU on mainstream hardware. The assignment tests your skill across data curation, Hugging Face Trainer, optimisation with Optimum, and front‑end deployment using either Transformers.js or ONNX Runtime Web.

## 1 Challenge Overview

### 1.1 Objective

Create a browser‑only sarcasm detector that takes an English sentence and returns a confidence score (0 = sincere, 1 = sarcastic) plus a friendly emoji. The entire pipeline—tokeniser, model, post‑processing—must run locally in the user's tab; no server calls allowed.

### 1.2 Why Sarcasm?

Binary labels keep the task approachable, yet sarcasm demands nuanced language understanding, so rule‑based baselines fall short. This lets us judge your modelling chops without huge compute.

### 1.3 Recommended Ingredients

| Component | Default Pick | Rationale | Key Spec |
|-----------|--------------|-----------|----------|
| Base model | distilbert-base-uncased | 66 M params, 6 layers, 97 % BERT-style accuracy at 40 % size | ~250 MB FP32 → ≤ 20 MB INT8 |
| Datasets | Sarcasm Corpus V2, iSarcasm, News‑Headlines | Balanced labels, permissive licences | ~9 k–20 k samples |
| Exporter | Optimum CLI export onnx --quantize dynamic | One‑command ONNX + quantisation | 18–22 MB ONNX |
| Browser runtime | (A) Transformers.js pipeline (WASM/WebGPU) or (B) ONNX Runtime Web with WebGPU EP | Both support INT8; WebGPU halves latency on modern laptops | < 150 ms on desktop |
| Frontend | React + Vite | Instant HMR dev‑server, zero‑config static build | Bundle ≤ 1 MB gz |

## 2 Deliverables

Public Git repo with reproducible code and a README.md detailing:

- dataset(s) used and licence links
- training hyper‑parameters
- final metrics (F1, accuracy)
- model.onnx ≤ 25 MB and tokenizer.json committed under /public/model.

Browser demo at <your-url>/sarcasm/ with:

- textarea input
- sarcasm probability bar ± emoji
- latency log in console

Short screencast (< 2 min, GIF or MP4) or live URL proving offline mode—reload with Wi‑Fi disabled.

## 3 Step‑by‑Step Guide

### 3.1 Environment
