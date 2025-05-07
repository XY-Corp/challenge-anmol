# challenge-anmol

Below is the complete Contestant Manual for the “Sarcasm-in-the-Browser” fine-tuning challenge. It explains the goal, resources, step-by-step workflow, deliverables, and evaluation rubric. Feel free to copy it verbatim into your onboarding docs or tweak the tone to match your brand voice.

⸻

Executive summary

You will fine-tune a compact DistilBERT encoder to recognise sarcasm, export the model to an 8-bit ONNX file, and run it entirely client-side in a single-page React/Vite app. Public sarcasm datasets (~10 k sentences) fit on a laptop, and INT8 quantisation shrinks the model to ≈ 18 MB, achieving sub-200 ms latency with WebGPU on mainstream hardware. The assignment tests your skill across data curation, Hugging Face Trainer, optimisation with Optimum, and front-end deployment using either Transformers.js or ONNX Runtime Web. ￼ ￼ ￼ ￼

⸻

1 Challenge overview

1.1 Objective

Create a browser-only sarcasm detector that takes an English sentence and returns a confidence score (0 = sincere, 1 = sarcastic) plus a friendly emoji. The entire pipeline—tokeniser, model, post-processing—must run locally in the user’s tab; no server calls allowed. ￼ ￼

1.2 Why sarcasm?

Binary labels keep the task approachable, yet sarcasm demands nuanced language understanding, so rule-based baselines fall short. This lets us judge your modelling chops without huge compute. ￼ ￼

1.3 Recommended ingredients

Component	Default pick	Rationale	Key spec
Base model	distilbert-base-uncased	66 M params, 6 layers, 97 % Bert-style accuracy at 40 % size	~250 MB FP32 → ≤20 MB INT8
Datasets	Sarcasm Corpus V2, iSarcasm, News-Headlines	Balanced labels, permissive licences	~9 k – 20 k samples
Exporter	Optimum CLI export onnx --quantize dynamic	One-command ONNX + quantisation	18–22 MB ONNX
Browser runtime	(A) Transformers.js pipeline → WASM/WebGPU, or (B) ONNX Runtime Web with WebGPU EP	Both support INT8; WebGPU halves latency on modern laptops	<150 ms on desktop
Frontend	React + Vite	Instant HMR dev-server, zero-config static build	Bundle ≤1 MB gz



⸻

2 Deliverables
	1.	Public Git repo with reproducible code and a README.md detailing:
	•	dataset(s) used and licence links
	•	training hyper-parameters
	•	final metrics (F1, accuracy)
	2.	model.onnx ≤ 25 MB and tokenizer.json committed under /public/model.
	3.	Browser demo at <your-url>/sarcasm/ with:
	•	textarea input
	•	sarcasm probability bar ± emoji
	•	latency log in console
	4.	Short screencast (<2 min, GIF or MP4) or live URL proving offline mode—reload with Wi-Fi disabled.

⸻

3 Timeline & workload

Phase	Suggested effort	Deadline
Kick-off & environment setup	½ day	Day 1
Data wrangling & EDA	½ day	Day 2
Fine-tune & evaluate	1 day	Day 3
Quantise & export	½ day	Day 4
Front-end integration & polish	1 day	Day 5
Buffer, screencast & submit	½ day	Day 6



⸻

4 Step-by-step guide

4.1 Environment

conda create -n sarcasm python=3.10
conda activate sarcasm
pip install "transformers>=4.40" datasets evaluate accelerate \
           "optimum[onnxruntime,gpu]" onnxruntime-web==1.17.0 \
           @huggingface/transformers        # for JS side

ONNX Runtime Web ≥ 1.17 adds official WebGPU support. ￼

4.2 Load data

from datasets import load_dataset
ds = load_dataset("Orbay/sarcasm_corpus_v2", "default")  # or your choice

Filter/clean as needed; keep a 10 % hold-out test split. ￼

4.3 Fine-tune with Trainer

Follow the Hugging Face sequence-classification recipe; two epochs, learning-rate 2 e-5, batch 16 on a single GPU hits F1 ≈ 0.82. ￼ ￼

4.4 Checkpoint & metrics

Save the best checkpoint (save_total_limit=1) and log F1/accuracy with evaluate for reproducibility. ￼

4.5 Export & quantise

optimum-cli export onnx --model path/to/ckpt onnx/ --quantize dynamic

Dynamic INT8 squeezes DistilBERT to ≈ 18 MB with negligible accuracy loss. ￼

4.6 Browser deployment

Option A — Transformers.js

import { pipeline } from '@huggingface/transformers';
const clf = await pipeline('text-classification', '/public/model', { quantized: true });
const { label, score } = (await clf(userInput))[0];

Transforms.js auto-detects WebGPU when enabled. ￼ ￼

Option B — ONNX Runtime Web

import * as ort from 'onnxruntime-web';
const sess = await ort.InferenceSession.create('/model.onnx',
           { executionProviders: ['webgpu', 'wasm'] });

WebGPU cuts inference nearly in half vs pure WASM. ￼

4.7 UI polish
	•	Progress bar coloured by score (score > 0.5 ? 🫠 : 🙂).
	•	Optional token heat-map via attention weights if you fancy.
	•	Build static site:

npm run build             # Vite

and deploy to GitHub Pages or Vercel. ￼ ￼

⸻

5 Evaluation rubric

Category	Threshold (pass)	Bonus
Accuracy	F1 ≥ 0.80	F1 > 0.85 or multilingual
Model size	≤ 25 MB	≤ 10 MB (4-bit / pruning)
Latency	≤ 150 ms on desktop	≤ 200 ms on mid-tier phone
UX	clear score + emoji	token heat-map, PWA offline
Code	reproducible scripts	CI pipeline, Dockerfile



⸻

6 Submission checklist
	•	Repo pushed, public or add us as collaborators
	•	README.md explains data licences & commands
	•	model.onnx + tokenizer.json in /public/model
	•	npm run build artefacts committed (or live url)
	•	Screencast or hosted demo link provided

⸻

7 Further reading
	•	Sarcasm Corpus V2 dataset card  ￼
	•	iSarcasm paper & dataset  ￼
	•	DistilBERT model card  ￼
	•	Optimum ONNX export guide  ￼
	•	ONNX Runtime WebGPU announcement  ￼
	•	Quantisation docs (8-bit)  ￼
	•	Transformers.js pipeline docs  ￼
	•	HF Trainer sequence-classification tutorial  ￼
	•	Vite static deployment guide  ￼
	•	Vercel deployment overview  ￼

Good luck—show us your wit-detecting wizardry!
