# Domain Adaptive Conversational Model

---

## Project Overview

This project fine-tunes a **T5-Large** language model to build a domain-adaptive conversational agent for legal-tech. The agent is designed to understand tone, maintain contextual awareness across turns, and transfer to multiple business use-cases within legal workflows.

Key capabilities

* Domain-specific language understanding for legal documents and conversational tone.
* Context-controlled decoding to manage turn-level and conversation-level context.
* Efficient fine-tuning using PEFT / LoRA for faster experimentation & smaller checkpoint sizes.

---

## Tech stack

* Python
* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* SpaCy, NLTK (preprocessing)
* Optional: Gradio / FastAPI for demo/serving

---

## Repository structure

```
Domain-Adaptive-Conversational-Model/
├─ data/
│  ├─ raw/                       # original domain-specific legal text files (txt, jsonl)
│  ├─ processed/                 # cleaned and tokenized dataset ready for training
│  └─ README.md                  # notes about dataset source & license
├─ src/
│  ├─ preprocessing.py           # scripts to clean, segment, and create prompt templates
│  ├─ dataset.py                 # HF Dataset conversion & DataCollator
│  ├─ train.py                   # main training / fine-tuning loop (HF Trainer or custom)
│  ├─ finetune_peft.py           # PEFT/LoRA training wrapper
│  ├─ eval.py                    # evaluation scripts (ROUGE/BLEU and custom metrics)
│  ├─ inference.py               # single-shot and multi-turn inference utilities
│  ├─ decode.py                  # context-controlled decoding utilities
│  └─ utils.py                   # helpers (logging, seeding, config parsing)
├─ demos/
│  ├─ app.py                     # Gradio / FastAPI demo app for local testing
│  └─ example_prompts.md         # example prompts & templates
├─ outputs/                      # model checkpoints and logs (gitignored)
├─ requirements.txt
├─ experiments.md                # notes on hyperparams, experiments, ablations
├─ README.md                     # this file
└─ tests/
   ├─ test_preprocessing.py
   └─ test_inference.py
```

---

## Quickstart (local)

> These commands assume a Unix-like shell. Replace paths and env variables as needed.

### 1. Create environment & install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Prepare dataset

Put your domain legal data into `data/raw/` (jsonl / txt). Run the preprocessing pipeline:

```bash
python src/preprocessing.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --tokenizer_name t5-large \
  --max_source_length 512 \
  --max_target_length 256
```

This will:

* Clean & normalize text (SpaCy, NLTK)
* Apply prompt templates (system + user + assistant framing)
* Save processed splits compatible with `datasets` or plain JSONL

### 3. Convert to HF Dataset (optional)

```bash
python src/dataset.py --input data/processed --output datasets/legal_dataset --split_ratio 0.9,0.05,0.05
```

### 4. Fine-tune (supervised) — vanilla HF Trainer

```bash
python src/train.py \
  --model_name_or_path t5-large \
  --dataset_path datasets/legal_dataset \
  --output_dir outputs/t5-legal-finetuned \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --max_source_length 512 \
  --max_target_length 256 \
  --fp16 \
  --save_strategy epoch
```

### 5. Fine-tune with PEFT / LoRA (recommended)

```bash
python src/finetune_peft.py \
  --base_model t5-large \
  --dataset_path datasets/legal_dataset \
  --output_dir outputs/t5-legal-lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --learning_rate 3e-4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --fp16
```

Notes:

* PEFT/LoRA drastically reduces trainable params and checkpoint sizes.
* Use `accelerate` for multi-GPU or mixed-precision training.

### 6. Inference / Context-controlled decoding

Single-turn inference:

```bash
python src/inference.py \
  --model_dir outputs/t5-legal-lora \
  --prompt "User: Draft a non-disclosure clause for a consulting agreement." \
  --max_new_tokens 256 \
  --temperature 0.2 \
  --top_p 0.95
```

Multi-turn (preserve history & control tone):

```bash
python src/decode.py \
  --model_dir outputs/t5-legal-lora \
  --history_file examples/conv_history.jsonl \
  --context_window 1024 \
  --tone professional \
  --max_new_tokens 200
```

`decode.py` implements the logic for context truncation, system prompts for tone, and decoding strategies (beam search / sampling).

### 7. Evaluation

Run automated metrics (ROUGE / BLEU) and run a small human-review test-set for style & factuality checks.

```bash
python src/eval.py \
  --model_dir outputs/t5-legal-lora \
  --dataset_path datasets/legal_dataset/test.jsonl \
  --metrics rouge,bleu
```

Additionally, create a small human evaluation spreadsheet to judge tone, appropriateness, and hallucination risk.

---

## Running the demo (optional)

Start a local demo (Gradio):

```bash
python demos/app.py --model outputs/t5-legal-lora --port 7860
```

Open `http://localhost:7860` and interact with the model. The demo includes toggles for tone, max tokens, and whether to include full history.

---

## Tests & linting

Run unit tests:

```bash
pytest -q
```

Format and lint:

```bash
black .
flake8 src tests
```

---

## Tips & Best Practices

* Use small smoke-run datasets while iterating.
* Track experiments with `wandb` or HF Hub.
* When deploying to production, add explicit guardrails (RLHF, rule-based filters) for legal advice disclaimers and hallucination mitigation.
* Always record provenance (source doc, extraction time) for training data to respect privacy / licensing.

---

## License & Attribution

Specify project license (e.g., MIT) and include attributions for any public corpora used.
