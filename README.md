# dialog-finetuning-50k

**Full supervised fine-tuning (SFT)** of `Langboat/bloom-389m-zh` on
~50 K human/assistant dialog turns sampled from the YeungNLP
simplified-Chinese corpus. All 389 M parameters are updated — no LoRA,
no quantisation. This is the most straightforward SFT setup in the
project.

---

## Table of Contents

- [What This Does](#what-this-does)
- [Model & Dataset](#model--dataset)
- [Training Setup](#training-setup)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Key Files](#key-files)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Reproducing Training](#reproducing-training)
- [Running Inference](#running-inference)
- [Notes](#notes)
- [Files Not in Repo](#files-not-in-repo)
- [References](#references)
- [License](#license)

---

## What This Does

Two Jupyter notebooks form an end-to-end full-SFT pipeline for a
Chinese dialogue model:

1. Sample and tokenise ~50 K turns from the YeungNLP dialog corpus.
2. Full-parameter SFT of `Langboat/bloom-389m-zh` for 1 epoch on
   Google Colab (T4, ~12 GB VRAM).

The result is a small (~389 M) BLOOM-family model that can hold
multi-turn conversations in Simplified Chinese. It is intended as a
counterpart to the sibling `qlora-dialog-100k` repo — same base family,
same task, but full-SFT instead of parameter-efficient QLoRA — to
highlight the training-cost / VRAM / forgetting trade-offs.

---

## Model & Dataset

| Field         | Value |
| ------------- | ----- |
| Base model    | `Langboat/bloom-389m-zh` |
| Params        | ~389 M (float32, ~1.65 GB) |
| Task          | Causal LM, multi-turn dialogue |
| Fine-tuning   | Full SFT (100 % params trainable) |
| Dataset       | YeungNLP human/assistant simplified dialog, sampled to ~50 K turns |
| Train rows    | ~48 085 |
| Val rows      | ~2 000 |
| Data format   | HuggingFace Arrow (`data_train/`, `data_val/`) |
| Tokenizer     | `BloomTokenizerFast` from `Langboat/bloom-389m-zh` |

**Prompt template used during training and inference:**

```
Human: <user turn 1>
Assistant: <bot turn 1><eos>
Human: <user turn 2>
Assistant:
```

Each turn ends with the EOS token so the loss cleanly separates turns.

---

## Training Setup

Actual values (from `10-15-sft-train.ipynb`):

```python
BATCH_SIZE                  = 64
MICRO_BATCH_SIZE            = 4
GRADIENT_ACCUMULATION_STEPS = 16       # effective batch = 64

TrainingArguments(
    num_train_epochs             = 1,
    learning_rate                = 8e-6,     # conservative Belle-style LR
    warmup_ratio                 = 0.05,
    weight_decay                 = 0.00001,
    per_device_train_batch_size  = 4,
    per_device_eval_batch_size   = 8,
    save_strategy                = "steps",
    save_steps                   = 20,
    save_total_limit             = 2,        # keep only 2 most recent
    optim                        = "adamw_torch",
    output_dir                   = "my-checkpoints",
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer, return_tensors="pt", padding=True,
    label_pad_token_id = -100,               # pad excluded from loss
)
```

| Setting              | Value                        |
| -------------------- | ---------------------------- |
| Method               | Full SFT                     |
| Epochs               | 1                            |
| Effective batch      | 64 (micro 4 × grad-accum 16) |
| Learning rate        | 8e-6                         |
| Warm-up ratio        | 0.05                         |
| Weight decay         | 1e-5                         |
| Optimiser            | `adamw_torch`                |
| Precision            | fp32 base weights (Colab T4) |
| Padding side         | **left** (BLOOM requirement) |
| GPU used             | Colab T4, ~12 GB VRAM        |
| Steps / epoch        | ~750                         |
| Wall time (T4)       | ~90–120 min                  |
| Final train loss     | ~1.7–1.8                     |

The final loss is *higher* than the ABSA experiment (~0.13) because the
50 K general-dialog corpus has much more varied patterns — a healthier
generalisation outcome for open-domain chat.

### Full-SFT vs. QLoRA (sibling repo `qlora-dialog-100k`)

| Property                     | Full SFT 50 K (this) | QLoRA 100 K |
| ---------------------------- | -------------------- | ----------- |
| Trainable params             | 389 M (100 %)        | ~25 M (~6.7 %) |
| VRAM for model               | ~1.65 GB (fp32)      | ~317 MB (4-bit) |
| Training time                | ~90–120 min          | ~2–3 h      |
| Catastrophic-forgetting risk | Moderate             | Low         |

---

## System Architecture

```
                ┌────────────────────────────────────────┐
                │ YeungNLP human/assistant dialog corpus │
                │  (100 K+ raw turns)                    │
                └───────────────┬────────────────────────┘
                                │  sampling + Human/Assistant format
                                │  + <eos>
                                ▼
                ┌────────────────────────────────────────┐
                │ 10-10-dialog-data-sampling-            │
                │   preprocessing.ipynb                  │
                │  - BloomTokenizerFast                  │
                │  - drop overlong, dedup                │
                └───────────────┬────────────────────────┘
                                │  HuggingFace Arrow
                                ▼
                ┌────────────────────────────────────────┐
                │ train_dataset_YeungNLP_traditional/    │
                │   ├─ data_train/  (~48 K rows)         │
                │   └─ data_val/    (~2 K rows)          │
                └───────────────┬────────────────────────┘
                                │
                                ▼
     ┌─────────────────────────────────────────────────────┐
     │ 10-15-sft-train.ipynb                               │
     │  - AutoModelForCausalLM(Langboat/bloom-389m-zh)     │
     │  - DataCollatorForSeq2Seq (label pad = -100)        │
     │  - TrainingArguments(1 ep, lr=8e-6, eff. batch=64,  │
     │                      warmup 5 %, save every 20 st.) │
     │  - Trainer.train() → my-checkpoints/                │
     │  - model.save_pretrained("my-bloom-sft-50k")        │
     └───────────────┬─────────────────────────────────────┘
                     │  ~750 steps, ~90–120 min on T4
                     ▼
     ┌─────────────────────────────────────────────────────┐
     │ my-bloom-sft-50k/  (BLOOM checkpoint, ~1.5 GB)      │
     └───────────────┬─────────────────────────────────────┘
                     │
                     ▼
     Served by the sibling Gradio UI in `dialog-multi-epoch-experiments`
     (`app.py` / `chat_streaming.py`) — just point `model_name_or_path`
     at this checkpoint.
```

---

## Repository Layout

```
dialog-finetuning-50k/
├── 10-10-dialog-data-sampling-preprocessing.ipynb   # Step 1: sample + tokenise
├── 10-15-sft-train.ipynb                             # Step 2: full-SFT training
├── notebooks/
│   ├── 10-10-dialog-data-sampling-preprocessing.ipynb   # mirror
│   └── 10-15-sft-train.ipynb                             # mirror
├── requirements.txt
├── README.md
└── train_dataset_YeungNLP_traditional/               # (created by Step 1, gitignored)
    ├── data_train/
    └── data_val/
```

---

## Key Files

| File | Purpose |
| ---- | ------- |
| `10-10-dialog-data-sampling-preprocessing.ipynb` | **Step 1.** Loads the raw YeungNLP dialog corpus (100 K+ turns), samples ~50 K, formats each into a `Human: … Assistant: … <eos>` template, tokenises with `BloomTokenizerFast.from_pretrained('Langboat/bloom-389m-zh')`, and saves as HuggingFace Arrow to `train_dataset_YeungNLP_traditional/data_train` + `data_val`. |
| `10-15-sft-train.ipynb` | **Step 2.** Loads the Arrow dataset, instantiates `AutoModelForCausalLM.from_pretrained('Langboat/bloom-389m-zh', torch_dtype='auto')`, sets up `TrainingArguments` (see above), runs `Trainer.train()`, and saves the final model. Supports `resume_from_checkpoint=True` for Colab reconnects. |
| `notebooks/*.ipynb` | Mirrors of the two step notebooks kept under a `notebooks/` folder for clarity. |
| `requirements.txt` | Minimal training deps: `torch>=2.0`, `transformers>=4.32`, `datasets>=2.14`, `accelerate>=0.23`, `pandas`, `numpy`, `tqdm`. |

There is no `app.py` / `demo.py` in this repo — inference happens in the
sibling `dialog-multi-epoch-experiments` repo's Gradio UI by pointing
its `model_name_or_path` at the checkpoint produced here.

---

## Requirements

From `requirements.txt`:

```
torch>=2.0.0
transformers>=4.32.0
datasets>=2.14.0
accelerate>=0.23.0
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
```

Hardware / platform:

- Python 3.10+, CUDA 11.8+ recommended
- **Google Colab T4 (~12 GB VRAM)** is enough for the settings above
- ~16 GB system RAM
- For local training: any single CUDA GPU with ≥ 12 GB VRAM
- CPU-only training is not practical for a 389 M model at this scale

Install:

```bash
pip install -r requirements.txt
```

---

## Configuration

There are no env vars or YAML configs — hyperparameters live inline at
the top of `10-15-sft-train.ipynb`:

```python
BATCH_SIZE                  = 64
MICRO_BATCH_SIZE            = 4
GRADIENT_ACCUMULATION_STEPS = 16

# TrainingArguments — see full block above
```

If you have less than 12 GB VRAM, lower `MICRO_BATCH_SIZE` (e.g. to 2)
and raise `GRADIENT_ACCUMULATION_STEPS` proportionally to keep the
effective batch at 64.

---

## Reproducing Training

```bash
# 1. Clone
git clone https://github.com/Elainedu/dialog-finetuning-50k.git
cd dialog-finetuning-50k

# 2. Install deps
pip install -r requirements.txt

# 3. Step 1 — prepare the dataset
jupyter notebook 10-10-dialog-data-sampling-preprocessing.ipynb
#    Produces  train_dataset_YeungNLP_traditional/data_train + data_val

# 4. Step 2 — full-SFT training
jupyter notebook 10-15-sft-train.ipynb
#    Writes checkpoints to  my-checkpoints/
#    Save the final model with:
#        model.save_pretrained("my-bloom-sft-50k")
```

Runtime on Colab T4: ~90–120 min for 1 epoch (~750 steps).

If Colab disconnects mid-training, restart the runtime, re-run the
setup cells, and continue with:

```python
trainer.train(resume_from_checkpoint=True)
```

---

## Running Inference

This repo does **not** ship a Gradio UI. Two options:

**Option A — reuse the sibling Gradio UI** in
`dialog-multi-epoch-experiments`:

```bash
cd ../dialog-multi-epoch-experiments
# edit app.py:  model_name_or_path = "../dialog-finetuning-50k/my-bloom-sft-50k"
python app.py
```

**Option B — quick programmatic test:**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "my-bloom-sft-50k"
tok  = AutoTokenizer.from_pretrained(path)
mdl  = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype="auto", device_map="auto"
)

prompt = "Human: Hello, please introduce yourself\nAssistant:"
ids    = tok(prompt, return_tensors="pt").to(mdl.device)
out    = mdl.generate(**ids, max_new_tokens=200, do_sample=True,
                      temperature=1.0, top_p=0.95, top_k=200,
                      repetition_penalty=1.2)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## Notes

- **`my-checkpoints/` cleanup.** The original output directory was moved
  under `_empty-folders/` during repo cleanup — no permanent checkpoints
  were kept there in this backup copy. Retrain to produce fresh weights.
- **Padding side.** BLOOM tokenizers default to **left** padding for
  causal LM. Do not flip it to right.
- **Epoch count.** 1 epoch is generally sufficient for open-domain
  conversational fine-tuning at this data volume. Running 2 epochs may
  begin to over-train / cause repetition.
- **`torch_dtype='auto'`.** Loads the base at its stored dtype (fp32 on
  CPU/T4 for this checkpoint). Switch to `torch.float16` / `torch.bfloat16`
  if your GPU supports it and you want to shrink VRAM usage further.
- **Legacy launcher name.** Some earlier notes reference `gradio.py` as
  the demo entry point — the current Gradio front-end lives in the
  sibling repo as `app.py`.

---

## Files Not in Repo

| Excluded                                    | Size    | How to obtain / regenerate |
| ------------------------------------------- | ------- | -------------------------- |
| `train_dataset_YeungNLP_traditional/`       | ~50 MB  | Run `10-10-dialog-data-sampling-preprocessing.ipynb`. |
| `my-checkpoints/` intermediate checkpoints  | ~1.5 GB each | Rerun `10-15-sft-train.ipynb`. |
| Final `my-bloom-sft-50k/` checkpoint        | ~1.5 GB | `model.save_pretrained("my-bloom-sft-50k")` at end of Step 2. |
| Raw YeungNLP dialog source                  | varies  | Download from Hugging Face — e.g. <https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M>. |
| `Langboat/bloom-389m-zh` base weights       | ~1.65 GB| Auto-downloaded by `transformers` on first run. |

---

## References

- Le Scao et al. **BLOOM: A 176B-Parameter Open-Access Multilingual
  Language Model.** arXiv:2211.05100. <https://arxiv.org/abs/2211.05100>
- BELLE project (LR / data recipe inspiration):
  <https://github.com/LianjiaTech/BELLE>
- Base model: <https://huggingface.co/Langboat/bloom-389m-zh>
- YeungNLP model / dataset collection:
  <https://huggingface.co/YeungNLP>
- HuggingFace `Trainer`:
  <https://huggingface.co/docs/transformers/main_classes/trainer>
- `DataCollatorForSeq2Seq`:
  <https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForSeq2Seq>

---

## License

Educational use only. Base model weights and dataset follow their
upstream licenses (BLOOM RAIL License for BLOOM-derived checkpoints).
Notebooks and configuration in this repository may be freely used,
modified, and redistributed for teaching and research.
