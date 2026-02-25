# Bloom-389M Full SFT — YeungNLP Dialog 50K

## Task Overview

Full supervised fine-tuning of `Langboat/bloom-389m-zh` on a ~50K-sample Chinese dialog dataset. All model weights are updated (no LoRA). This is the most straightforward SFT setup in this project.

**Method**: Full SFT (Supervised Fine-Tuning)
**Base model**: `Langboat/bloom-389m-zh` (389M params, float32, ~1.65 GB)
**Dataset**: YeungNLP human-assistant simplified dialog, sampled to 50K turns
**Platform**: Google Colab (T4 GPU, ~12 GB VRAM)

---

## Notebooks

| File | Purpose |
|---|---|
| `10-10-dialog-data-sampling-preprocessing.ipynb` | Sample + tokenize raw dialog data |
| `10-15-sft-train.ipynb` | Main SFT training |

---

## Step 1 — Data Preprocessing

`10-10-dialog-data-sampling-preprocessing.ipynb` prepares the training data:

1. Load raw YeungNLP dialog corpus (100K+ turns)
2. Sample ~50K turns for manageability
3. Format each turn as a "Human / Assistant" template ending with EOS token
4. Tokenize with `BloomTokenizerFast` from `Langboat/bloom-389m-zh`
5. Save to disk as HuggingFace Arrow format:

```
train_dataset_YeungNLP_traditional/
├── data_train/   (train split)
└── data_val/     (validation split)
```

Dataset stats: **~48,085 train rows**, ~2K validation rows.

---

## Step 2 — SFT Training

`10-15-sft-train.ipynb` runs the actual fine-tuning:

### Model & Tokenizer

```python
tokenizer = BloomTokenizerFast.from_pretrained('Langboat/bloom-389m-zh')
model = AutoModelForCausalLM.from_pretrained('Langboat/bloom-389m-zh', torch_dtype='auto')
```

### Training Configuration (actual values from notebook)

```python
BATCH_SIZE = 64
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16   # effective batch = 64

TrainingArguments(
    num_train_epochs=1,
    learning_rate=8e-6,          # conservative Belle-style LR
    warmup_ratio=0.05,
    weight_decay=0.00001,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    save_strategy='steps',
    save_steps=20,
    save_total_limit=2,           # keep only 2 most recent checkpoints
    optim='adamw_torch',
    output_dir='my-checkpoints',
)
```

### Data Collator

```python
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    return_tensors="pt",
    padding=True,
    label_pad_token_id=-100   # pad tokens excluded from loss
)
```

Bloom uses **left-side padding** — do not change `tokenizer.padding_side`.

### Resume from Checkpoint

If Colab disconnects mid-training:
```python
trainer.train(resume_from_checkpoint=True)
```

---

## Expected Results

| Metric | Estimate |
|---|---|
| Train samples | ~48,000 |
| Effective batch size | 64 |
| Steps per epoch | ~750 |
| Wall time (T4) | ~90–120 min |
| Final train loss | ~1.7–1.8 |

Loss is higher than the ABSA-specific 15K experiment (loss 0.13) because the 50K general dialog dataset has more diverse patterns — a healthier generalization outcome.

---

## Comparison: This Experiment vs QLoRA (100K)

| Property | Full SFT 50K (this) | QLoRA 100K |
|---|---|---|
| Trainable params | 389M (100%) | 25M (6.7%) |
| VRAM for model | ~1.65 GB | ~317 MB (4-bit) |
| Training time | ~90–120 min | ~4 hours |
| Catastrophic forgetting risk | Moderate | Low |

---

## How to Run

1. Run `10-10-dialog-data-sampling-preprocessing.ipynb` to prepare the dataset
2. Verify Arrow files exist in `train_dataset_YeungNLP_traditional/data_train/` and `data_val/`
3. Open `10-15-sft-train.ipynb`
4. Adjust `MICRO_BATCH_SIZE` if you have less than 12 GB VRAM
5. Run all cells — checkpoint saved every 20 steps
6. Save the final model: `model.save_pretrained('my-bloom-sft-50k')`

---

## Notes

- The `my-checkpoints/` subfolder was originally created as the output directory. It was moved to `_empty-folders/` during cleanup (no checkpoints were saved to this backup copy).
- The saved model from training is compatible with the Gradio demo in `../my-training/` — just update the model path in `gradio.py`.
- 1 epoch is generally sufficient for conversational fine-tuning. Running 2 epochs on this dataset may cause over-training.
