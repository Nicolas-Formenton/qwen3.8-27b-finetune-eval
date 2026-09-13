import os, sys, time, json
import torch
from datasets import load_dataset, Dataset

# Order matters: importing unsloth first sets UNSLOTH_IS_PRESENT, which
# unsloth_zoo.__init__ requires or it raises "Please install Unsloth".
from unsloth import FastModel

# Belt-and-braces: fix_untrained_tokens scans the embedding matrix and crashes
# with "Cannot copy out of meta tensor" on models whose embeddings start on the
# meta device (VLM configs, tied weights). No-op it before FastModel loads.
import unsloth_zoo.tokenizer_utils
unsloth_zoo.tokenizer_utils.fix_untrained_tokens = lambda *a, **k: (None, None)

from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

MODEL_IN = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.8-27B"
TAG = sys.argv[2] if len(sys.argv) > 2 else "base"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 240
N_ROWS = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
MAX_SEQ = 2048
OUT = f"/root/agent-native-distill-{TAG}-{time.strftime('%m%d-%H%M')}"

print(f"== unsloth stacked distill on {MODEL_IN} (tag={TAG}, steps={STEPS}, rows={N_ROWS}) ==", flush=True)
t0 = time.time()
model, _tok = FastModel.from_pretrained(
    model_name=MODEL_IN,
    max_seq_length=MAX_SEQ,
    load_in_4bit=False,
    full_finetuning=False,
)
print(f"  loaded in {round(time.time()-t0)}s", flush=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407, use_rslora=False, loftq_config=None,
)

print("step2: distill slice", flush=True)
ds = load_dataset("lordx64/reasoning-distill-kimi-k2-6-max-sft", split="train", streaming=True)
rows = []
for i, r in enumerate(ds):
    rows.append(r["text"])
    if len(rows) >= N_ROWS:
        break
print(f"  rows: {len(rows)}", flush=True)

def tok_fn(batch):
    enc = tokenizer(batch["_t"], truncation=True, max_length=MAX_SEQ, padding=False)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": enc["input_ids"]}

data = Dataset.from_dict({"_t": rows}).map(tok_fn, batched=True, num_proc=4, remove_columns=["_t"])
print("  tokenized:", len(data), flush=True)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir=OUT, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        max_steps=STEPS, learning_rate=2e-4, warmup_steps=10, bf16=True, logging_steps=5, save_steps=100,
        report_to="none", dataloader_drop_last=True, optim="adamw_torch", gradient_checkpointing=True),
    train_dataset=data,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
print("TRAINING_START", flush=True)
trainer.train()
print("SAVE_ADAPTER", flush=True)
model.save_pretrained(OUT + "-adapter")
tokenizer.save_pretrained(OUT + "-adapter")
print("DONE " + OUT, flush=True)
