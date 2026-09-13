import os, json, time
import torch
from datasets import load_dataset, Dataset
from unsloth import FastModel
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

os.environ["HF_TOKEN"] = open("/root/.hf_token").read().strip()

MAX_SEQ, N_ROWS, STEPS = 2048, 6000, 240
OUT = f"/root/agent-native-xlam-unsloth-{time.strftime('%m%d-%H%M')}"

print("step1: unsloth FastModel BF16 (cached base)", flush=True)
t0 = time.time()
model, _tok = FastModel.from_pretrained(
    model_name="Qwen/Qwen3.8-27B",   # cached BF16 -> no downloads
    max_seq_length=MAX_SEQ,
    load_in_4bit=False,
    full_finetuning=False,
)
print(f"  loaded in {round(time.time()-t0)}s", flush=True)
# FastModel tokenizer is a VLM processor whose apply_chat_template breaks on plain
# strings ("Incorrect image source ... <|im_start|>system"). Use plain AutoTokenizer
# for data rendering + collator (same vocab/template -> training labels consistent).
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407, use_rslora=False, loftq_config=None,
)

print("step2: xLAM slice", flush=True)
ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train", streaming=True)
rows = []
for i, r in enumerate(ds):
    rows.append(r)
    if i >= N_ROWS - 1:
        break
print(f"  rows: {len(rows)}", flush=True)

def fmt(row):
    tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    ans = row["answers"] if isinstance(row["answers"], str) else json.dumps(row["answers"])
    user_c = ("You are a helpful assistant with access to functions. When a tool is needed, reply with ONLY "
              "a JSON list: [{\"name\": \"<fn>\", \"arguments\": {...}}].\nFunctions:\n" + json.dumps(tools) +
              "\n\nQuery: " + row["query"])
    return [{"role": "user", "content": user_c}, {"role": "assistant", "content": ans}]

def tok_fn(batch):
    texts = [tokenizer.apply_chat_template(fmt(r), tokenize=False, add_generation_prompt=False) for r in batch["_r"]]
    enc = tokenizer(texts, truncation=True, max_length=MAX_SEQ, padding=False)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": enc["input_ids"]}

data = Dataset.from_dict({"_r": rows}).map(tok_fn, batched=True, num_proc=4, remove_columns=["_r"])
print("  tokenized:", len(data), flush=True)

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir=OUT, per_device_train_batch_size=1, gradient_accumulation_steps=16,
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
print("DONE", OUT, flush=True)
