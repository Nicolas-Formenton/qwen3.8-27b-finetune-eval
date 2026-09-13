import sys, time, torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

MODEL_IN = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.8-27B"
TAG = sys.argv[2] if len(sys.argv) > 2 else "base"
MAX_SEQ = 2048
STEPS = 240
N_ROWS = 1500
LR = 2e-4
OUT = f"/root/agent-native-distill-{TAG}-{time.strftime('%m%d-%H%M')}"

print(f"== stacked distill on {MODEL_IN} (tag={TAG}) ==", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_IN, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model = model.to("cuda")
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")

names = [n for n, m in model.named_modules() if m.__class__.__name__ == "Linear"]
targets = sorted({n.split(".")[-1] for n in names if n.split(".")[-1] in
                  {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}})[:8]
peft_cfg = LoraConfig(r=16, lora_alpha=16, target_modules=targets, lora_dropout=0.0, bias="none")
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

ds = load_dataset("lordx64/reasoning-distill-kimi-k2-6-max-sft", split="train", streaming=True)
rows = []
for i, r in enumerate(ds):
    rows.append(r["text"])
    if len(rows) >= N_ROWS:
        break
print("rows:", len(rows), flush=True)

def tok_f(ex):
    ids = tokenizer(ex["text"], truncation=True, max_length=MAX_SEQ)["input_ids"]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)}

tok_ds = Dataset.from_list([{"text": t} for t in rows]).map(tok_f, num_proc=1, remove_columns=["text"])
print("tokenized", flush=True)

args = TrainingArguments(output_dir=OUT, max_steps=STEPS, per_device_train_batch_size=1,
    gradient_accumulation_steps=8, learning_rate=LR, warmup_ratio=0.05, bf16=True,
    logging_steps=5, save_strategy="no", optim="adamw_torch", gradient_checkpointing=True)
trainer = Trainer(model=model, args=args, train_dataset=tok_ds,
                  data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
print("TRAINING_START", flush=True)
trainer.train()
print("SAVE_ADAPTER", flush=True)
model.save_pretrained(OUT + "-adapter")
tokenizer.save_pretrained(OUT + "-adapter")
print("DONE " + OUT, flush=True)
