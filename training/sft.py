import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import glob
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from torch.utils.data import DataLoader

MODEL_ID     = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "Parametric_and_Contextual_training.json"
OUTPUT_DIR   = "./qwen-0.5b-instruct-sft"

LOSS_THRESHOLDS = [0.05, 0.01]

def format_chatml(example):
    instruction = example.get("instruction", "Answer the following question.")
    user_text   = instruction + "\n" + example["question"]

    assistant_text = example.get("answer_cot", "")

    messages = [
        {"role": "system",    "content": "You are a helpful and logical assistant."},
        {"role": "user",      "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    return {"messages": messages}


def find_last_checkpoint(output_dir):
    REQUIRED_META = ["trainer_state.json", "training_args.bin"]
    WEIGHT_FILES  = [
        "model.safetensors", "pytorch_model.bin",
        "adapter_model.safetensors", "adapter_model.bin",
    ]

    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        print("[RESUME] No checkpoints found — starting from scratch.\n")
        return None

    checkpoints = sorted(checkpoints, key=lambda p: int(p.split("-")[-1]), reverse=True)
    print(f"\n[RESUME] Found {len(checkpoints)} checkpoint(s). Validating...")

    for ckpt in checkpoints:
        step    = ckpt.split("-")[-1]
        missing = [f for f in REQUIRED_META if not os.path.exists(os.path.join(ckpt, f))]
        if missing:
            print(f"[RESUME]   step-{step}: INCOMPLETE (missing {missing}), skipping.")
            continue
        if not any(os.path.exists(os.path.join(ckpt, w)) for w in WEIGHT_FILES):
            print(f"[RESUME]   step-{step}: INCOMPLETE (no weight file), skipping.")
            continue
        print(f"[RESUME]   step-{step}: VALID")
        print(f"[RESUME] Resuming from: {ckpt}\n")
        return ckpt

    print("[RESUME] No valid checkpoints — starting from scratch.\n")
    return None

class LossThresholdCallback(TrainerCallback):

    def __init__(self, output_dir, thresholds):
        self.output_dir  = output_dir
        
        self.thresholds  = sorted(thresholds, reverse=True)
        self.remaining   = sorted(thresholds, reverse=True)  
        self.last_loss   = None
        self._trainer    = None

        snap_dir = os.path.join(output_dir, "loss_checkpoints")
        os.makedirs(snap_dir, exist_ok=True)
        print(f"\n[THRESHOLD] Watching loss thresholds : {sorted(thresholds)}")
        print(f"[THRESHOLD] Snapshots -> {snap_dir}\n")

    def set_trainer(self, trainer):
        self._trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss")
        if loss is None:
            return

        self.last_loss = loss

        if state.global_step % 100 == 0:
            remaining_str = str([f"{t:.2f}" for t in sorted(self.remaining)])
            print(
                f"[LOSS] step {state.global_step:>6} | "
                f"loss {loss:.4f} | "
                f"remaining thresholds: {remaining_str}"
            )

      
        fired = []
        for threshold in self.remaining:
            if loss <= threshold:
                self._save_snapshot(threshold, loss, state.global_step)
                fired.append(threshold)

        for t in fired:
            self.remaining.remove(t)

        if not self.remaining:
            print("\n[THRESHOLD] All thresholds reached. Continuing to end of training.\n")

    def _save_snapshot(self, threshold, actual_loss, step):
        if self._trainer is None:
            print(f"[THRESHOLD] Trainer not set — cannot save snapshot for {threshold}")
            return

        name = (
            f"threshold-{threshold:.2f}"
            f"-step{step}"
            f"-actual-{actual_loss:.4f}"
        )
        path = os.path.join(self.output_dir, "loss_checkpoints", name)

        print(f"\n[THRESHOLD] Loss {actual_loss:.4f} crossed threshold {threshold:.2f}")
        print(f"[THRESHOLD] Saving -> {path}")
        self._trainer.save_model(path)
        self._trainer.tokenizer.save_pretrained(path)
        print(f"[THRESHOLD] Saved.\n")

last_checkpoint = find_last_checkpoint(OUTPUT_DIR)
load_from       = last_checkpoint if last_checkpoint else MODEL_ID
print(f"Loading model from: {load_from}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    load_from,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False

print("Loading and formatting dataset...")

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.shuffle(seed=42)
dataset = dataset.map(format_chatml)
dataset = dataset.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(
            x["messages"], tokenize=False, add_generation_prompt=False,
            add_special_tokens=False
        )
    }
)
dataset = dataset.filter(
    lambda x: x["text"] is not None and len(x["text"]) > 0
)

dataset = dataset.shuffle(seed=42)

print(f"Total training examples : {len(dataset)}")
print("\nSample (first 500 chars):")
print(dataset[0]["text"][:500])

response_template_ids = tokenizer.encode(
    "<|im_start|>assistant", 
    add_special_tokens=False
)

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
)

print("\nRunning collator sanity check...")

def tokenize_for_check(example):
    return tokenizer(example["text"], truncation=True, max_length=2048)

tokenized_check = dataset.select(range(4)).map(tokenize_for_check)
tokenized_check = tokenized_check.remove_columns(
    [c for c in tokenized_check.column_names if c not in ("input_ids", "attention_mask")]
)
sample_batch = next(iter(DataLoader(tokenized_check, batch_size=4, collate_fn=collator)))
num_unmasked = (sample_batch["labels"] != -100).sum().item()
print(f"[COLLATOR] Unmasked label tokens: {num_unmasked}")

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,

    learning_rate=3e-4,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": 3e-5},
    warmup_ratio=0.05,

    num_train_epochs=14,
    max_steps=-1,

    eval_strategy="no",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,        

    weight_decay=0.01,        

    bf16=True,
    logging_steps=10,
    optim="adamw_torch_fused",
    report_to="none",
)


threshold_cb = LossThresholdCallback(
    output_dir=OUTPUT_DIR,
    thresholds=LOSS_THRESHOLDS,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=collator,
    callbacks=[threshold_cb],
    dataset_kwargs={"add_special_tokens": False}
)


threshold_cb.set_trainer(trainer)

steps_per_epoch = len(dataset) // (4 * 32)
total_steps     = steps_per_epoch * 16

print("Starting training...")
print(f"Resume          : {'yes — ' + last_checkpoint if last_checkpoint else 'no — fresh start'}\n")

trainer.train(resume_from_checkpoint=last_checkpoint)

final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print("\n" + "=" * 60)
print("Training complete!")
print(f"Final model      : {final_model_dir}")
print(f"Loss checkpoints : {os.path.join(OUTPUT_DIR, 'loss_checkpoints')}")
