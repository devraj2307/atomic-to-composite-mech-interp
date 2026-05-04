import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob
import math
import torch
torch.cuda.set_per_process_memory_fraction(0.4, device=0)

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import GRPOTrainer, GRPOConfig

import upt as reward_module
from upt import reward_fn


SFT_MODEL_PATH        = ""
DATASET_PATH          = ""
OUTPUT_DIR            = "./qwen-1.5b-rl-comp-36k-upt-36k"
NUMBER_OF_DATA_POINTS = 36000
REWARD_THRESHOLDS     = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def format_prompt(example, tokenizer):
    instruction = example.get("instruction", "Answer the following question.")
    question    = example.get("question", "")
    user_text   = instruction + "\n" + question

    messages = [
        {"role": "system", "content": "You are a helpful and logical assistant."},
        {"role": "user",   "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

class StepCounterCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        reward_module.CURRENT_STEP = state.global_step


class RewardThresholdCallback(TrainerCallback):
    def __init__(self, output_dir, thresholds):
        self.output_dir  = output_dir
        self.remaining   = sorted(thresholds, reverse=True)
        self.last_reward = None
        self._trainer    = None

        ckpt_dir = os.path.join(output_dir, "reward_checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"\n[REWARD] Watching reward thresholds: {sorted(thresholds)}")

    def set_trainer(self, trainer):
        self._trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        reward = logs.get("reward") or logs.get("rewards/mean")
        if reward is None:
            return

        self.last_reward = reward

        if state.global_step % 10 == 0:
            print(f"[REWARD] step {state.global_step:>6} | mean_reward {reward:.4f}")

        fired = []
        for threshold in self.remaining:
            if reward >= threshold:
                self._save_snapshot(threshold, reward, state.global_step)
                fired.append(threshold)
        for t in fired:
            self.remaining.remove(t)

    def _save_snapshot(self, threshold, actual_reward, step):
        if self._trainer is None:
            return
        name = f"threshold-{threshold:.2f}-step{step}-actual-{actual_reward:.4f}"
        path = os.path.join(self.output_dir, "reward_checkpoints", name)
        print(f"\n[REWARD] Mean reward {actual_reward:.4f} crossed {threshold:.2f}! Saving -> {path}\n")
        self._trainer.save_model(path)
        self._trainer.tokenizer.save_pretrained(path)

def find_last_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda p: int(p.split("-")[-1]), reverse=True)
    for ckpt in checkpoints:
        if os.path.exists(os.path.join(ckpt, "trainer_state.json")):
            print(f"\n[RESUME] Resuming from: {ckpt}\n")
            return ckpt
    return None


last_checkpoint = find_last_checkpoint(OUTPUT_DIR)
load_from       = last_checkpoint if last_checkpoint else SFT_MODEL_PATH

print(f"Loading model from: {load_from}")
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

im_end_token_id        = tokenizer.convert_tokens_to_ids("<|im_end|>")
tokenizer.eos_token    = "<|im_end|>"
tokenizer.eos_token_id = im_end_token_id
print(f"EOS token set to: <|im_end|> (id={im_end_token_id})")

model = AutoModelForCausalLM.from_pretrained(
    load_from,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
model.config.use_cache = False

raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = raw_dataset.filter(
    lambda x: x.get("question", "").strip() != "" and x.get("answer", "").strip() != ""
)
dataset = dataset.shuffle(seed=42).select(range(48800, 72000))
print(f"Sampled {len(dataset)} examples for RL.")
dataset = dataset.map(lambda x: {"prompt": format_prompt(x, tokenizer)})

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,

    num_generations=8,
    per_device_train_batch_size=32,

    max_completion_length=128,

    learning_rate=7e-6,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": 5e-7},
    warmup_ratio=0.05,
    beta=0.001,

    num_train_epochs=2,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,

    bf16=True,
    logging_steps=10,
    report_to="none",
    gradient_checkpointing=True,

)

steps_per_epoch = math.ceil(len(dataset) / grpo_config.per_device_train_batch_size)
T_total         = steps_per_epoch * grpo_config.num_train_epochs * grpo_config.num_generations


print(f"\n[REWARD SCHEDULE]")
print(f"  Dataset size       : {len(dataset)}")
print(f"  Batch size         : {grpo_config.per_device_train_batch_size}")
print(f"  Steps per epoch    : {steps_per_epoch}")
print(f"  Total epochs       : {grpo_config.num_train_epochs}")
print(f"  T (total steps)    : {T_total}")

reward_cb  = RewardThresholdCallback(output_dir=OUTPUT_DIR, thresholds=REWARD_THRESHOLDS)
step_cb    = StepCounterCallback()

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=reward_fn,
    callbacks=[reward_cb, step_cb],
)

model.generation_config.do_sample     = True
model.generation_config.temperature   = 1.0
model.generation_config.top_p         = 0.95
model.generation_config.max_new_tokens = 128
model.generation_config.pad_token_id  = tokenizer.pad_token_id
model.generation_config.eos_token_id  = tokenizer.eos_token_id

reward_cb.set_trainer(trainer)

print("\nStarting GRPO Training...\n")
trainer.train(resume_from_checkpoint=last_checkpoint)

final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)