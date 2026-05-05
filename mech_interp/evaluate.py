import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import string
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = ""
TEST_DATA_PATH = ""

BATCH_SIZE = 256

def format_prompt(example, tokenizer):
    user_text =  "Answer the following question."+ "\n\n" + example["question"] + "\n\n" + example["context"]
    
    messages = [
        {"role": "system", "content": "You are a helpful and logical assistant."},
        {"role": "user",   "content": user_text},
    ]
 
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def normalize_answer(s):
    if not s:
        return ""
    s = s.strip().lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s

def extract_prediction(generated_text):
    target_phrase = "so, the answer is:"
    gen_lower = generated_text.lower()

    if target_phrase in gen_lower:
        prediction = gen_lower.split(target_phrase)[1].strip()
        prediction = prediction.split("\n")[0].strip()
        prediction = prediction.replace("<|im_end|>", "").strip()
        return prediction
    return ""

print(f"Loading tokenizer and model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.chat_template = base_tokenizer.chat_template

tokenizer.eos_token    = "<|im_end|>"
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<|endoftext|>"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"EOS token id      : {tokenizer.eos_token_id}  ({tokenizer.eos_token})")
print(f"<|im_end|> token id: {im_end_token_id}")
eos_token_ids = list({tokenizer.eos_token_id, im_end_token_id})
print(f"Using EOS ids     : {eos_token_ids}")

print(f"\nLoading test data from {TEST_DATA_PATH}...")
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

data = [ex for ex in data if ex.get("question", "").strip() and ex.get("answer", "").strip()]
print(f"Loaded {len(data)} valid QA examples (skipped biography-only entries)")

results = {
    "overall":     {"correct": 0, "total": 0},
    "iid":         {"correct": 0, "total": 0},
    "composition": {"correct": 0, "total": 0},
    "zero-shot":   {"correct": 0, "total": 0}
}

print("\nStarting evaluation...\n")
debug_printed = False

for i in tqdm(range(0, len(data), BATCH_SIZE)):
    batch = data[i:i + BATCH_SIZE]
    prompts = [format_prompt(ex, tokenizer) for ex in batch]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,
        )

    input_lengths = inputs.input_ids.shape[1]
    new_tokens = outputs[:, input_lengths:]
    generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    if not debug_printed:
        print("\n===== SAMPLE MODEL OUTPUTS (first 3) =====")
        for k in range(min(3, len(generated_texts))):
            pred_raw = extract_prediction(generated_texts[k])
            print(f"\n--- Example {k+1} ---")
            print(f"QUESTION   : {batch[k].get('question', '')}")
            print(f"GENERATED  : {generated_texts[k][:300]}")
            print(f"EXTRACTED  : '{pred_raw}'")
            print(f"EXPECTED   : '{batch[k].get('answer', '')}'")
        print("===========================================\n")
        debug_printed = True

    for j, ex in enumerate(batch):
        pred_raw  = extract_prediction(generated_texts[j])
        truth_raw = ex.get("answer", "")

        pred_norm  = normalize_answer(pred_raw)
        truth_norm = normalize_answer(truth_raw)

        is_correct = (pred_norm == truth_norm)

        gen_type = ex.get("gen_type", "overall").lower().replace("_", "-")
        if gen_type not in results:
            results[gen_type] = {"correct": 0, "total": 0}

        results["overall"]["total"] += 1
        results[gen_type]["total"]  += 1

        if is_correct:
            results["overall"]["correct"] += 1
            results[gen_type]["correct"]  += 1

print("\n" + "=" * 50)
print(f"{'EVALUATION RESULTS (Exact Match)':^50}")
print("=" * 50)
for category, metrics in results.items():
    if metrics["total"] > 0:
        accuracy = (metrics["correct"] / metrics["total"]) * 100
        print(f"{category.upper():<15} : {accuracy:>6.2f}%  ({metrics['correct']}/{metrics['total']})")
print("=" * 50)
