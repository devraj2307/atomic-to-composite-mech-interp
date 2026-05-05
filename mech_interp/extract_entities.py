import json
import argparse
from pathlib import Path


def load_entity_set(entity_file: str) -> set:
    with open(entity_file, "r", encoding="utf-8") as f:
        return set(json.load(f))


def extract_entities_from_cot(cot: str, entity_set: set) -> list:
    """
    Scan the answer_cot string for any entity in the global entity set.
    """
    cot_lower = cot.strip().lower()
    return sorted(e for e in entity_set if e in cot_lower)


def annotate_training_file(input_path: str, entity_file: str, output_path: str):
    entity_set = load_entity_set(entity_file)
    print(f"[INFO] Loaded {len(entity_set)} entities from {entity_file}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    total       = len(data)
    zero_entity = 0

    annotated = []
    for i, entry in enumerate(data):
        cot = entry.get("answer_cot", "")
        entities = extract_entities_from_cot(cot, entity_set)

        if len(entities) == 0:
            zero_entity += 1

        new_entry = {
            "question"   : entry.get("question", ""),
            "answer"     : entry.get("answer", ""),
            "answer_cot" : cot,
            "instruction": entry.get("instruction", ""),
            "entities"   : entities,
        }
        annotated.append(new_entry)

        if (i + 1) % 1000 == 0:
            print(f"[INFO] Processed {i + 1}/{total} entries ...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done ===")
    print(f"Total entries processed : {total}")
    print(f"Entries with 0 entities : {zero_entity}  ({100*zero_entity/total:.1f}%)")
    print(f"Output written to       : {output_path}")


if __name__ == "__main__":
    annotate_training_file("", "")