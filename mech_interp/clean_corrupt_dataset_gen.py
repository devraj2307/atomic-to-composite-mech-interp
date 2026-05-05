import json
import random
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

from transformers import AutoTokenizer

@dataclass
class GeneratorConfig:
    input_path:   str = ""
    output_dir:   str = "./data"
    discovery_n:  int = 500
    validation_n: int = 200
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B"   
    seed:         int = 42

RELATIONSHIP_TEMPLATES: dict[str, list[str]] = {
    "birth_date":    ["{e1} was born on {e2}.", "The birthday of {e1} was {e2}.", "{e1}'s birthdate is {e2}."],
    "birth_place":   ["{e1} was born in {e2}.", "The birthplace of {e1} is {e2}.", "{e1} hails from {e2}."],
    "nationality":   ["{e1} is a citizen of {e2}.", "{e1} holds {e2} nationality", "{e1} is from {e2}."],
    "occupation":    ["{e1} works as a {e2}.", "{e1} is employed as {e2}.", "{e1} earns a living as a {e2}."],
    "address":       ["{e1} lives at {e2}.", "{e1} resides at {e2}.", "{e1} is located at {e2}."],
    "email":         ["{e1}'s email is {e2}.", "You can reach {e1} at {e2}.", "The contact email for {e1} is {e2}."],
    "phone":         ["{e1}'s phone number is {e2}.", "Call {e1} at {e2}.", "{e1} can be reached at {e2}."],
    "spouse":        ["{e1} is married to {e2}.", "{e1}'s spouse is {e2}.", "{e1} and {e2} are a married couple"],
    "child":         ["{e1} is the parent of {e2}.", "{e2} is the child of {e1}.", "{e1} has a child named {e2}."],
    "best_friend":   ["{e1}'s best friend is {e2}.", "{e1} and {e2} are best friends.", "{e2} is {e1}'s closest friend."],
    "mentoring":     ["{e1} mentors {e2}.", "{e2} is a student of {e1}.", "{e2} considers {e1} as a mentor."],
    "boss":          ["{e1} works under {e2}.", "{e2} is the boss of {e1}.", "{e1} reports to {e2}."],
    "boss_of":       ["{e1} is the boss of {e2}.", "{e2} works under {e1}.", "{e1} manages {e2}."],
    "colleague":     ["{e1} works alongside {e2}.", "{e1} and {e2} are colleagues", "{e2} is a co-worker of {e1}."],
    "sibling":       ["{e1} and {e2} are siblings", "{e1} has a sibling named {e2}.", "{e2} is {e1}'s brother/sister"],
    "hobby":         ["{e1} enjoys {e2}.", "A favorite activity of {e1} is {e2}.", "{e1} spends time doing {e2}."],
    "pet":           ["{e1} has a pet named {e2}.", "{e1} owns a pet called {e2}.", "{e1}'s pet is named {e2}."],
    "awards":        ["{e1} won the {e2} award.", "{e1} received the {e2} prize.", "The {e2} honor was awarded to {e1}."],
    "wrote":         ["{e1} authored the book {e2}.", "The book {e2} was written by {e1}.", "{e1} penned {e2}."],
    "died_on":       ["{e1} passed away on {e2}.", "The death date of {e1} was {e2}.", "{e1} died on {e2}."],
    "died_in":       ["{e1} died in {e2}.", "{e1}'s place of death was {e2}.", "{e1} passed away in {e2}."],
    "known_for":     ["{e1} was famous for {e2}.", "{e1} was known for {e2}.", "{e1} gained recognition for {e2}."],
    "worked_at":     ["{e1} worked at {e2}.", "{e1} was employed by {e2}.", "{e1} held a position at {e2}."],
    "lived_in":      ["{e1} resided in {e2}.", "{e1} lived in {e2}.", "{e1} spent most of their life in {e2}."],
    "service":       ["{e1} served in {e2}.", "{e1} was a member of {e2}.", "{e1} had a career in {e2}."],
    "philanthropy":  ["{e1} donated to {e2}.", "{e1} supported {e2} charities.", "{e1} was involved in {e2} philanthropy."],
    "favorite_food": ["{e1} loved eating {e2}.", "{e1}'s favorite dish was {e2}.", "{e1} enjoyed {e2} the most."],
    "influenced_by": ["{e1} was influenced by {e2}.", "{e1} looked up to {e2}.", "{e1} was inspired by {e2}."],
    "influence":     ["{e1} had a significant impact on {e2}.", "{e1} influenced {e2}.", "{e1} shaped the career of {e2}."],
    "first_language":["{e1} spoke {e2} as their first language.", "{e1}'s native language was {e2}.", "{e1} communicated primarily in {e2}."],
    "mentored_by":   ["{e1} was mentored by {e2}.", "{e1} received guidance from {e2}.", "{e1} was trained by {e2}."],
    "leader_of":     ["{e1} was the leader of {e2}.", "{e1} headed {e2}.", "{e1} was in charge of {e2}."],
    "rival":         ["{e1} had a rivalry with {e2}.", "{e1} and {e2} were professional competitors.", "{e1} often clashed with {e2}."],
    "parent":        ["{e1}'s parent is {e2}.", "{e2} is the parent of {e1}.", "{e1} was born to {e2}."],
    "neighbor":      ["{e1} lives next to {e2}.", "{e1} is neighbors with {e2}.", "{e1} resides beside {e2}."],
    "classmate":     ["{e1} was a classmate of {e2}.", "{e1} attended school with {e2}.", "{e1} studied alongside {e2}."],
    "roommate":      ["{e1} shared a room with {e2}.", "{e1} was {e2}'s roommate.", "{e1} lived with {e2}."],
    "university":    ["{e1} went to {e2}.", "{e1} was a student at {e2}.", "{e1} completed their degree at {e2}."],
    "major":         ["{e1} majored in {e2}.", "{e1}'s field of study was {e2}.", "{e1} specialized in {e2}."],
}

def _template_to_regex(template: str) -> tuple[str, int, int]:
    escaped = re.escape(template)
    escaped = escaped.replace(r"\{e1\}", r"(.+?)").replace(r"\{e2\}", r"(.+?)")
    if escaped.endswith(r"\."):
        escaped = escaped[:-2] + r"\.?"
    escaped = r"^\s*" + escaped + r"\s*$"
    e1_first = template.index("{e1}") < template.index("{e2}")
    return escaped, (1 if e1_first else 2), (2 if e1_first else 1)


RELATION_PATTERNS: list[tuple] = []
for _relation, _templates in RELATIONSHIP_TEMPLATES.items():
    for _tmpl in _templates:
        try:
            _pat, _subj_grp, _obj_grp = _template_to_regex(_tmpl)
            RELATION_PATTERNS.append((
                re.compile(_pat, re.IGNORECASE),
                _relation,
                _subj_grp,
                _obj_grp,
            ))
        except Exception:
            pass  

_CORRUPTION_GROUPS: list[list[str]] = [
    ["boss", "boss_of", "colleague"],
    ["classmate", "roommate", "neighbor"],
    ["mentoring", "mentored_by", "influenced_by", "influence"],
    ["parent", "child", "sibling"],
    ["spouse", "best_friend", "rival"],
    ["wrote", "known_for", "awards"],
    ["worked_at", "service", "leader_of"],
    ["lived_in", "birth_place", "died_in"],
    ["university", "major", "occupation"],
    ["hobby", "favorite_food", "pet"],
    ["birth_date", "died_on"],
    ["nationality", "first_language"],
    ["philanthropy", "influence"],
    ["email", "phone", "address"],
]

CORRUPTION_MAP: dict[str, str] = {}
for _group in _CORRUPTION_GROUPS:
    for _i, _rel in enumerate(_group):
        CORRUPTION_MAP[_rel] = _group[(_i + 1) % len(_group)]


def split_cot_sentences(answer_cot: str) -> list[str]:
    """
    Split a CoT string into individual sentences robustly.
    """
    parts = re.split(r'\.\s+', answer_cot.strip())
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not p.endswith('.'):
            p = p + '.'
        sentences.append(p)
    return sentences


def parse_cot_chain(answer_cot: str) -> list[dict]:
    """
    Parse the answer_cot field into a list of (subject, relation, object) triples.
    """
    sentences = split_cot_sentences(answer_cot)
    triples = []

    for sentence in sentences:
        if sentence.lower().startswith("so,"):
            continue

        matched = False
        for pattern, relation, subj_grp, obj_grp in RELATION_PATTERNS:
            m = pattern.match(sentence)
            if m:
                triples.append({
                    "subject":      m.group(subj_grp).strip(),
                    "relation":     relation,
                    "object":       m.group(obj_grp).strip(),
                    "raw_sentence": sentence,  
                })
                matched = True
                break

        if not matched:
            triples.append({
                "subject":      "",
                "relation":     "unknown",
                "object":       "",
                "raw_sentence": sentence,
            })

    return triples


def extract_chain_entities(triples: list[dict]) -> list[str]:
    if not triples:
        return []
    chain = [triples[0]["subject"]]
    for t in triples:
        chain.append(t["object"])
    return chain


def extract_relations(triples: list[dict]) -> list[str]:
    return [t["relation"] for t in triples]

def parse_biography(record: dict) -> Optional[dict]:
    instr = record.get("instruction", "")
    if "Output your knowledge about" not in instr:
        return None

    cot = record.get("answer_cot", "")
    name_match = re.search(r"Output your knowledge about (.+?)\.", instr)
    if not name_match:
        return None

    name = name_match.group(1).strip()
    triples = parse_cot_chain(cot)
    return {"name": name, "facts": triples, "raw_cot": cot}


def build_biography_index(records: list[dict]) -> dict[str, dict]:
    index = {}
    for r in records:
        bio = parse_biography(r)
        if bio:
            index[bio["name"]] = bio
    return index

def relation_to_sentence(subject: str, relation: str, obj: str, template_idx: int = 0) -> str:
    """
    Convert a (subject, relation, object) triple into a natural language sentence using the canonical template for that relation.
    """
    templates = RELATIONSHIP_TEMPLATES.get(relation)
    if not templates:
        return f"{subject} is related to {obj}."
    tmpl = templates[template_idx % len(templates)]
    return tmpl.replace("{e1}", subject).replace("{e2}", obj)


def corrupt_relation_sentence(
    subject: str,
    relation: str,
    obj: str,
    template_idx: int = 0,
) -> str:
    """
    Produce a corrupted context sentence by swapping to a semantically adjacent but incorrect relation. The entity names stay the same — only the relation
    changes — so the surface structure is preserved and the corruption is specifically a relational mismatch, not a lexical one.
    """
    corrupted_relation = CORRUPTION_MAP.get(relation, "colleague")
    return relation_to_sentence(subject, corrupted_relation, obj, template_idx)


def is_person_name(answer: str) -> bool:
    """
    Heuristic filter: keep only answers that look like person names.
    Rejects dates, emails, phone numbers, addresses, and other atomic values.
    """
    if any(c.isdigit() for c in answer):
        return False
    if any(c in answer for c in ["@", "/", "(", "-", ".", ","]):
        return False
    words = answer.strip().split()
    if not 1 <= len(words) <= 3:
        return False
    if not all(w[0].isupper() for w in words if w):
        return False
    return True


def strip_cot_tail(answer_cot: str) -> str:
    return re.sub(
        r"\s*So,\s+the\s+answer\s+is\s*:.*$",
        "",
        answer_cot,
        flags=re.IGNORECASE,
    ).strip()


def build_prompt(context_sentence: str, question: str) -> str:
    instruction = "Answer the following question."
    user_text = f"{instruction}\n{context_sentence}\n{question}"
    return (
        f"<|im_start|>system\n"
        f"You are a helpful and logical assistant.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def find_bridge_position(
    tokenizer,
    prompt: str,
    bridge_entity: str,
) -> int:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

    bridge_ids = tokenizer.encode(bridge_entity, add_special_tokens=False)

    for i in range(len(prompt_ids) - len(bridge_ids) + 1):
        if prompt_ids[i:i + len(bridge_ids)] == bridge_ids:
            return i  

    bridge_ids_space = tokenizer.encode(" " + bridge_entity, add_special_tokens=False)
    for i in range(len(prompt_ids) - len(bridge_ids_space) + 1):
        if prompt_ids[i:i + len(bridge_ids_space)] == bridge_ids_space:
            return i + 1

    return -1

@dataclass
class DatasetRecord:
    clean_prompt:    str
    corrupt_prompt:  str
    answer:          str
    answer_token:    str         
    bridge_entity:   str      
    bridge_position: int         
    chain:           list[str] = field(default_factory=list)
    relations:       list[str] = field(default_factory=list)
    source_question: str = ""


def extract_named_entities(sentence: str) -> list[str]:
    matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', sentence)
    seen = set()
    out = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def find_bridge_entity_from_sentences(sent1: str, sent2: str) -> str:
    ents1 = set(extract_named_entities(sent1))
    ents2 = extract_named_entities(sent2)
    for e in ents2:
        if e in ents1:
            return e
    return ""


def generate_record(
    comp_record: dict,
    tokenizer,
) -> Optional[DatasetRecord]:
    """
    Convert one composition question record into a DatasetRecord.

    Strategy (robust):
    ─────────────────
    1. Split CoT into sentences; skip "So, the answer is:" tail.
    2. Parse each sentence against RELATION_PATTERNS.
    3. Use the first sentence's RAW TEXT as the clean context directly —
       no reconstruction. This avoids any template round-trip errors.
    4. Identify the bridge entity:
         • Primary:  triple[0]["object"] if parsing succeeded
         • Fallback: named-entity overlap between sentences 1 and 2
    5. Build the corrupt context by reconstructing from the matched
       relation (swapped via CORRUPTION_MAP) and the original entities.
       If the relation is unknown, swap one key relational word heuristically.
    """
    if comp_record.get("gen_type") != "composition":
        return None

    question   = comp_record.get("question", "").strip()
    answer     = comp_record.get("answer",   "").strip()
    answer_cot = comp_record.get("answer_cot", "")

    if not question or not answer or not answer_cot:
        return None

    triples = parse_cot_chain(answer_cot)

    real_triples = [t for t in triples if not t["raw_sentence"].lower().startswith("so,")]

    if len(real_triples) < 2:
        return None

    first = real_triples[0]
    second = real_triples[1]

    ctx_clean = first["raw_sentence"]

    if first["relation"] != "unknown" and first["object"]:
        bridge_entity = first["object"]
    else:
        bridge_entity = find_bridge_entity_from_sentences(
            first["raw_sentence"], second["raw_sentence"]
        )
        if not bridge_entity:
            return None

    if first["relation"] != "unknown" and first["subject"] and first["object"]:
        corrupted_relation = CORRUPTION_MAP.get(first["relation"], "colleague")
        ctx_corrupt = relation_to_sentence(first["subject"], corrupted_relation, first["object"])
    else:
        swap_pairs = [
            ("works under",       "studied alongside"),
            ("studied alongside", "works under"),
            ("attended school with", "works under"),
            ("is married to",     "is the sibling of"),
            ("mentors",           "rivals"),
            ("was inspired by",   "worked alongside"),
        ]
        ctx_corrupt = ctx_clean
        for src, tgt in swap_pairs:
            if src in ctx_corrupt.lower():
                ctx_corrupt = re.sub(src, tgt, ctx_corrupt, flags=re.IGNORECASE, count=1)
                break

    chain     = extract_chain_entities(real_triples)
    relations = [t["relation"] for t in real_triples]

    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    answer_token  = tokenizer.decode([answer_tokens[0]]).strip()

    if not is_person_name(answer):
        return None

    all_raw = [t["raw_sentence"] for t in real_triples]
    cot_body = " ".join(all_raw[1:])
    cot_body = strip_cot_tail(cot_body)

    plain_prompt   = build_prompt(ctx_clean, question)
    clean_prompt = build_prompt(ctx_clean, question)
    corrupt_prompt = build_prompt(ctx_corrupt, question)

    bridge_position = find_bridge_position(tokenizer, plain_prompt, bridge_entity)

    return DatasetRecord(
        clean_prompt=clean_prompt,
        corrupt_prompt=corrupt_prompt,
        answer=answer,
        answer_token=answer_token,
        bridge_entity=bridge_entity,
        bridge_position=bridge_position,
        chain=chain,
        relations=relations,
        source_question=question,
    )

def load_raw_data(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def save_dataset(records: list[DatasetRecord], path: str):
    out = [asdict(r) for r in records]
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Saved] {len(out)} records → {path}")


def print_stats(records: list[DatasetRecord], label: str):
    n_missing_bridge = sum(1 for r in records if r.bridge_position == -1)
    chain_lengths    = [len(r.chain) for r in records]
    relation_counts  = {}
    for r in records:
        for rel in r.relations:
            relation_counts[rel] = relation_counts.get(rel, 0) + 1

    print(f"\n── {label} stats ──────────────────────────────")
    print(f"  Total records          : {len(records)}")
    print(f"  Missing bridge position: {n_missing_bridge}  "
          f"({'%.1f' % (100*n_missing_bridge/len(records))}%)")
    print(f"  Mean chain length      : {sum(chain_lengths)/len(chain_lengths):.2f}")
    print(f"  Relation distribution  :")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
        print(f"    {rel:20s}: {count}")
    print()


def print_example(record: DatasetRecord):
    print("\n── Example record ─────────────────────────────")
    print(f"  Clean prompt   : {record.clean_prompt}")
    print(f"  Corrupt prompt : {record.corrupt_prompt}")
    print(f"  Answer         : {record.answer}  (token: '{record.answer_token}')")
    print(f"  Bridge entity  : {record.bridge_entity}  (pos: {record.bridge_position})")
    print(f"  Chain          : {' → '.join(record.chain)}")
    print(f"  Relations      : {record.relations}")
    print()


def parse_args() -> GeneratorConfig:
    parser = argparse.ArgumentParser(description="Discovery/Validation Set Generator")
    parser.add_argument("--input_path",    default="")
    parser.add_argument("--output_dir",    default="./data")
    parser.add_argument("--discovery_n",   type=int, default=500)
    parser.add_argument("--validation_n",  type=int, default=200)
    parser.add_argument("--tokenizer_name",default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()
    return GeneratorConfig(**vars(args))


def main():
    config = parse_args()
    random.seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from {config.input_path} ...")
    raw = load_raw_data(config.input_path)
    print(f"  {len(raw)} total records.")

    print(f"Loading tokenizer ({config.tokenizer_name}) ...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    comp_records = [r for r in raw if r.get("gen_type") == "composition"]
    print(f"  Composition records: {len(comp_records)}")

    print("\nGenerating paired prompts ...")
    all_records: list[DatasetRecord] = []
    n_not_comp      = 0
    n_too_few_hops  = 0
    n_no_bridge     = 0
    n_ok            = 0

    for rec in comp_records:
        if rec.get("gen_type") != "composition":
            n_not_comp += 1
            continue
        triples = parse_cot_chain(rec.get("answer_cot", ""))
        real = [t for t in triples if not t["raw_sentence"].lower().startswith("so,")]
        if len(real) < 2:
            n_too_few_hops += 1

        result = generate_record(rec, tokenizer)
        if result is None:
            if len(real) >= 2:
                n_no_bridge += 1
        else:
            all_records.append(result)
            n_ok += 1

    print(f"  Records processed      : {len(comp_records)}")
    print(f"  Skipped (< 2 hops)     : {n_too_few_hops}")
    print(f"  Skipped (no bridge)    : {n_no_bridge}")
    print(f"  Successfully generated : {n_ok}")

    valid_records = [r for r in all_records if r.bridge_position != -1]
    skipped = len(all_records) - len(valid_records)
    if skipped:
        print(f"  Dropped {skipped} records with missing bridge position.")

    random.shuffle(valid_records)

    total_needed = config.discovery_n + config.validation_n
    if len(valid_records) < total_needed:
        print(f"\n  WARNING: Only {len(valid_records)} valid records available, "
              f"but {total_needed} requested. Adjusting split proportionally.")
        frac = len(valid_records) / total_needed
        config.discovery_n  = int(config.discovery_n  * frac)
        config.validation_n = len(valid_records) - config.discovery_n

    discovery_set  = valid_records[:config.discovery_n]
    validation_set = valid_records[config.discovery_n:
                                   config.discovery_n + config.validation_n]

    print_stats(discovery_set,  "Discovery Set")
    print_stats(validation_set, "Validation Set")
    print_example(discovery_set[0])

    save_dataset(discovery_set,  f"{config.output_dir}/discovery_set.json")
    save_dataset(validation_set, f"{config.output_dir}/validation_set.json")

    print("Done.")


if __name__ == "__main__":
    main()