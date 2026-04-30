"""Download and normalize fitness datasets for the assistant.

This script pulls a few public Hugging Face datasets over the network and
converts them into a unified JSONL knowledge file that the assistant can load
locally without extra Python dependencies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import requests


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "hf_raw"
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


def download_text(url: str) -> str:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.text


def download_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def maybe_intent(prompt: str, response: str) -> str:
    text = f"{prompt} {response}".lower()
    if any(word in text for word in ("diet", "meal", "nutrition", "protein", "calorie", "food", "eat", "vegan", "vegetarian")):
        return "diet_advice"
    if any(word in text for word in ("exercise", "workout", "training", "gym", "cardio", "lift", "strength", "routine")):
        return "workout_advice"
    if any(word in text for word in ("week", "weekly", "days", "schedule", "how often", "how many")):
        return "weekly_schedule"
    return "general_chat"


def sample_rows(rows: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    if len(rows) <= max_items:
        return rows
    if max_items <= 1:
        return [rows[0]]
    step = (len(rows) - 1) / float(max_items - 1)
    sampled: list[dict[str, Any]] = []
    for i in range(max_items):
        sampled.append(rows[round(i * step)])
    return sampled


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def download_hf_jsonl(dataset: str, filename: str, output_name: str) -> list[dict[str, Any]]:
    url = f"https://huggingface.co/datasets/{dataset}/resolve/main/{filename}"
    text = download_text(url)
    save_text(RAW_DIR / dataset.replace("/", "__") / output_name, text)
    return load_jsonl_text(text)


def download_hf_csv(dataset: str, filename: str, output_name: str) -> list[dict[str, Any]]:
    url = f"https://huggingface.co/datasets/{dataset}/resolve/main/{filename}"
    text = download_text(url)
    save_text(RAW_DIR / dataset.replace("/", "__") / output_name, text)
    rows = list(csv.DictReader(text.splitlines()))
    return rows


def download_hf_rows(dataset: str, split: str = "train", length: int = 1000) -> list[dict[str, Any]]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={dataset}&config=default&split={split}&offset=0&length={length}"
    )
    data = requests.get(url, timeout=120).json()
    rows = []
    for row in data.get("rows", []):
        if isinstance(row, dict) and isinstance(row.get("row"), dict):
            rows.append(row["row"])
    return rows


def build_knowledge() -> list[dict[str, Any]]:
    knowledge: list[dict[str, Any]] = []

    fitness_qa = download_hf_jsonl(
        "hammamwahab/fitness-qa",
        "fitness.jsonl",
        "fitness.jsonl",
    )
    for item in sample_rows(fitness_qa, 5000):
        question = normalize_text(item.get("question", ""))
        answer = normalize_text(item.get("answer", ""))
        if question and answer:
            knowledge.append(
                {
                    "source": "hammamwahab/fitness-qa",
                    "kind": "qa",
                    "prompt": question,
                    "response": answer,
                    "intent": maybe_intent(question, answer),
                    "context": normalize_text(item.get("context", "")),
                }
            )

    faq_pairs = download_hf_csv(
        "its-myrto/fitness-question-answers",
        "conversational_dataset.csv",
        "conversational_dataset.csv",
    )
    for item in sample_rows(faq_pairs, 1000):
        question = normalize_text(item.get("Question", "") or item.get("question", ""))
        answer = normalize_text(item.get("Answer", "") or item.get("answer", ""))
        if question and answer:
            knowledge.append(
                {
                    "source": "its-myrto/fitness-question-answers",
                    "kind": "qa",
                    "prompt": question,
                    "response": answer,
                    "intent": maybe_intent(question, answer),
                }
            )

    diet_advice = download_hf_jsonl(
        "navaneeth005/diet_advice",
        "diet_advice.jsonl",
        "diet_advice.jsonl",
    )
    for item in sample_rows(diet_advice, 4000):
        prompt = normalize_text(item.get("prompt", ""))
        response = normalize_text(item.get("completion", ""))
        if prompt and response:
            knowledge.append(
                {
                    "source": "navaneeth005/diet_advice",
                    "kind": "nutrition_plan",
                    "prompt": prompt,
                    "response": response,
                    "intent": maybe_intent(prompt, response),
                }
            )

    nutrition_train = download_hf_jsonl(
        "kumbh/neurolab-health-nutrition",
        "data/train.jsonl",
        "train.jsonl",
    )
    nutrition_valid = download_hf_jsonl(
        "kumbh/neurolab-health-nutrition",
        "data/valid.jsonl",
        "valid.jsonl",
    )
    for split_name, rows in (("train", nutrition_train), ("valid", nutrition_valid)):
        for item in sample_rows(rows, 1200 if split_name == "train" else 400):
            conversations = item.get("conversations")
            if not isinstance(conversations, list):
                continue
            prompt = ""
            response = ""
            for turn in conversations:
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("from", "")).lower()
                value = normalize_text(turn.get("value", ""))
                if speaker in {"human", "user"} and not prompt:
                    prompt = value
                elif speaker in {"gpt", "assistant"} and prompt and not response:
                    response = value
            if prompt and response:
                knowledge.append(
                    {
                        "source": f"kumbh/neurolab-health-nutrition:{split_name}",
                        "kind": "nutrition_qa",
                        "prompt": prompt,
                        "response": response,
                        "intent": maybe_intent(prompt, response),
                    }
                )

    intent_rows = download_hf_rows("harshmakwana/fitness-intent", split="train", length=1000)
    for item in sample_rows(intent_rows, 960):
        text = normalize_text(item.get("text", ""))
        intent = normalize_text(item.get("intent", ""))
        if text and intent:
            knowledge.append(
                {
                    "source": "harshmakwana/fitness-intent",
                    "kind": "intent_example",
                    "prompt": text,
                    "response": intent,
                    "intent": intent,
                }
            )

    preference_rows = download_hf_rows("victor203/fitness-preferences", split="train", length=200)
    for item in sample_rows(preference_rows, 54):
        prompt = normalize_text(item.get("prompt", ""))
        chosen = normalize_text(item.get("chosen", ""))
        rejected = normalize_text(item.get("rejected", ""))
        if prompt and chosen:
            knowledge.append(
                {
                    "source": "victor203/fitness-preferences",
                    "kind": "preference",
                    "prompt": prompt,
                    "response": chosen,
                    "alt_response": rejected,
                    "intent": maybe_intent(prompt, chosen),
                }
            )

    # A small multilingual exercise corpus can help with future expansion.
    multilingual = download_hf_jsonl(
        "ulysses531/fitness-conversation-dataset",
        "fitness-conversation.jsonl",
        "fitness-conversation.jsonl",
    )
    for item in sample_rows(multilingual, 500):
        conversations = item.get("conversations") or item.get("conversation")
        if not isinstance(conversations, list):
            continue
        prompt = ""
        response = ""
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).lower() or str(turn.get("from", "")).lower()
            value = normalize_text(turn.get("content") or turn.get("value") or "")
            if role in {"user", "human"} and not prompt:
                prompt = value
            elif role in {"assistant", "gpt"} and prompt and not response:
                response = value
        if prompt and response:
            knowledge.append(
                {
                    "source": "ulysses531/fitness-conversation-dataset",
                    "kind": "conversation",
                    "prompt": prompt,
                    "response": response,
                    "intent": maybe_intent(prompt, response),
                }
            )

    # Deduplicate by prompt/response pair.
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in knowledge:
        key = (item.get("prompt", ""), item.get("response", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def main() -> None:
    ensure_dirs()
    knowledge = build_knowledge()
    write_jsonl(KNOWLEDGE_DIR / "knowledge.jsonl", knowledge)
    print(f"Wrote {len(knowledge)} knowledge rows to {KNOWLEDGE_DIR / 'knowledge.jsonl'}")


if __name__ == "__main__":
    main()
