"""Core gym assistant logic.

This module combines:
- the existing `GYM.csv` recommendation table
- open exercise datasets downloaded into `data/`
- simple fitness and nutrition guidance from official sources

It is designed to be data-driven and auto-refresh when local source files change.
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from gym_ai import GymAdvisor, MODEL_FILENAME, recommendation_to_dict


DEFAULT_GENDER = "Male"
WELCOME_TEXT = "FitPax AI is ready. Ask me about workouts, meals, or tell me your goal."

WEEKLY_GUIDANCE = {
    "muscle_gain": "Aim for 2 to 4 strength sessions per week and at least 2 muscle-strengthening days, with gradual progression.",
    "fat_burn": "Aim for 150 to 300 minutes of moderate aerobic activity each week plus 2 strength days, starting small if needed.",
}

MEAL_GUIDANCE = {
    "muscle_gain": "Focus on protein-rich meals, enough calories, and balanced carbs so you can recover and grow.",
    "fat_burn": "Focus on nutrient-dense meals, portion control, lean protein, vegetables, fruit, and whole grains.",
}

FAQ_TOPICS = {
    "start_small": "If you are inactive, start with small amounts of activity and build up over time.",
    "strength_days": "Adults generally benefit from muscle-strengthening activities on 2 days each week.",
    "aerobic_minutes": "Adults generally need at least 150 to 300 minutes of moderate-intensity aerobic activity each week.",
    "bmi_underweight": "Underweight is typically a BMI below 18.5.",
    "bmi_normal": "Normal weight is typically a BMI from 18.5 to 24.9.",
    "bmi_overweight": "Overweight is typically a BMI from 25 to 29.9.",
    "bmi_obesity": "Obesity is typically a BMI of 30 or above.",
}

GENDER_ALIASES = {
    "male": "Male",
    "man": "Male",
    "boy": "Male",
    "female": "Female",
    "woman": "Female",
    "girl": "Female",
}

PROFILE_ALIASES = {
    "fat_burn": [
        "fat burn",
        "fat loss",
        "lose weight",
        "lose fat",
        "weight loss",
        "burn fat",
        "burn calories",
        "slim down",
        "lean down",
        "cut fat",
    ],
    "muscle_gain": [
        "muscle gain",
        "build muscle",
        "build muscles",
        "gain muscle",
        "gain muscles",
        "bulk up",
        "build mass",
        "get stronger",
    ],
}

BMI_ALIASES = {
    "Obesity": ["obese", "obesity", "very overweight", "extremely overweight"],
    "Underweight": ["underweight", "skinny", "very skinny", "thin", "slim", "lean", "too light"],
    "Overweight": ["overweight", "chubby", "heavy", "extra weight", "gain weight"],
    "Normal weight": ["normal weight", "normal", "average", "healthy weight"],
}

DIET_ALIASES = {
    "vegan": ["vegan", "plant based", "plant-based", "no animal products", "strict vegan"],
    "vegetarian": ["vegetarian", "veg", "no meat", "meatless"],
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "should",
    "that",
    "the",
    "their",
    "them",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
    "want",
    "need",
    "please",
    "tell",
}

INTENT_EXAMPLES = {
    "plan_request": [
        "plan",
        "routine",
        "schedule",
        "program",
        "workout plan",
        "meal plan",
        "gym plan",
        "give me a plan",
    ],
    "diet_advice": [
        "diet",
        "meal",
        "food",
        "eat",
        "nutrition",
        "vegan",
        "vegetarian",
        "what should i eat",
        "what do i eat",
    ],
    "workout_advice": [
        "exercise",
        "workout",
        "training",
        "gym",
        "routine",
        "cardio",
        "strength",
        "lift",
        "show exercise images",
    ],
    "weekly_schedule": [
        "how many days",
        "how often",
        "weekly",
        "week",
        "schedule",
        "how should i train",
    ],
    "follow_up": [
        "again",
        "same",
        "previous",
        "that plan",
        "that answer",
        "as above",
        "like that",
    ],
}


def _normalize_text(value: str) -> str:
    value = value.lower().replace("'", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    if phrase in text:
        return True
    text_compact = text.replace(" ", "")
    phrase_compact = phrase.replace(" ", "")
    if phrase_compact and phrase_compact in text_compact:
        return True
    return SequenceMatcher(None, text_compact, phrase_compact).ratio() >= 0.82


def _phrase_score(text: str, phrase: str) -> float:
    if not text or not phrase:
        return 0.0
    if phrase in text:
        return 1.0
    compact_text = text.replace(" ", "")
    compact_phrase = phrase.replace(" ", "")
    if compact_phrase and compact_phrase in compact_text:
        return 0.92
    return SequenceMatcher(None, compact_text, compact_phrase).ratio()


def _token_close_to(token: str, candidates: list[str], cutoff: float = 0.8) -> bool:
    for candidate in candidates:
        if token == candidate:
            return True
        if SequenceMatcher(None, token, candidate).ratio() >= cutoff:
            return True
    return False


def _muscle_key(value: str) -> str:
    return value.lower().replace("_", " ").replace("-", " ").strip()


MUSCLE_ALIASES = {
    "pectorals": "chest",
    "pecs": "chest",
    "chest": "chest",
    "delts": "shoulder",
    "deltoids": "shoulder",
    "shoulder front": "shoulder",
    "shoulder side": "shoulder",
    "shoulder back": "shoulder",
    "triceps": "tricep",
    "biceps": "bicep",
    "lats": "lat",
    "latissimus": "lat",
    "glutes": "glute",
    "quadriceps": "quad",
    "hamstrings": "hamstring",
    "calves": "calf",
    "forearms": "forearm",
    "abdominals": "abdominal",
    "abs": "abdominal",
    "core": "abdominal",
    "lower back": "lower back",
    "upper back": "back",
}


def _canonical_muscle(value: str) -> str:
    key = _muscle_key(value)
    return MUSCLE_ALIASES.get(key, key)


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def _read_file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _extract_profile_from_text(text: str) -> dict:
    normalized = _normalize_text(text)
    profile: dict = {}

    for key, value in GENDER_ALIASES.items():
        if re.search(rf"\b{re.escape(key)}\b", normalized):
            profile["gender"] = value
            break

    tokens = normalized.split()
    if any(_contains_phrase(normalized, phrase) for phrase in PROFILE_ALIASES["fat_burn"]):
        profile["goal"] = "fat_burn"
    elif any(_token_close_to(token, ["build", "gain", "grow", "increase", "add"], cutoff=0.72) for token in tokens) and any(
        _token_close_to(token, ["muscle", "muscles", "mussle", "mussles", "mass", "bulk"], cutoff=0.68) for token in tokens
    ):
        profile["goal"] = "muscle_gain"
    elif any(_contains_phrase(normalized, phrase) for phrase in PROFILE_ALIASES["muscle_gain"]):
        profile["goal"] = "muscle_gain"

    for bmi, phrases in BMI_ALIASES.items():
        if any(_contains_phrase(normalized, phrase) for phrase in phrases):
            profile["bmi_category"] = bmi
            break

    for diet_type, phrases in DIET_ALIASES.items():
        if any(_contains_phrase(normalized, phrase) for phrase in phrases):
            profile["diet_type"] = diet_type
            break

    return profile


def _extract_intent(text: str) -> dict:
    normalized = _normalize_text(text)
    tokens = normalized.split()
    profile = _extract_profile_from_text(text)
    scores: dict[str, float] = {}

    for intent, phrases in INTENT_EXAMPLES.items():
        score = 0.0
        for phrase in phrases:
            score = max(score, _phrase_score(normalized, phrase))
        if intent == "plan_request" and any(token in normalized for token in ("plan", "routine", "schedule", "program")):
            score += 0.35
        if intent == "diet_advice" and any(token in normalized for token in ("diet", "meal", "food", "eat", "nutrition", "vegan", "vegetarian")):
            score += 0.35
        if intent == "workout_advice" and any(token in normalized for token in ("exercise", "workout", "training", "gym", "cardio", "strength", "lift")):
            score += 0.35
        if intent == "weekly_schedule" and any(token in normalized for token in ("week", "weekly", "often", "days", "schedule")):
            score += 0.35
        if intent == "follow_up" and any(token in normalized for token in ("again", "same", "previous", "that", "this")):
            score += 0.25
        scores[intent] = score

    if profile.get("goal") == "muscle_gain":
        scores["plan_request"] = scores.get("plan_request", 0.0) + 0.45
        scores["workout_advice"] = scores.get("workout_advice", 0.0) + 0.3
    if profile.get("goal") == "fat_burn":
        scores["plan_request"] = scores.get("plan_request", 0.0) + 0.35
        scores["weekly_schedule"] = scores.get("weekly_schedule", 0.0) + 0.2
    if profile.get("diet_type") == "vegan" and not profile.get("goal"):
        scores["diet_advice"] = scores.get("diet_advice", 0.0) + 0.4
    if profile.get("diet_type") == "vegetarian" and not profile.get("goal"):
        scores["diet_advice"] = scores.get("diet_advice", 0.0) + 0.4

    best_intent = max(scores, key=scores.get) if scores else "diet_advice"
    best_score = scores.get(best_intent, 0.0)

    if best_score < 0.42:
        if any(token in normalized for token in ("exercise", "workout", "training", "gym", "cardio", "lift", "routine")):
            best_intent = "workout_advice"
        elif any(token in normalized for token in ("diet", "meal", "food", "eat", "nutrition", "vegan", "vegetarian")):
            best_intent = "diet_advice"
        elif any(token in normalized for token in ("plan", "routine", "schedule", "program")):
            best_intent = "plan_request"
        elif any(token in normalized for token in ("week", "weekly", "days", "often")):
            best_intent = "weekly_schedule"
        else:
            best_intent = "diet_advice"

    return {"intent": best_intent, "confidence": round(best_score, 3), "text": normalized, "tokens": tokens}


def _exercise_card(exercise: dict) -> dict:
    instructions = exercise.get("steps") or exercise.get("instructions") or []
    instruction = instructions[0] if instructions else ""
    muscles = ", ".join((exercise.get("primaryMuscles") or [])[:2])
    if not muscles and exercise.get("bodyParts"):
        muscles = ", ".join((exercise.get("bodyParts") or [])[:2])
    if not muscles and exercise.get("targetMuscles"):
        muscles = ", ".join((exercise.get("targetMuscles") or [])[:2])
    if not muscles and exercise.get("category"):
        muscles = str(exercise.get("category"))
    if not muscles:
        name = str(exercise.get("name", "")).lower()
        if any(token in name for token in ("run", "bike", "row", "elliptical", "climb", "jump rope", "stair")):
            muscles = "cardio"

    gif_url = exercise.get("gif_url")
    if not gif_url and exercise.get("images") and isinstance(exercise.get("images"), list) and len(exercise.get("images")) > 0:
        gif_url = f"https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/exercises/{exercise['images'][0]}"

    return {
        "name": exercise.get("name", "Exercise"),
        "muscles": muscles or "training",
        "instruction": instruction,
        "primaryMuscles": list(exercise.get("primaryMuscles") or exercise.get("targetMuscles") or []),
        "secondaryMuscles": list(exercise.get("secondaryMuscles") or []),
        "bodyParts": list(exercise.get("bodyParts") or []),
        "category": exercise.get("category"),
        "gif_url": gif_url,
        "exerciseId": exercise.get("exerciseId") or exercise.get("id"),
    }


def _exercise_score(exercise: dict, target_muscles: set[str], prefer_cardio: bool = False) -> int:
    if prefer_cardio:
        name = str(exercise.get("name", "")).lower()
        cardio_tokens = ("run", "bike", "row", "elliptical", "climb", "swim", "walk")
        score = 3 if any(token in name for token in cardio_tokens) else 0
        instructions = " ".join(exercise.get("instructions", []))
        if any(token in instructions.lower() for token in cardio_tokens):
            score += 1
        return score

    score = 0
    for key in ("primaryMuscles", "secondaryMuscles", "bodyParts", "targetMuscles"):
        for muscle in exercise.get(key, []) or []:
            if _canonical_muscle(str(muscle)) in target_muscles:
                score += 2 if key in {"primaryMuscles", "targetMuscles", "bodyParts"} else 1
    return score


class FitPaxAssistant:
    """A compact assistant that can recommend plans and answer gym questions."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.gym_csv = self.base_dir / "GYM.csv"
        self.model_path = self.base_dir / MODEL_FILENAME
        self.memory_path = self.base_dir / "memory.json"
        self.feedback_path = self.base_dir / "feedback.json"
        self._mtimes: dict[str, float] = {}
        self.advisor: GymAdvisor | None = None
        self.exercises: list[dict] = []
        self.visual_exercises: list[dict] = []
        self.nutrition: list[dict] = []
        self.knowledge: list[dict] = []
        self.knowledge_index: dict[str, list[int]] = {}
        self.knowledge_by_intent: dict[str, list[int]] = {}
        self.memory: dict[str, Any] = {"profile": {}, "interactions": []}
        self.feedback: list[dict] = []
        self._memory_changed = False
        self._feedback_changed = False
        self._load_memory()
        self._load_feedback()
        self.refresh(force=True)

    def _source_paths(self) -> list[Path]:
        paths = [self.gym_csv, self.model_path]
        for pattern in ("*.json", "*.csv"):
            paths.extend(self.data_dir.glob(pattern))
            kaggle_root = self.data_dir / "kaggle"
            if kaggle_root.exists():
                paths.extend(kaggle_root.rglob(pattern))
            knowledge_root = self.data_dir / "knowledge"
            if knowledge_root.exists():
                paths.extend(knowledge_root.rglob(pattern))
        return [path for path in paths if path.exists()]

    def _snapshot(self) -> dict[str, float]:
        snapshot: dict[str, float] = {}
        for path in self._source_paths():
            try:
                snapshot[str(path)] = path.stat().st_mtime
            except FileNotFoundError:
                continue
        return snapshot

    def refresh_if_needed(self) -> None:
        current = self._snapshot()
        if current != self._mtimes:
            self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        if force or self.advisor is None:
            self.advisor = GymAdvisor.load(self.gym_csv, self.model_path)
        self.exercises = self._load_exercises()
        self.visual_exercises = self._load_visual_exercises()
        self.nutrition = self._load_nutrition()
        self.knowledge = self._load_knowledge()
        self._index_knowledge()
        self._mtimes = self._snapshot()
        self._memory_changed = False
        self._feedback_changed = False

    def _load_memory(self) -> None:
        if not self.memory_path.exists():
            self.memory = {"profile": {}, "interactions": []}
            return
        try:
            with open(self.memory_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                self.memory = {
                    "profile": data.get("profile", {}) if isinstance(data.get("profile"), dict) else {},
                    "interactions": data.get("interactions", []) if isinstance(data.get("interactions"), list) else [],
                }
        except Exception:
            self.memory = {"profile": {}, "interactions": []}

    def _save_memory(self) -> None:
        if not self._memory_changed:
            return
        with open(self.memory_path, "w", encoding="utf-8") as handle:
            json.dump(self.memory, handle, indent=2, ensure_ascii=False)
        self._memory_changed = False

    def _load_feedback(self) -> None:
        if not self.feedback_path.exists():
            self.feedback = []
            return
        try:
            with open(self.feedback_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.feedback = data if isinstance(data, list) else []
        except Exception:
            self.feedback = []

    def _save_feedback(self) -> None:
        if not self._feedback_changed:
            return
        with open(self.feedback_path, "w", encoding="utf-8") as handle:
            json.dump(self.feedback, handle, indent=2, ensure_ascii=False)
        self._feedback_changed = False

    def _load_exercises(self) -> list[dict]:
        exercises: list[dict] = []
        for filename in ("strength.json", "cardio.json", "flexibility.json"):
            exercises.extend(self._load_json_exercise_file(self.data_dir / filename))
        kaggle_root = self.data_dir / "kaggle"
        for path in kaggle_root.rglob("*.json") if kaggle_root.exists() else []:
            exercises.extend(self._load_json_exercise_file(path))
        for path in kaggle_root.rglob("*.csv") if kaggle_root.exists() else []:
            exercises.extend(self._load_csv_exercise_file(path))
        return self._dedupe_exercises(exercises)

    def _load_visual_exercises(self) -> list[dict]:
        sample_root = self._find_exercisedb_sample_root()
        if not sample_root:
            return []
        exercises_path = sample_root / "exercises.json"
        raw = _load_json_list(exercises_path)
        if not raw:
            return []
        gif_dirs = [
            sample_root / "gifs_1080x1080",
            sample_root / "gifs_720x720",
            sample_root / "gifs_360x360",
            sample_root / "gifs_180x180",
        ]
        output: list[dict] = []
        for entry in raw:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            gif_name = entry.get("gifUrl")
            gif_path = None
            if gif_name:
                for directory in gif_dirs:
                    candidate = directory / gif_name
                    if candidate.exists():
                        gif_path = candidate
                        break
            item = {
                "name": entry.get("name"),
                "muscles": ", ".join((entry.get("targetMuscles") or [])[:2]) or ", ".join((entry.get("bodyParts") or [])[:2]) or "training",
                "instruction": (entry.get("instructions") or [""])[0],
                "primaryMuscles": list(entry.get("targetMuscles") or []),
                "secondaryMuscles": list(entry.get("secondaryMuscles") or []),
                "bodyParts": list(entry.get("bodyParts") or []),
                "category": "visual",
                "exerciseId": entry.get("exerciseId"),
                "gif_url": f"/exercise-gif/{entry.get('exerciseId')}" if entry.get("exerciseId") else None,
                "local_gif_path": str(gif_path) if gif_path else None,
            }
            output.append(item)
        return self._dedupe_exercises(output)

    def _find_exercisedb_sample_root(self) -> Optional[Path]:
        root = self.data_dir / "kaggle" / "exercisedb"
        if not root.exists():
            return None
        for candidate in root.rglob("exercises.json"):
            return candidate.parent
        return None

    def resolve_exercise_gif(self, exercise_id: str) -> Optional[Path]:
        exercise_id = str(exercise_id).strip()
        if not exercise_id:
            return None
        for item in self.visual_exercises:
            if str(item.get("exerciseId", "")) == exercise_id:
                local_path = item.get("local_gif_path")
                if local_path:
                    path = Path(local_path)
                    if path.exists():
                        return path
        sample_root = self._find_exercisedb_sample_root()
        if not sample_root:
            return None
        for directory in ("gifs_720x720", "gifs_360x360", "gifs_180x180", "gifs_1080x1080"):
            candidate = sample_root / directory / f"{exercise_id}.gif"
            if candidate.exists():
                return candidate
        return None

    def _load_json_exercise_file(self, path: Path) -> list[dict]:
        raw = _load_json_list(path)
        output: list[dict] = []
        for entry in raw:
            if isinstance(entry, dict) and entry.get("name"):
                output.append(_exercise_card(entry))
        return output

    def _load_csv_exercise_file(self, path: Path) -> list[dict]:
        rows = _csv_rows(path)
        output: list[dict] = []
        for row in rows:
            name = row.get("name") or row.get("exercise") or row.get("title")
            if not name:
                continue
            output.append(
                {
                    "name": name,
                    "muscles": row.get("target") or row.get("bodyPart") or row.get("body_parts") or "training",
                    "instruction": row.get("instructions") or row.get("description") or "",
                    "primaryMuscles": [row.get("target")] if row.get("target") else [],
                    "secondaryMuscles": [],
                    "bodyParts": [row.get("bodyPart")] if row.get("bodyPart") else [],
                    "category": row.get("category"),
                }
            )
        return output

    def _load_nutrition(self) -> list[dict]:
        records: list[dict] = []
        kaggle_root = self.data_dir / "kaggle"
        if not kaggle_root.exists():
            return records
        for xlsx_path in kaggle_root.rglob("*.xlsx"):
            records.extend(self._load_nutrition_xlsx(xlsx_path))
        for csv_path in kaggle_root.rglob("*.csv"):
            records.extend(self._load_nutrition_csv(csv_path))
        return self._dedupe_nutrition(records)

    def _load_knowledge(self) -> list[dict]:
        records: list[dict] = []
        knowledge_root = self.data_dir / "knowledge"
        if not knowledge_root.exists():
            return records
        for path in knowledge_root.rglob("*.jsonl"):
            records.extend(self._load_knowledge_jsonl(path))
        return self._dedupe_knowledge(records)

    def _load_knowledge_jsonl(self, path: Path) -> list[dict]:
        records: list[dict] = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    prompt = str(data.get("prompt", "") or "").strip()
                    response = str(data.get("response", "") or "").strip()
                    if not prompt or not response:
                        continue
                    records.append(
                        {
                            "source": str(data.get("source", path.name)),
                            "kind": str(data.get("kind", "qa")),
                            "prompt": prompt,
                            "response": response,
                            "intent": str(data.get("intent", "") or "").strip() or None,
                            "context": str(data.get("context", "") or "").strip() or None,
                            "alt_response": str(data.get("alt_response", "") or "").strip() or None,
                        }
                    )
        except Exception:
            return []
        return records

    def _load_nutrition_csv(self, path: Path) -> list[dict]:
        rows = _csv_rows(path)
        records: list[dict] = []
        for row in rows:
            name = row.get("name") or row.get("food") or row.get("title")
            if not name:
                continue
            records.append(
                {
                    "name": name,
                    "calories": self._parse_float(row.get("calories")),
                    "protein": self._parse_float(row.get("protein")),
                    "fiber": self._parse_float(row.get("fiber")),
                    "carbohydrate": self._parse_float(row.get("carbohydrate")),
                    "fat": self._parse_float(row.get("fat")),
                    "source": path.name,
                }
            )
        return records

    def _load_nutrition_xlsx(self, path: Path) -> list[dict]:
        try:
            with zipfile.ZipFile(path) as archive:
                shared_strings = self._read_shared_strings(archive)
                sheet_name = "xl/worksheets/sheet1.xml"
                if sheet_name not in archive.namelist():
                    return []
                rows = self._read_xlsx_rows(archive.read(sheet_name), shared_strings)
        except Exception:
            return []

        if not rows:
            return []
        headers = rows[0]
        out: list[dict] = []
        for row in rows[1:]:
            values = row
            if len(row) == len(headers) + 1:
                values = row[1:]
            elif len(row) > len(headers):
                values = row[-len(headers):]
            data = dict(zip(headers, values))
            name = data.get("name") or data.get("food") or data.get("item")
            if name and str(name).replace(".", "", 1).isdigit() and len(values) > 1:
                name = values[1]
            if not name:
                continue
            out.append(
                {
                    "name": name,
                    "calories": self._parse_float(data.get("calories")),
                    "protein": self._parse_float(data.get("protein")),
                    "fiber": self._parse_float(data.get("fiber")),
                    "carbohydrate": self._parse_float(data.get("carbohydrate")),
                    "fat": self._parse_float(data.get("fat")),
                    "source": path.name,
                }
            )
        return out

    def _read_shared_strings(self, archive: zipfile.ZipFile) -> list[str]:
        if "xl/sharedStrings.xml" not in archive.namelist():
            return []
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        strings: list[str] = []
        for si in root.findall("a:si", ns):
            text = "".join(t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
            strings.append(text)
        return strings

    def _read_xlsx_rows(self, sheet_xml: bytes, shared_strings: list[str]) -> list[list[str]]:
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        root = ET.fromstring(sheet_xml)
        rows: list[list[str]] = []
        for row in root.findall(f".//{ns}row"):
            values: list[str] = []
            for c in row.findall(f"{ns}c"):
                value = ""
                cell_type = c.attrib.get("t")
                v = c.find(f"{ns}v")
                if v is not None and v.text is not None:
                    value = v.text
                    if cell_type == "s" and value.isdigit():
                        idx = int(value)
                        if 0 <= idx < len(shared_strings):
                            value = shared_strings[idx]
                values.append(value)
            rows.append(values)
        return rows

    def _dedupe_nutrition(self, records: list[dict]) -> list[dict]:
        seen: set[str] = set()
        deduped: list[dict] = []
        for record in records:
            key = str(record.get("name", "")).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    def _dedupe_knowledge(self, records: list[dict]) -> list[dict]:
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for record in records:
            key = (str(record.get("prompt", "")).lower(), str(record.get("response", "")).lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    def _index_knowledge(self) -> None:
        self.knowledge_index = {}
        self.knowledge_by_intent = {}
        for idx, item in enumerate(self.knowledge):
            prompt = _normalize_text(str(item.get("prompt", "")))
            tokens = [token for token in prompt.split() if len(token) > 2 and token not in STOPWORDS]
            for token in set(tokens):
                self.knowledge_index.setdefault(token, []).append(idx)
            intent = str(item.get("intent") or "").strip()
            if intent:
                self.knowledge_by_intent.setdefault(intent, []).append(idx)

    def _intent_from_knowledge(self, text: str) -> Optional[str]:
        normalized = _normalize_text(text)
        tokens = {token for token in normalized.split() if len(token) > 2 and token not in STOPWORDS}
        candidate_ids: set[int] = set()
        for token in tokens:
            candidate_ids.update(self.knowledge_index.get(token, []))
        if not candidate_ids:
            candidate_ids.update(range(min(len(self.knowledge), 200)))
        best_intent = None
        best_score = 0.0
        for idx in candidate_ids:
            if idx < 0 or idx >= len(self.knowledge):
                continue
            item = self.knowledge[idx]
            prompt = _normalize_text(str(item.get("prompt", "")))
            if not prompt:
                continue
            score = SequenceMatcher(None, normalized, prompt).ratio()
            overlap = len(set(normalized.split()) & set(prompt.split()))
            score += min(overlap / 8.0, 0.35)
            if item.get("intent") and item.get("intent") != "general_chat":
                if item.get("intent") in normalized:
                    score += 0.1
            if score > best_score:
                best_score = score
                best_intent = str(item.get("intent") or "")
        if best_score >= 0.56 and best_intent:
            return best_intent
        return None

    def _retrieve_knowledge(self, text: str, intent: str | None, profile: dict, limit: int = 3) -> list[dict]:
        normalized = _normalize_text(text)
        tokens = {token for token in normalized.split() if len(token) > 2 and token not in STOPWORDS}
        desired_goal = profile.get("goal")
        desired_diet = profile.get("diet_type")
        candidate_ids: set[int] = set()
        for token in tokens:
            candidate_ids.update(self.knowledge_index.get(token, []))
        if intent:
            candidate_ids.update(self.knowledge_by_intent.get(intent, []))
        if not candidate_ids:
            candidate_ids.update(range(min(len(self.knowledge), 200)))
        if intent in {"diet_advice", "nutrition_info"}:
            diet_terms = ("diet", "meal", "protein", "calorie", "food", "eat", "vegan", "vegetarian", "nutrition")
            filtered_ids = [
                idx
                for idx in candidate_ids
                if idx < len(self.knowledge)
                and self.knowledge[idx].get("kind") in {"nutrition_plan", "preference", "qa"}
                and any(term in f"{self.knowledge[idx].get('prompt', '')} {self.knowledge[idx].get('response', '')}".lower() for term in diet_terms)
            ]
            if filtered_ids:
                candidate_ids = set(filtered_ids)
        ranked: list[tuple[float, dict]] = []
        for idx in candidate_ids:
            if idx < 0 or idx >= len(self.knowledge):
                continue
            item = self.knowledge[idx]
            prompt = _normalize_text(str(item.get("prompt", "")))
            response = _normalize_text(str(item.get("response", "")))
            if not prompt or not response:
                continue
            if item.get("kind") == "intent_example":
                continue
            score = SequenceMatcher(None, normalized, prompt).ratio()
            prompt_tokens = {token for token in prompt.split() if len(token) > 2 and token not in STOPWORDS}
            score += min(len(tokens & prompt_tokens) / 5.0, 0.4)
            if intent and item.get("intent") == intent:
                score += 0.35
            if intent in {"diet_advice", "nutrition_info"} and item.get("kind") in {"nutrition_plan", "nutrition_qa"}:
                score += 0.3
            if intent in {"workout_advice", "plan_request"} and item.get("kind") in {"qa", "conversation", "preference"}:
                score += 0.2
            if intent == "weekly_schedule" and item.get("kind") in {"qa", "conversation"}:
                score += 0.2
            if desired_goal == "muscle_gain" and any(word in (prompt + " " + response) for word in ("protein", "muscle", "strength", "lift")):
                score += 0.15
            if desired_goal == "fat_burn" and any(word in (prompt + " " + response) for word in ("fat", "weight", "calorie", "cardio")):
                score += 0.15
            if desired_diet == "vegan" and any(word in (prompt + " " + response) for word in ("vegan", "tofu", "lentil", "bean", "soy", "chickpea")):
                score += 0.2
            if desired_diet == "vegetarian" and any(word in (prompt + " " + response) for word in ("vegetarian", "egg", "dairy", "tofu", "lentil")):
                score += 0.2
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (pair[0], len(str(pair[1].get("response", "")))), reverse=True)
        seen: set[tuple[str, str]] = set()
        results: list[dict] = []
        for score, item in ranked:
            key = (item["prompt"].lower(), item["response"].lower())
            if key in seen:
                continue
            seen.add(key)
            results.append({**item, "score": round(score, 3)})
            if len(results) >= limit:
                break
        return results

    def _generate_knowledge_reply(self, text: str, profile: dict, intent: str) -> tuple[str, list[dict]]:
        matches = self._retrieve_knowledge(text, intent, profile)
        if not matches:
            return "", []

        top = matches[0]
        response = str(top.get("response", "")).strip()
        if not response:
            return "", matches

        summary = response
        if len(summary) > 240:
            cut = summary[:240]
            last_period = cut.rfind(".")
            summary = cut[: last_period + 1] if last_period >= 80 else cut.rsplit(" ", 1)[0] + "..."

        if intent in {"diet_advice", "nutrition_info"}:
            prefix = "Here is a more grounded diet answer based on the nutrition knowledge base: "
        elif intent in {"workout_advice", "plan_request"}:
            prefix = "Here is a more grounded fitness answer based on the exercise knowledge base: "
        elif intent == "weekly_schedule":
            prefix = "Here is a schedule answer grounded in the fitness knowledge base: "
        else:
            prefix = "Here is a dataset-grounded answer: "

        if summary:
            summary = summary[0].upper() + summary[1:]
        return prefix + summary, matches

    def _feedback_weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for item in self.feedback:
            if not isinstance(item, dict):
                continue
            key = _normalize_text(str(item.get("name", "")))
            if not key:
                continue
            rating = str(item.get("rating", "")).lower().strip()
            delta = 1.0 if rating in {"up", "like", "liked", "good"} else -1.0
            weights[key] = weights.get(key, 0.0) + delta
        return weights

    def _exercise_feedback_bonus(self, exercise_name: str) -> float:
        key = _normalize_text(exercise_name)
        best = 0.0
        for candidate, weight in self._feedback_weights().items():
            if candidate and (candidate == key or candidate in key or key in candidate):
                best = max(best, weight)
        return best

    def _nutrition_feedback_bonus(self, food_name: str) -> float:
        key = _normalize_text(food_name)
        best = 0.0
        for candidate, weight in self._feedback_weights().items():
            if candidate and (candidate == key or candidate in key or key in candidate):
                best = max(best, weight)
        return best

    def _parse_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip().lower().replace(",", "")
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _nutrition_picks(self, profile: dict) -> list[dict]:
        goal = profile.get("goal")
        bmi = profile.get("bmi_category")
        diet_type = profile.get("diet_type")
        ranked = [item for item in self.nutrition if item.get("name")]
        recent_foods = self._recent_nutrition_keys(profile)

        muscle_food_keywords = (
            "egg",
            "milk",
            "chicken",
            "salmon",
            "fish",
            "beef",
            "turkey",
            "yogurt",
            "oats",
            "rice",
            "peanut",
            "nuts",
            "beans",
        )
        vegan_keywords = (
            "tofu",
            "tempeh",
            "lentil",
            "chickpea",
            "soy",
            "edamame",
            "bean",
            "oat",
            "seed",
            "nuts",
            "almond",
            "rice",
            "quinoa",
            "avocado",
            "spinach",
            "broccoli",
            "pea",
        )
        fat_loss_keywords = (
            "apple",
            "banana",
            "berry",
            "broccoli",
            "spinach",
            "carrot",
            "lettuce",
            "cucumber",
            "yogurt",
            "fish",
            "chicken",
            "lentil",
            "bean",
            "oats",
            "brown rice",
            "whole wheat",
        )
        avoid_words = (
            "isolate",
            "powder",
            "gelatin",
            "supplement",
            "protein shake",
            "candies",
            "candy",
            "chocolate",
            "soup",
            "cookie",
            "cake",
            "ice cream",
            "dessert",
            "drink mix",
            "gelatin",
            "beef",
            "chicken",
            "turkey",
            "pork",
            "fish",
            "salmon",
            "tuna",
            "shrimp",
            "egg",
            "milk",
            "cheese",
            "yogurt",
            "whey",
        )

        def score(item: dict) -> tuple:
            calories = item.get("calories") or 0.0
            protein = item.get("protein") or 0.0
            fiber = item.get("fiber") or 0.0
            name = str(item.get("name", "")).lower()
            penalties = sum(1 for word in avoid_words if word in name)
            feedback_bonus = self._nutrition_feedback_bonus(str(item.get("name", "")))
            vegan_penalty = 0
            vegan_focus_bonus = 0
            if diet_type == "vegan":
                vegan_penalty = sum(
                    1
                    for word in (
                        "beef",
                        "chicken",
                        "turkey",
                        "pork",
                        "fish",
                        "salmon",
                        "tuna",
                        "shrimp",
                        "egg",
                        "milk",
                        "cheese",
                        "yogurt",
                        "whey",
                        "butter",
                        "cream",
                        "honey",
                    )
                    if word in name
                )
                vegan_focus_bonus = sum(2 for word in vegan_keywords if word in name)
            if goal == "muscle_gain" or bmi == "Underweight":
                bonus = sum(1 for word in muscle_food_keywords if word in name)
                return (penalties + vegan_penalty, -vegan_focus_bonus, -feedback_bonus, -bonus, -protein, -calories, item["name"])
            if goal == "fat_burn" or bmi in {"Overweight", "Obesity"}:
                bonus = sum(1 for word in fat_loss_keywords if word in name)
                return (penalties + vegan_penalty, -vegan_focus_bonus, -feedback_bonus, calories, -bonus, -fiber, item["name"])
            bonus = sum(1 for word in muscle_food_keywords if word in name)
            return (penalties + vegan_penalty, -vegan_focus_bonus, -feedback_bonus, -bonus, -protein, calories, item["name"])

        ranked.sort(key=score)
        picks = [item for item in ranked if str(item.get("name", "")).strip().lower() not in recent_foods][:4]
        if len(picks) < 4:
            for item in ranked:
                if item in picks:
                    continue
                picks.append(item)
                if len(picks) >= 4:
                    break
        return [
            {
                "name": item["name"],
                "calories": item.get("calories"),
                "protein": item.get("protein"),
                "fiber": item.get("fiber"),
            }
            for item in picks
        ]

    def _dedupe_exercises(self, exercises: list[dict]) -> list[dict]:
        seen: set[Tuple[str, str]] = set()
        deduped: list[dict] = []
        for exercise in exercises:
            key = (exercise.get("name", ""), exercise.get("muscles", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(exercise)
        return deduped

    def _recent_exercise_keys(self, profile: dict, limit: int = 12) -> set[str]:
        recent: set[str] = set()
        for item in reversed(self.memory.get("interactions", []) if isinstance(self.memory.get("interactions"), list) else []):
            if not isinstance(item, dict):
                continue
            item_profile = item.get("profile", {})
            if not isinstance(item_profile, dict):
                continue
            if profile.get("goal") and item_profile.get("goal") != profile.get("goal"):
                continue
            if profile.get("bmi_category") and item_profile.get("bmi_category") != profile.get("bmi_category"):
                continue
            for exercise in item.get("exercise_examples", []) or []:
                if isinstance(exercise, dict):
                    key = str(exercise.get("exerciseId") or exercise.get("name") or "").strip().lower()
                    if key:
                        recent.add(key)
                        if len(recent) >= limit:
                            return recent
        return recent

    def _recent_nutrition_keys(self, profile: dict, limit: int = 12) -> set[str]:
        recent: set[str] = set()
        for item in reversed(self.memory.get("interactions", []) if isinstance(self.memory.get("interactions"), list) else []):
            if not isinstance(item, dict):
                continue
            item_profile = item.get("profile", {})
            if not isinstance(item_profile, dict):
                continue
            if profile.get("goal") and item_profile.get("goal") != profile.get("goal"):
                continue
            if profile.get("bmi_category") and item_profile.get("bmi_category") != profile.get("bmi_category"):
                continue
            for food in item.get("nutrition_examples", []) or []:
                if isinstance(food, dict):
                    key = str(food.get("name") or "").strip().lower()
                    if key:
                        recent.add(key)
                        if len(recent) >= limit:
                            return recent
        return recent

    def _pick_exercises(self, profile: dict) -> list[dict]:
        goal = profile.get("goal")
        bmi_category = profile.get("bmi_category")
        feedback_bonus = self._exercise_feedback_bonus
        pool = self.visual_exercises + self.exercises
        pool = self._dedupe_exercises(pool)
        recent_keys = self._recent_exercise_keys(profile)

        if goal == "fat_burn":
            ranked = sorted(
                pool,
                key=lambda exercise: (
                    -_exercise_score(exercise, set(), prefer_cardio=True),
                    -(3 if exercise.get("gif_url") else 0),
                    -feedback_bonus(str(exercise.get("name", ""))),
                    -len(exercise.get("name", "")),
                    str(exercise.get("name", "")),
                ),
            )
            ranked = [exercise for exercise in ranked if exercise.get("instruction")] or ranked
            low_impact = [e for e in ranked if any(token in str(e.get("name", "")).lower() for token in ("bike", "row", "elliptical", "stair", "walk"))]
            ordered = (low_impact[:2] + ranked[:1])[:3] if bmi_category == "Obesity" and low_impact else ranked[:3]
            picks = [exercise for exercise in ordered if str(exercise.get("exerciseId") or exercise.get("name") or "").strip().lower() not in recent_keys]
            if len(picks) < 3:
                for exercise in ranked:
                    key = str(exercise.get("exerciseId") or exercise.get("name") or "").strip().lower()
                    if key in recent_keys or exercise in picks:
                        continue
                    picks.append(exercise)
                    if len(picks) >= 3:
                        break
            if len(picks) < 3:
                for exercise in ordered:
                    if exercise not in picks:
                        picks.append(exercise)
                    if len(picks) >= 3:
                        break
            return picks

        muscle_keywords = {
            "squat": {"glute", "quad", "thigh", "hamstring"},
            "press": {"chest", "shoulder", "tricep"},
            "row": {"back", "lat", "trap", "shoulder"},
            "pull": {"lat", "bicep", "back"},
            "curl": {"bicep", "forearm"},
            "lunge": {"glute", "quad", "hamstring"},
            "deadlift": {"lower back", "hamstring", "glute"},
            "plank": {"abdominal", "core"},
            "bench": {"chest", "tricep", "shoulder"},
        }

        desired = set()
        if bmi_category == "Underweight":
            desired.update({"glute", "quad", "hamstring", "chest", "lat", "bicep", "tricep", "abdominal"})
        else:
            desired.update({"chest", "lat", "bicep", "tricep", "quad", "glute", "hamstring", "abdominal"})

        ranked = sorted(
            pool,
            key=lambda exercise: (
                -(
                    _exercise_score(exercise, desired)
                    + sum(2 for token, muscles in muscle_keywords.items() if token in str(exercise.get("name", "")).lower() and muscles)
                    + feedback_bonus(str(exercise.get("name", "")))
                    + (3 if exercise.get("gif_url") else 0)
                    + (2 if exercise.get("primaryMuscles") else 0)
                    + (1 if exercise.get("secondaryMuscles") else 0)
                ),
                -len(exercise.get("primaryMuscles") or []),
                -len(exercise.get("secondaryMuscles") or []),
                str(exercise.get("name", "")),
            ),
        )
        ranked = [exercise for exercise in ranked if exercise.get("instruction")] or ranked
        picks = [
            exercise
            for exercise in ranked
            if _exercise_score(exercise, desired) > 0
            and str(exercise.get("exerciseId") or exercise.get("name") or "").strip().lower() not in recent_keys
        ][:3]
        if len(picks) < 3:
            for exercise in ranked:
                key = str(exercise.get("exerciseId") or exercise.get("name") or "").strip().lower()
                if key in recent_keys or exercise in picks:
                    continue
                picks.append(exercise)
                if len(picks) >= 3:
                    break
        if len(picks) < 3:
            for exercise in ranked:
                if exercise not in picks:
                    picks.append(exercise)
                if len(picks) >= 3:
                    break
        return picks[:3]

    def recommend(self, payload: dict) -> dict:
        self.refresh_if_needed()
        profile = {
            "gender": payload.get("gender"),
            "goal": payload.get("goal"),
            "bmi_category": payload.get("bmi_category"),
            "diet_type": payload.get("diet_type"),
        }

        description = str(payload.get("description", "") or "")
        inferred = _extract_profile_from_text(description)
        parsed = _extract_intent(description)
        for key, value in inferred.items():
            if not profile.get(key):
                profile[key] = value

        requested_profile = dict(profile)
        knowledge_reply, knowledge_matches = self._generate_knowledge_reply(description, requested_profile, parsed.get("intent") or "plan_request")
        requested_diet_only = bool(requested_profile.get("diet_type")) and not requested_profile.get("goal") and not requested_profile.get("bmi_category")

        if requested_diet_only:
            if requested_profile["diet_type"] == "vegan":
                meal_guidance = "Use vegan protein sources like tofu, tempeh, beans, lentils, soy milk, chickpeas, oats, nuts, seeds, and fortified foods."
                reply = "Here is a vegan diet focus. " + meal_guidance
            else:
                meal_guidance = "Use vegetarian protein sources like eggs or dairy if you eat them, plus beans, lentils, tofu, tempeh, nuts, and seeds."
                reply = "Here is a vegetarian diet focus. " + meal_guidance
            nutrition_examples = self._nutrition_picks(requested_profile)
            self.memory["profile"] = requested_profile
            interactions = self.memory.setdefault("interactions", [])
            interactions.append(
                {
                    "type": "plan",
                    "profile": requested_profile,
                    "message": description,
                    "reply": reply,
                }
            )
            self.memory["interactions"] = interactions[-100:]
            self._memory_changed = True
            self._save_memory()
            return {
                "ok": True,
                "kind": "plan",
                "reply": reply,
                "profile": requested_profile,
                "defaults_used": [],
                "weekly_guidance": "Tell me your fitness goal if you want workout suggestions too.",
                "meal_guidance": meal_guidance,
                "exercise_examples": [],
                "nutrition_examples": nutrition_examples,
                "suggestions": [
                    "What are good vegan protein foods?",
                    "Give me a vegan meal plan",
                    "What exercise should I do?",
                ],
                "recommendation": None,
                "parsed": parsed,
                "knowledge_examples": knowledge_matches,
            }

        memory_profile = self.memory.get("profile", {}) if isinstance(self.memory.get("profile"), dict) else {}
        for key in ("gender", "goal", "bmi_category", "diet_type"):
            if not profile.get(key) and memory_profile.get(key):
                profile[key] = memory_profile.get(key)

        defaults_used: list[str] = []
        if not profile.get("gender"):
            profile["gender"] = DEFAULT_GENDER
            defaults_used.append("gender")

        if not profile.get("goal") and not profile.get("diet_type") and not profile.get("bmi_category"):
            return {
                "ok": False,
                "kind": "clarify",
                "missing": ["goal", "bmi_category"],
                "reply": "Tell me your goal and body type in one sentence, like: 'I am skinny and want to build muscles.'",
            }

        recommendation = self.advisor.recommend(
            gender=profile["gender"],
            goal=profile["goal"],
            bmi_category=profile["bmi_category"],
        )
        exercises = self._pick_exercises(profile)

        weekly_guidance = WEEKLY_GUIDANCE.get(profile["goal"], WEEKLY_GUIDANCE["fat_burn"])
        meal_guidance = MEAL_GUIDANCE.get(profile["goal"], MEAL_GUIDANCE["fat_burn"])
        if profile.get("diet_type") == "vegan":
            meal_guidance = "Use vegan protein sources like tofu, tempeh, beans, lentils, soy milk, chickpeas, oats, nuts, seeds, and fortified foods."
        elif profile.get("diet_type") == "vegetarian":
            meal_guidance = "Use vegetarian protein sources like eggs or dairy if you eat them, plus beans, lentils, tofu, tempeh, nuts, and seeds."
        if profile["bmi_category"] == "Underweight":
            meal_guidance = "You likely need more calories and protein than a fat-loss plan. Keep meals frequent and nutrient-dense."
        elif profile["bmi_category"] == "Obesity":
            meal_guidance = "Use a gentle calorie deficit with nutrient-dense foods and portion control, while keeping protein high."

        nutrition_examples = self._nutrition_picks(profile)
        reply = (
            f"Here is your plan for a {profile['goal'].replace('_', ' ')} goal. "
            f"{weekly_guidance} "
            f"{meal_guidance}"
        )
        if knowledge_reply:
            reply = f"{reply} {knowledge_reply}"

        self.memory["profile"] = profile
        interactions = self.memory.setdefault("interactions", [])
        interactions.append(
            {
                "type": "plan",
                "profile": profile,
                "message": description,
                "reply": reply,
                "exercise_examples": exercises,
                "nutrition_examples": nutrition_examples,
                "knowledge_examples": knowledge_matches,
            }
        )
        self.memory["interactions"] = interactions[-100:]
        self._memory_changed = True
        self._save_memory()

        return {
            "ok": True,
            "kind": "plan",
            "reply": reply,
            "profile": profile,
            "defaults_used": defaults_used,
            "weekly_guidance": weekly_guidance,
            "meal_guidance": meal_guidance,
            "exercise_examples": exercises,
            "nutrition_examples": nutrition_examples,
            "suggestions": self._suggestions(profile, "plan"),
            "recommendation": recommendation_to_dict(recommendation),
            "parsed": parsed,
            "knowledge_examples": knowledge_matches,
        }

    def _answer_topic(self, message: str, profile: dict) -> dict:
        text = _normalize_text(message)
        parsed = _extract_intent(message)
        knowledge_intent = parsed.get("intent") or self._intent_from_knowledge(message) or "general_chat"
        knowledge_reply, knowledge_matches = self._generate_knowledge_reply(message, profile, knowledge_intent)

        if any(keyword in text for keyword in ("how many days", "how often", "schedule", "week", "weekly")):
            goal = profile.get("goal") or _extract_profile_from_text(text).get("goal") or "fat_burn"
            reply = knowledge_reply or WEEKLY_GUIDANCE.get(goal, WEEKLY_GUIDANCE["fat_burn"])
            if not knowledge_reply:
                reply = WEEKLY_GUIDANCE.get(goal, WEEKLY_GUIDANCE["fat_burn"])
            elif goal == "muscle_gain":
                reply += " For muscle gain, keep most days focused on strength work and progressive overload."
            elif goal == "fat_burn":
                reply += " For fat loss, mix steady cardio with 2 strength days to keep muscle."
            return {"ok": True, "kind": "answer", "reply": reply, "exercise_examples": [], "nutrition_examples": self._nutrition_picks(profile), "suggestions": self._suggestions(profile, "answer"), "parsed": parsed, "knowledge_examples": knowledge_matches}

        if any(keyword in text for keyword in ("exercise", "workout", "routine", "training", "gym", "show me exercises", "show exercise images")):
            goal = profile.get("goal") or _extract_profile_from_text(text).get("goal")
            bmi = profile.get("bmi_category") or _extract_profile_from_text(text).get("bmi_category")
            exercise_profile = {"goal": goal or "muscle_gain", "bmi_category": bmi}
            exercises = self._pick_exercises(exercise_profile)
            if goal == "fat_burn" or bmi in {"Overweight", "Obesity"}:
                reply = "Here are exercise ideas that fit a fat-loss focus. I mixed in cardio and strength work."
            elif bmi == "Underweight" or goal == "muscle_gain":
                reply = "Here are exercise ideas that fit a muscle-gain or underweight profile."
            else:
                reply = "Here are exercise ideas based on your current goal."
            if knowledge_reply:
                reply = knowledge_reply
            return {
                "ok": True,
                "kind": "answer",
                "reply": reply,
                "exercise_examples": exercises,
                "nutrition_examples": self._nutrition_picks(exercise_profile),
                "suggestions": self._suggestions(exercise_profile, "answer"),
                "parsed": parsed,
                "knowledge_examples": knowledge_matches,
            }

        if any(keyword in text for keyword in ("protein", "calorie", "diet", "meal", "food", "eat", "nutrition")):
            goal = profile.get("goal") or _extract_profile_from_text(text).get("goal")
            bmi = profile.get("bmi_category") or _extract_profile_from_text(text).get("bmi_category")
            diet_type = profile.get("diet_type") or _extract_profile_from_text(text).get("diet_type")
            if diet_type == "vegan":
                if goal == "muscle_gain":
                    reply = "For a vegan muscle-gain diet, focus on tofu, tempeh, beans, lentils, chickpeas, soy milk, oats, nuts, seeds, and enough calories."
                elif bmi == "Underweight":
                    reply = "For a vegan underweight diet, focus on more calories, regular meals, tofu, tempeh, beans, lentils, chickpeas, soy milk, oats, nuts, seeds, and fortified foods."
                else:
                    reply = "For a vegan diet, focus on tofu, tempeh, beans, lentils, chickpeas, soy milk, oats, nuts, seeds, and fortified foods."
                if knowledge_reply:
                    reply = knowledge_reply
                if goal == "muscle_gain":
                    reply += " For muscle gain, build each meal around a solid protein source and enough total calories."
                return {"ok": True, "kind": "answer", "reply": reply, "exercise_examples": [], "nutrition_examples": self._nutrition_picks({**profile, "diet_type": "vegan", "goal": goal}), "suggestions": self._suggestions(profile, "answer"), "parsed": parsed, "knowledge_examples": knowledge_matches}
            if diet_type == "vegetarian":
                if goal == "muscle_gain":
                    reply = "For a vegetarian muscle-gain diet, use eggs or dairy if you eat them, plus beans, lentils, tofu, tempeh, nuts, seeds, and enough calories."
                elif bmi == "Underweight":
                    reply = "For a vegetarian underweight diet, eat frequently and use eggs or dairy if you eat them, plus beans, lentils, tofu, tempeh, nuts, and seeds."
                else:
                    reply = "For a vegetarian diet, use eggs or dairy if you eat them, plus beans, lentils, tofu, tempeh, nuts, and seeds."
                if knowledge_reply:
                    reply = knowledge_reply
                if goal == "muscle_gain":
                    reply += " For muscle gain, aim for protein at every meal and enough calories to recover."
                return {"ok": True, "kind": "answer", "reply": reply, "exercise_examples": [], "nutrition_examples": self._nutrition_picks({**profile, "diet_type": "vegetarian", "goal": goal}), "suggestions": self._suggestions(profile, "answer"), "parsed": parsed, "knowledge_examples": knowledge_matches}
            if bmi == "Underweight":
                reply = "For underweight users, focus on more calories, enough protein, and regular meals."
            elif bmi == "Obesity" or goal == "fat_burn":
                reply = "For fat loss, keep meals nutrient-dense, watch portions, and keep protein high."
            else:
                reply = "A balanced diet with protein, vegetables, fruit, and whole grains is a strong default."
            if knowledge_reply:
                reply = knowledge_reply
            if goal == "muscle_gain":
                reply += " For muscle gain, prioritize protein, carbs, and consistent meal timing."
            return {"ok": True, "kind": "answer", "reply": reply, "exercise_examples": [], "nutrition_examples": self._nutrition_picks(profile), "suggestions": self._suggestions(profile, "answer"), "parsed": parsed, "knowledge_examples": knowledge_matches}

        if any(keyword in text for keyword in ("underweight", "skinny", "thin", "gain muscle", "muscle", "build muscle")):
            inferred = _extract_profile_from_text(text)
            goal = inferred.get("goal") or profile.get("goal")
            bmi = inferred.get("bmi_category") or profile.get("bmi_category")
            if bmi == "Underweight" and goal == "muscle_gain":
                reply = "If you are skinny and want muscle, lift 2 to 4 times a week, eat more calories, and get enough protein."
            elif bmi == "Underweight":
                reply = "If you are underweight, prioritize calorie-dense meals, enough protein, and progressive strength training."
            else:
                reply = "Progressive overload, enough protein, and consistent training are key for muscle gain."
            exercises = self._pick_exercises({"goal": "muscle_gain", "bmi_category": bmi})
            if knowledge_reply:
                reply = knowledge_reply
            return {"ok": True, "kind": "answer", "reply": reply, "exercise_examples": exercises, "nutrition_examples": self._nutrition_picks({"goal": "muscle_gain", "bmi_category": bmi}), "suggestions": self._suggestions({"goal": "muscle_gain", "bmi_category": bmi}, "answer"), "parsed": parsed, "knowledge_examples": knowledge_matches}

        if any(keyword in text for keyword in ("hiit", "cardio", "running", "bike", "row", "elliptical", "fat burn", "lose weight")):
            reply = "For fat loss, combine regular cardio with strength training so you keep muscle while burning calories."
            exercises = self._pick_exercises({"goal": "fat_burn", "bmi_category": profile.get("bmi_category")})
            if knowledge_reply:
                reply = knowledge_reply
            return {"ok": True, "kind": "answer", "reply": reply, "exercise_examples": exercises, "nutrition_examples": self._nutrition_picks({"goal": "fat_burn", "bmi_category": profile.get("bmi_category")}), "suggestions": self._suggestions({"goal": "fat_burn", "bmi_category": profile.get("bmi_category")}, "answer"), "parsed": parsed, "knowledge_examples": knowledge_matches}

        return {
            "ok": True,
            "kind": "answer",
            "reply": knowledge_reply or "I can help with workout plans, muscle gain, fat loss, meal guidance, and exercise ideas. Try telling me your goal or body type.",
            "exercise_examples": [],
            "nutrition_examples": self._nutrition_picks(profile),
            "suggestions": self._suggestions(profile, "answer"),
            "parsed": parsed,
            "knowledge_examples": knowledge_matches,
        }

    def _memory_match(self, message: str) -> Optional[dict]:
        text = _normalize_text(message)
        best: Optional[dict] = None
        best_score = 0.0
        for item in reversed(self.memory.get("interactions", []) if isinstance(self.memory.get("interactions"), list) else []):
            if not isinstance(item, dict):
                continue
            other = _normalize_text(str(item.get("message", "")))
            if not other:
                continue
            score = SequenceMatcher(None, text, other).ratio()
            if score > best_score:
                best_score = score
                best = item
        if best_score >= 0.88:
            return best
        return None

    def _suggestions(self, profile: dict, kind: str) -> list[str]:
        goal = profile.get("goal")
        bmi = profile.get("bmi_category")
        if kind == "plan":
            if goal == "muscle_gain":
                return [
                    "Show me the workout routine",
                    "What should I eat for muscle gain?",
                    "Give me a weekly schedule",
                ]
            if goal == "fat_burn":
                return [
                    "Show me the cardio exercises",
                    "What should I eat to lose fat?",
                    "Give me a weekly fat-loss schedule",
                ]
            return [
                "Show me exercise details",
                "What should I eat?",
                "How many days should I train?",
            ]
        if kind == "answer":
            if goal == "muscle_gain" or bmi == "Underweight":
                return [
                    "Give me my workout plan",
                    "Show exercise images",
                    "What foods should I eat?",
                ]
            return [
                "Give me a workout plan",
                "Show exercise images",
                "What should I eat?",
            ]
        return [
            "Give me a workout plan",
            "What should I eat?",
            "Show exercise images",
        ]

    def record_feedback(self, payload: dict) -> dict:
        rating = str(payload.get("rating", "")).lower().strip()
        if rating not in {"up", "down", "like", "dislike"}:
            return {"ok": False, "error": "invalid_rating"}
        entry = {
            "rating": "up" if rating in {"up", "like"} else "down",
            "name": payload.get("name"),
            "reply": payload.get("reply"),
            "message": payload.get("message"),
            "profile": payload.get("profile", {}),
            "created_at": payload.get("created_at"),
        }
        self.feedback.append(entry)
        self.feedback = self.feedback[-500:]
        self._feedback_changed = True
        self._save_feedback()
        return {"ok": True, "message": "Feedback saved.", "feedback_count": len(self.feedback)}

    def chat(self, payload: dict) -> dict:
        self.refresh_if_needed()
        message = str(payload.get("message", "") or "").strip()
        state = payload.get("state", {}) or {}
        profile = {
            "gender": state.get("profile", {}).get("gender") if isinstance(state.get("profile"), dict) else None,
            "goal": state.get("profile", {}).get("goal") if isinstance(state.get("profile"), dict) else None,
            "bmi_category": state.get("profile", {}).get("bmi_category") if isinstance(state.get("profile"), dict) else None,
            "diet_type": state.get("profile", {}).get("diet_type") if isinstance(state.get("profile"), dict) else None,
        }

        incoming_profile = {
            "gender": payload.get("gender"),
            "goal": payload.get("goal"),
            "bmi_category": payload.get("bmi_category"),
            "diet_type": payload.get("diet_type"),
        }
        for key, value in incoming_profile.items():
            if value:
                profile[key] = value

        description = str(payload.get("description", "") or "")
        inferred = _extract_profile_from_text(f"{message} {description}")
        for key, value in inferred.items():
            if not profile.get(key):
                profile[key] = value

        requested_profile = dict(profile)
        parsed = _extract_intent(f"{message} {description}")

        text = _normalize_text(message)
        memory_hit = self._memory_match(message)
        repeat_intent = any(keyword in text for keyword in ("repeat", "same", "previous", "that plan", "that answer", "as before", "last one"))
        if memory_hit and memory_hit.get("reply") and repeat_intent:
            if any(profile.values()):
                self.memory["profile"] = profile
            interactions = self.memory.setdefault("interactions", [])
            interactions.append({"type": "memory", "message": message, "profile": profile, "reply": memory_hit.get("reply")})
            self.memory["interactions"] = interactions[-100:]
            self._memory_changed = True
            self._save_memory()
            return {
                "ok": True,
                "kind": "memory",
                "reply": str(memory_hit.get("reply")),
                "profile": profile,
                "defaults_used": [],
                "exercise_examples": [],
                "nutrition_examples": self._nutrition_picks(profile),
                "suggestions": self._suggestions(profile, "answer"),
                "parsed": parsed,
            }

        memory_profile = self.memory.get("profile", {}) if isinstance(self.memory.get("profile"), dict) else {}
        for key in ("gender", "goal", "bmi_category", "diet_type"):
            if not profile.get(key) and memory_profile.get(key):
                profile[key] = memory_profile.get(key)

        question_mode = parsed.get("intent") in {"diet_advice", "workout_advice", "weekly_schedule"} or (
            any(
                keyword in text
                for keyword in ("what should", "how many", "how often", "what can i", "tell me", "advice", "help me", "should i", "what do i")
            )
            and "plan" not in text
        )

        if question_mode:
            answer = self._answer_topic(message or description, profile)
            answer.setdefault("profile", profile)
            answer.setdefault("defaults_used", [])
            answer.setdefault("suggestions", self._suggestions(profile, "answer"))
            answer.setdefault("parsed", parsed)
            interactions = self.memory.setdefault("interactions", [])
            interactions.append(
                {
                    "type": "question",
                    "message": message,
                    "profile": profile,
                    "reply": answer.get("reply"),
                    "exercise_examples": answer.get("exercise_examples", []),
                    "nutrition_examples": answer.get("nutrition_examples", []),
                }
            )
            self.memory["interactions"] = interactions[-100:]
            self._memory_changed = True
            self._save_memory()
            return answer

        plan_keywords = ("plan", "routine", "recommend", "build", "cut", "lose", "gain", "fat", "muscle", "skinny", "bulk", "diet", "vegan", "vegetarian")
        want_plan = parsed.get("intent") == "plan_request" or any(keyword in _normalize_text(message) for keyword in plan_keywords)
        if want_plan and requested_profile.get("diet_type") and not requested_profile.get("goal") and not requested_profile.get("bmi_category"):
            return self.recommend({"gender": profile.get("gender"), "diet_type": requested_profile.get("diet_type"), "diet_only_request": True, "description": description})

        if want_plan and (profile.get("goal") or profile.get("bmi_category") or profile.get("diet_type") or description):
            return self.recommend({**payload, "gender": profile.get("gender"), "goal": profile.get("goal"), "bmi_category": profile.get("bmi_category"), "diet_type": profile.get("diet_type")})

        if want_plan and not profile.get("goal"):
            return {
                "ok": True,
                "kind": "clarify",
                "reply": "Tell me your goal and body type in one sentence, for example: 'I am skinny and want to build muscles.'",
                "profile": profile,
                "exercise_examples": [],
            }

        answer = self._answer_topic(message or description, profile)
        answer.setdefault("profile", profile)
        answer.setdefault("defaults_used", [])
        answer.setdefault("suggestions", self._suggestions(profile, "answer"))
        answer.setdefault("parsed", parsed)
        interactions = self.memory.setdefault("interactions", [])
        interactions.append(
            {
                "type": "question",
                "message": message,
                "profile": profile,
                "reply": answer.get("reply"),
                "exercise_examples": answer.get("exercise_examples", []),
                "nutrition_examples": answer.get("nutrition_examples", []),
            }
        )
        self.memory["interactions"] = interactions[-100:]
        if any(profile.values()):
            self.memory["profile"] = profile
        self._memory_changed = True
        self._save_memory()
        return answer

    def retrain(self) -> dict:
        self.refresh(force=True)
        return {
            "ok": True,
            "message": "Assistant refreshed from local datasets.",
            "exercise_count": len(self.exercises),
            "nutrition_count": len(self.nutrition),
            "visual_exercise_count": len(self.visual_exercises),
        }

    def form_options(self) -> dict:
        return {
            "gender": [
                {"value": "", "label": "Optional"},
                {"value": "Male", "label": "Male"},
                {"value": "Female", "label": "Female"},
            ],
            "goal": [
                {"value": "", "label": "Optional"},
                {"value": "fat_burn", "label": "Fat burn"},
                {"value": "muscle_gain", "label": "Muscle gain"},
            ],
            "bmi_category": [
                {"value": "", "label": "Optional"},
                {"value": "Underweight", "label": "Underweight"},
                {"value": "Normal weight", "label": "Normal weight"},
                {"value": "Overweight", "label": "Overweight"},
                {"value": "Obesity", "label": "Obesity"},
            ],
            "diet_type": [
                {"value": "", "label": "Optional"},
                {"value": "vegan", "label": "Vegan"},
                {"value": "vegetarian", "label": "Vegetarian"},
            ],
        }


def build_summary(csv_path: str | Path) -> dict[str, Any]:
    rows = _csv_rows(Path(csv_path))
    if not rows:
        return {"rows": 0, "unique_values": {}}
    unique_counts = {column: len({row[column] for row in rows}) for column in rows[0].keys()}
    return {"rows": len(rows), "unique_values": unique_counts}
