"""Gym recommendation engine built from the CSV dataset.

This first version is intentionally lightweight:
- It learns the exact mapping from (Gender, Goal, BMI Category)
  to Exercise Schedule and Meal Plan.
- It supports a small fallback strategy for unknown or messy inputs.
- It can save and load a JSON model so the app does not need to scan
  the CSV on every request.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MODEL_FILENAME = "gym_model.json"


def _normalize(value: Optional[str]) -> str:
    if value is None:
        return ""
    return " ".join(value.strip().lower().replace("_", " ").split())


def _canonical(value: Optional[str]) -> str:
    """Return a normalized key with underscores for API friendliness."""
    if value is None:
        return ""
    return _normalize(value).replace(" ", "_")


@dataclass(frozen=True)
class Recommendation:
    gender: str
    goal: str
    bmi_category: str
    exercise_schedule: str
    meal_plan: str
    confidence: float
    matched_on: str


class GymAdvisor:
    """Small recommender trained from the gym CSV."""

    def __init__(self, lookup: Dict[str, Dict[str, Dict[str, Dict[str, str]]]]) -> None:
        self.lookup = lookup
        self._index = self._build_index(lookup)

    @staticmethod
    def _build_index(
        lookup: Dict[str, Dict[str, Dict[str, Dict[str, str]]]]
    ) -> Dict[Tuple[str, str, str], Dict[str, str]]:
        index: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        for gender, goals in lookup.items():
            for goal, bmis in goals.items():
                for bmi_category, payload in bmis.items():
                    index[(gender, goal, bmi_category)] = payload
        return index

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "GymAdvisor":
        lookup: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                gender = _canonical(row["Gender"])
                goal = _canonical(row["Goal"])
                bmi_category = _canonical(row["BMI Category"])
                payload = {
                    "exercise_schedule": row["Exercise Schedule"].strip(),
                    "meal_plan": row["Meal Plan"].strip(),
                }
                lookup.setdefault(gender, {}).setdefault(goal, {})[bmi_category] = payload
        return cls(lookup)

    @classmethod
    def from_json(cls, model_path: str | Path) -> "GymAdvisor":
        with open(model_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(data["lookup"])

    def save(self, model_path: str | Path) -> None:
        data = {
            "lookup": self.lookup,
        }
        with open(model_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=True)

    def recommend(
        self,
        gender: str,
        goal: str,
        bmi_category: str,
    ) -> Recommendation:
        gender_key = _canonical(gender)
        goal_key = _canonical(goal)
        bmi_key = _canonical(bmi_category)

        exact = self._index.get((gender_key, goal_key, bmi_key))
        if exact:
            return Recommendation(
                gender=gender,
                goal=goal,
                bmi_category=bmi_category,
                exercise_schedule=exact["exercise_schedule"],
                meal_plan=exact["meal_plan"],
                confidence=1.0,
                matched_on="exact",
            )

        # Fallback: look for the closest partial match. This is useful if a
        # client sends a slightly different value or if the dataset expands.
        for key_name, key_value in (
            ("gender", gender_key),
            ("goal", goal_key),
            ("bmi_category", bmi_key),
        ):
            candidates = self._fallback_candidates(key_name, key_value)
            if candidates:
                best = candidates[0]
                return Recommendation(
                    gender=gender,
                    goal=goal,
                    bmi_category=bmi_category,
                    exercise_schedule=best["exercise_schedule"],
                    meal_plan=best["meal_plan"],
                    confidence=0.5,
                    matched_on=f"fallback:{key_name}",
                )

        raise ValueError(
            "No recommendation could be generated for the provided inputs."
        )

    def _fallback_candidates(self, key_name: str, value: str) -> List[Dict[str, str]]:
        matches: List[Dict[str, str]] = []
        if key_name == "gender":
            for goals in self.lookup.get(value, {}).values():
                matches.extend(goals.values())
        elif key_name == "goal":
            for gender_data in self.lookup.values():
                matches.extend(gender_data.get(value, {}).values())
        elif key_name == "bmi_category":
            for gender_data in self.lookup.values():
                for bmi_map in gender_data.values():
                    if value in bmi_map:
                        matches.append(bmi_map[value])
        return matches

    @classmethod
    def load(cls, csv_path: str | Path = "GYM.csv", model_path: str | Path = MODEL_FILENAME) -> "GymAdvisor":
        model_path = Path(model_path)
        csv_path = Path(csv_path)
        if model_path.exists():
            return cls.from_json(model_path)

        advisor = cls.from_csv(csv_path)
        advisor.save(model_path)
        return advisor


def build_training_summary(csv_path: str | Path) -> Dict[str, Any]:
    """Generate a small dataset summary for the README or a dashboard."""

    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    unique_counts = {
        column: len({row[column] for row in rows}) for column in reader.fieldnames or []
    }

    return {
        "rows": len(rows),
        "unique_values": unique_counts,
    }


def recommendation_to_dict(result: Recommendation) -> Dict[str, Any]:
    return asdict(result)
