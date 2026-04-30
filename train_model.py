"""Train the gym recommendation model and save it to disk."""

from __future__ import annotations

import json
from pathlib import Path

from gym_ai import GymAdvisor, MODEL_FILENAME, build_training_summary


def main() -> None:
    csv_path = Path("GYM.csv")
    model_path = Path(MODEL_FILENAME)

    advisor = GymAdvisor.from_csv(csv_path)
    advisor.save(model_path)

    summary = build_training_summary(csv_path)
    print(json.dumps(
        {
            "status": "trained",
            "model_path": str(model_path),
            "summary": summary,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
