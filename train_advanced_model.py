"""Train an advanced open-source transformer model on FitPax datasets.

This script fine-tunes a SentenceTransformer model (all-MiniLM-L6-v2) 
using the local GYM.csv and knowledge.jsonl datasets.
"""

import json
import pandas as pd
from pathlib import Path
import os

# Check for dependencies
try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    import torch
except ImportError:
    print("Error: Required libraries not found.")
    print("Please run: pip install torch sentence-transformers pandas")
    exit(1)

def train_model():
    # Configuration
    base_dir = Path(__file__).resolve().parent
    knowledge_path = base_dir / "data" / "knowledge" / "knowledge.jsonl"
    gym_csv_path = base_dir / "GYM.csv"
    output_path = base_dir / "fitpax_trained_model"
    
    print(f"--- Loading Open-Source Model: all-MiniLM-L6-v2 ---")
    # This downloads the model from Hugging Face if not present
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 1. Prepare Training Data
    train_examples = []
    
    # Load knowledge.jsonl (QA pairs)
    if knowledge_path.exists():
        print(f"Loading knowledge from {knowledge_path}...")
        with open(knowledge_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "prompt" in item and "response" in item:
                        # Map prompt to response
                        train_examples.append(InputExample(texts=[item['prompt'], item['response']]))
                except json.JSONDecodeError:
                    continue
    else:
        print(f"Warning: {knowledge_path} not found.")

    # Load GYM.csv (Recommendations)
    if gym_csv_path.exists():
        print(f"Loading gym recommendations from {gym_csv_path}...")
        gym_df = pd.read_csv(gym_csv_path)
        for _, row in gym_df.iterrows():
            # Associate goals and BMI with their plans
            goal_desc = f"I want to achieve {row['Goal']} and my BMI is {row['BMI Category']}"
            plan_desc = f"Recommended Workout: {row['Exercise Schedule']}. Recommended Diet: {row['Meal Plan']}."
            train_examples.append(InputExample(texts=[goal_desc, plan_desc]))

    # Load Exercise Datasets
    data_dir = base_dir / "data"
    for filename in ["strength.json", "cardio.json", "flexibility.json"]:
        path = data_dir / filename
        if path.exists():
            print(f"Loading exercises from {path}...")
            with open(path, "r", encoding="utf-8") as f:
                exercises = json.load(f)
                for ex in exercises:
                    name = ex.get("name", "")
                    instr = " ".join(ex.get("steps", []) or ex.get("instructions", []) or [])
                    if name and instr:
                        # Help the model associate exercise names with their descriptions
                        train_examples.append(InputExample(texts=[name, instr]))
    else:
        print(f"Warning: {gym_csv_path} not found.")

    if not train_examples:
        print("Error: No training data found. Check your datasets.")
        return

    # 2. Configure Training
    # MultipleNegativesRankingLoss is ideal for training models to retrieve 
    # the correct information from a large pool of data.
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 3. Start Training
    print(f"--- Starting training on {len(train_examples)} fitness patterns ---")
    print("This may take a few minutes depending on your hardware...")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        show_progress_bar=True,
        output_path=str(output_path)
    )

    print(f"\n--- SUCCESS ---")
    print(f"Advanced model saved to: {output_path}")
    print("You can now use this model in assistant_core.py for high-accuracy responses.")

if __name__ == "__main__":
    train_model()
