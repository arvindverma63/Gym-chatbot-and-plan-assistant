# FitPax AI

FitPax AI is a compact gym chatbot and plan assistant built from your `GYM.csv` dataset, public exercise datasets, and simple fitness guidance.

## What it can do

- Chat about workouts, fat loss, muscle gain, meals, and gym questions
- Generate a full workout and meal plan from either structured fields or plain English
- Handle rough text like `i am very skinny and i want build mussles`
- Suggest real exercise examples from open exercise datasets
- Show GIF exercise demos from the Kaggle exercise dataset
- Auto-refresh when local data files change
- Remember the last learned profile and reuse it later
- Learn from chat history and feedback so rankings improve over time

## How it works

- `GYM.csv` provides the base plan mapping
- `data/*.json` provides open exercise examples
- Optional Kaggle downloads can add more exercises and nutrition data
- The assistant reloads local data automatically when you refresh or restart the server

## Files

- `assistant_core.py`: chatbot and recommendation brain
- `gym_ai.py`: original recommendation engine and model loader
- `server.py`: local web server and API
- `static/`: compact UI
- `sync_kaggle.py`: optional Kaggle download helper
- `sync_fitness_datasets.py`: downloads Hugging Face fitness and nutrition datasets
- `data/`: local exercise datasets
- `data/knowledge/`: normalized dataset-backed knowledge base
- `memory.json`: local assistant memory
- `feedback.json`: feedback used to improve future rankings

## Run it

```powershell
python server.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Chat examples

- `How many days should I train each week?`
- `I am skinny and want to build muscles`
- `What should I eat for muscle gain?`
- `How much cardio should I do to lose fat?`

## Ask for a plan

You can either fill the quick fields in the UI or type a sentence like:

- `I am very skinny and I want build mussles`
- `female, overweight, want fat burn`

## Optional Kaggle sync

If you want to bring in more public Kaggle datasets, install the Kaggle CLI and configure your API token first, then run:

```powershell
python sync_kaggle.py
```

You can also pass specific dataset slugs:

```powershell
python sync_kaggle.py --dataset exercisedb/fitness-exercises-dataset --dataset niharika41298/gym-exercise-data --dataset gokulprasantht/nutrition-dataset
```

After the download finishes, restart the server or call:

```bash
curl -X POST http://127.0.0.1:8000/retrain
```

You can also send feedback from the UI or with:

```bash
curl -X POST http://127.0.0.1:8000/feedback -H "Content-Type: application/json" -d "{\"rating\":\"up\",\"name\":\"barbell incline bench press\",\"reply\":\"Great\",\"profile\":{\"goal\":\"muscle_gain\"}}"
```

## Dataset-backed NLP

To refresh the Hugging Face fitness knowledge base and rebuild the local retrieval corpus, run:

```powershell
python sync_fitness_datasets.py
```

This downloads and normalizes:

- `hammamwahab/fitness-qa`
- `its-myrto/fitness-question-answers`
- `navaneeth005/diet_advice`
- `kumbh/neurolab-health-nutrition`
- `harshmakwana/fitness-intent`
- `victor203/fitness-preferences`
- `ulysses531/fitness-conversation-dataset`

The assistant then uses those examples to ground chat replies, intent detection, and diet/workout guidance.

## Advanced AI Training

You can now train an advanced, open-source transformer model on your local datasets. This provides high-accuracy semantic search and intent recognition.

1. **Install Dependencies**:
   ```powershell
   pip install torch sentence-transformers pandas
   ```

2. **Run Advanced Training**:
   ```powershell
   python train_advanced_model.py
   ```

This will create a `fitpax_trained_model` directory. The `FitPaxAssistant` in `assistant_core.py` will automatically detect and load this model to provide superior, dataset-grounded responses.

## Good public Kaggle datasets to try

- `exercisedb/fitness-exercises-dataset`
- `niharika41298/gym-exercise-data`
- `gokulprasantht/nutrition-dataset`
- `smayanj/fitness-tracker-dataset`

## Notes

- Kaggle downloads require a Kaggle account and API token.
- The assistant does not magically become perfect, but more data and better prompts will make it much smarter and more useful.
