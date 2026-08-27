# University Inquiry Classification & Urgency Dashboard

A Streamlit dashboard that loads a fine-tuned DistilBERT category model
and BERT urgency model and lets you classify new student inquiries,
review a priority queue, and correct predictions (human-in-the-loop
feedback).

## Project structure

```
university-inquiry-dashboard/
├── app/
│   ├── app.py              # Main Streamlit app — run this
│   ├── model_utils.py      # Loads models, runs predictions
│   ├── preprocessing.py    # Text cleaning (must match training)
│   ├── keywords.py         # Heuristic trigger-word explainability
│   └── db_utils.py         # SQLite storage for the feedback loop
├── models/
│   ├── distilbert_category/   # <- put your trained model files here
│   └── distilbert_urgency/    # <- put your trained model files here
├── data/
│   └── feedback.db         # created automatically on first run
├── requirements.txt
└── README.md
```

## Step 1 — Install Python dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

If you don't need the optional Audio Upload tab, you can skip
`openai-whisper` — just remove that line from `requirements.txt`
before installing, or ignore the install error for it.

## Step 2 — Add your trained models

From your Google Colab training run, you should have two folders:
`distilbert_category` and `distilbert_urgency`, each containing
`model.safetensors`, `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `label_classes.json`, etc.

The urgency BERT export may use `label_mapping.json` instead of
`label_classes.json`; the application supports both formats.

Copy the **contents** of each into the matching folder here:
- `models/distilbert_category/`
- `models/distilbert_urgency/`

Delete the `PUT_MODEL_FILES_HERE.txt` placeholder files once done.

## Step 3 — Run the dashboard

```bash
cd app
streamlit run app.py
```

This opens the dashboard in your browser at `http://localhost:8501`.

## Notes

- **First load will be slow** (~10–30 seconds) while both models load
  from disk. This only happens once per server run, thanks to
  `@st.cache_resource` in `app.py` — after that, predictions are fast.
- **Urgency is inferred directly from the inquiry text** by the BERT
  urgency model, so students do not need to enter a separate deadline.
- **Feedback data** (accepted/corrected predictions) is stored in
  `data/feedback.db` (SQLite). You can open this with any SQLite
  browser, or use the "Download queue as CSV" button in the
  Priority Queue tab.
- **Audio Upload tab** requires `openai-whisper` and `ffmpeg` installed
  on your system. If you don't need ASR right now, you can safely
  ignore errors on that tab — the rest of the dashboard works
  independently.
- For your viva demo: run this locally on your laptop and also record
  a backup video, per your execution plan's risk mitigation (Phase 5,
  Step 5.2).
