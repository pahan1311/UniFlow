# UniFlow — University Inquiry Routing with Selective Prediction

An AI system that reads university inquiries, predicts the responsible
department and the urgency, and routes each one to that department's
queue with a response deadline attached — or **abstains** and holds it
for a human when it is not confident enough to route safely.

The abstention is the point. A closed-set classifier forced to answer
every time will confidently misroute the inquiries it understands least,
and a misrouted message is read, forwarded and delayed past its target.
So the system is built around a measured trade-off between how much it
automates and how often it is wrong when it does.

---

## Architecture

The deployed system is a React single-page app talking to a FastAPI
backend over a JSON API — no server-rendered pages, no framework running
on both sides:

```
┌─────────────────────┐   HTTP/JSON   ┌──────────────────────────────────┐
│  frontend/ (React)   │ ────────────► │  FastAPI (uniflow/) — one process │
│  no DB, no models,   │ ◄──────────── │  loads the models once, owns the │
│  just fetch() calls  │               │  database, runs the SLA sweep,   │
└─────────────────────┘               │  email intake and outbox jobs on │
                                       │  a schedule                      │
                                       └──────────────┬────────────────────┘
                                                      │
                                              PostgreSQL
```

In development that's two processes (`uvicorn` + the Vite dev server —
see **Running it** below); in production the frontend's build output is
served as static files by the same FastAPI process, so it collapses back
to one. Either way: the model is loaded exactly once, not once per
process, SLA escalation and email intake run on a schedule instead of
only when someone happens to have the dashboard open, and every routing
decision is a database row that records the exact model version and
policy that produced it rather than a value silently overwritten by the
next run of the evaluation pipeline.

The offline research pipeline (`evaluation/`) is unchanged in spirit: it
still runs standalone against the model weights and writes its results to
`models/*.json` and `evaluation/results/`. What changed is how those
results reach the live system — see
[The pipeline configures the live app](#the-pipeline-configures-the-live-app)
below.

There is no training code in this project: the transformer weights in
`models/` are trained externally (see
[Do I need to retrain my models?](#do-i-need-to-retrain-my-models)
below), and there is no automated feedback-retraining pipeline. Officers
correcting predictions in the **Review** tab still writes to the
`review_events` table — the human-in-the-loop signal is captured — it
simply is not consumed by anything yet.

---

## What is in here

```
uniflow/                    the backend
├── core/                   settings, security (Argon2id + legacy PBKDF2), enums, errors
├── db/                     SQLAlchemy models and repositories (12 tables)
├── domain/                 every business rule: triage, workflow, review, sla,
│                           dedupe, analytics, auth, explanations, policy
├── ml/                     model loading, calibrated inference, energy OOD,
│                           embeddings, Integrated Gradients, multi-task model
├── jobs/                   SLA sweep, email intake, outbox drain, retention —
│                           scheduled inside the API process, no broker
├── api/                    FastAPI routers, request/response schemas
└── scripts/seed.py         creates the tables and seeds departments, SLA
                            targets, the routing policy, demo users

frontend/                   the dashboard (React + Vite + TypeScript)
├── src/api/                a typed client generated from the live OpenAPI schema
├── src/auth/                token storage, refresh-on-401, route guards
├── src/components/ticket/  the shared ticket card / workflow / review / explanation UI
├── src/pages/               Overview, New inquiry, Queues, Triage, Analytics, Login
└── src/styles/theme.css    the one file the whole visual theme lives in

evaluation/                 the research pipeline (standalone, file-based)
├── dataset.py              corpus loading, splits, leakage guard
├── baselines.py            majority / TF-IDF+LR / TF-IDF+SVM
├── evaluate.py             per-class metrics, confusion matrices, McNemar
├── calibrate.py            temperature scaling, ECE, reliability diagrams
├── selective.py            risk–coverage analysis, threshold selection
├── ood.py                  energy-based out-of-scope detection
├── benchmark.py            parameters, disk size, latency
├── stats_tests.py          McNemar exact test, bootstrap CIs
├── logits_cache.py         disk-cached inference (also caps memory use)
├── policy_files.py         reads/writes models/thresholds.json + calibration.json
├── run_all.py              runs every stage, writes results/REPORT.md
└── make_sample_data.py     synthetic data for smoke-testing only

tests/test_uniflow.py       43 unit tests, no model weights required
```

---

## Do I need to retrain my models?

**No.** There is no training code in this project — `models/` holds
weights trained externally (e.g. a Colab notebook), and everything under
`evaluation/` only runs inference over them:

| Component | Retraining needed? | Why |
|---|---|---|
| Evaluation, baselines, confusion matrices, McNemar | **No** | Runs inference over your existing weights |
| Confidence calibration (temperature scaling) | **No** | Post-hoc. Fits one scalar on validation logits; provably cannot change any prediction |
| Selective routing / Triage threshold | **No** | Derived from calibrated confidences you already have |
| Out-of-scope detection (energy score) | **No** | Computed from the same logits; no extra head, no extra training |
| Explainability (Integrated Gradients) | **No** | Gradients through the model you already trained |
| SLA, escalation, auth, email intake, duplicates | **No** | Application logic |

So: run the evaluation pipeline against your current models. It costs
one afternoon and produces most of your results chapter.

### The one thing you must sort out first

The pipeline needs to know **which rows your models were trained on**.
If you evaluate on data the models saw during fine-tuning, every score
is inflated and the entire results chapter is invalid — this is the
single most common way a final-year ML project loses marks.

Export your corpus from Colab as `data/inquiries.csv` with columns:

```
text, category, urgency, split
```

where `split` is `train`, `val` or `test` **exactly as used in the
notebook that produced `models/distilbert_category` and
`models/distilbert_urgency`**.

If you no longer have that split, you have two honest options:

1. **Preferred — retrain on a recorded split.** Run
   `python evaluation/dataset.py data/inquiries.csv`, which writes
   `evaluation/results/splits.csv` with a fixed seed. Retrain both
   models on that file's `train` rows only, then evaluate. Now every
   number is genuinely held out.
2. **Fallback — report the limitation.** Run the pipeline without a
   `split` column. It will print a loud leakage warning and set a flag
   that propagates into the generated report. State plainly in your
   methodology that the split could not be reconstructed and that the
   scores are therefore an upper bound.

Option 1 costs a few GPU-hours on Colab and removes the biggest
criticism available to your examiner. It is worth it.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Place your trained models so the folders contain the weights directly:

```
models/distilbert_category/     model.safetensors, config.json, tokenizer.json, label_classes.json
models/distilbert_urgency/      model.safetensors, config.json, tokenizer.json, label_mapping.json
```

Both `label_classes.json` (a list) and `label_mapping.json` (an
id2label dict) exports are supported.

The application is PostgreSQL-only — create a database and a role for
it, then set `UNIFLOW_DATABASE_URL` in `.env` (copy `.env.example` first):

```
UNIFLOW_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/uniflow
```

Then create the tables and seed them:

```bash
python -m uniflow.scripts.seed
```

This creates every table (no migration tool — see
[Known limitations](#known-limitations)), one department per label the
category model can emit plus the shared Triage queue, the SLA targets, a
routing policy (imported from `models/thresholds.json` and
`calibration.json` if they exist, otherwise the same uncalibrated
defaults the original config used), and the demo accounts below.

Finally, install the frontend's dependencies:

```bash
cd frontend
npm install
```

---

## Running it

Two processes in development.

**Terminal 1 — the backend:**

```bash
uvicorn uniflow.api.main:app --reload
```

This loads the models once, starts the SLA-escalation and outbox jobs
(and the email-intake job, if `.env` has IMAP credentials — see
[Email intake](#email-intake)), and serves the API at
`http://127.0.0.1:8000`. Interactive docs are at `/docs`; `/readyz`
reports whether the database is reachable and the models are loaded.

**Terminal 2 — the dashboard:**

```bash
cd frontend
npm run dev
```

Opens at `http://localhost:5173`, proxying every API call to the backend
(see `frontend/vite.config.ts`) so both talk to the same origin from the
browser's point of view. Sign in with a demo account. Officers see only
their own department plus the shared Triage queue and cannot submit new
inquiries (that page is admin-only); `admin` sees everything, and
submitting an inquiry there redirects straight to wherever it landed —
the department's queue, or Triage if the model abstained.

For a single-process, production-style run instead: `npm run build` in
`frontend/`, then just `uvicorn uniflow.api.main:app` — the API serves
the built frontend from its own origin at `http://127.0.0.1:8000`, no
second process.

| Username | Password | Sees |
|---|---|---|
| `admin` | `admin123` | every department, plus New inquiry |
| `exams` | `exams123` | Examinations + Triage |
| `finance` | `finance123` | Fees and Payments + Triage |
| `it` | `it123` | Learning Platform / IT Access + Triage |
| `registry` | `registry123` | Academic Documents + Triage |
| `admissions` | `admissions123` | Admissions + Triage |
| `timetable` | `timetable123` | Timetable and Scheduling + Triage |
| `general` | `general123` | General Administration + Triage |

These are demo credentials committed on purpose so the project is
runnable by a marker; every seeded account is flagged to require a
password change, and seeding them at all is skipped automatically when
`UNIFLOW_ENV=production`. Say so in your report, and note that a real
deployment would use the university's identity provider.

---

## Running the evaluation

```bash
# 1. Verify the pipeline works, using throwaway synthetic data
python evaluation/make_sample_data.py
python evaluation/run_all.py data/sample_inquiries.csv

# 2. Then run it for real
python evaluation/run_all.py data/inquiries.csv
```

Everything lands in `evaluation/results/`, including `REPORT.md` with
every table pre-formatted, plus seven figures:

- `confusion_category.png`, `confusion_urgency.png`
- `reliability_category.png`, `reliability_urgency.png` — calibration before/after
- `risk_coverage_category.png`, `risk_coverage_urgency.png` — the abstention trade-off
- `ood_energy.png` — in-distribution vs out-of-scope energy separation

Individual stages can be run alone; they share a logit cache, so only
the first one actually runs the models:

```bash
python evaluation/baselines.py data/inquiries.csv
python evaluation/evaluate.py  data/inquiries.csv
python evaluation/calibrate.py data/inquiries.csv     # writes models/calibration.json
python evaluation/selective.py data/inquiries.csv     # writes models/thresholds.json
python evaluation/ood.py       data/inquiries.csv
python evaluation/benchmark.py
```

**Delete `evaluation/results/logits_*.npy` whenever you drop in new
weights.** The cache is keyed by split size and cannot detect that the
model itself changed.

### The pipeline configures the live app

`calibrate.py` and `selective.py` still write `models/calibration.json`
and `models/thresholds.json` exactly as before — the pipeline stays
file-based and runnable standalone, with no database required. What
changed is the last step to make new numbers live:

```bash
python -m uniflow.scripts.seed      # imports the files as a new routing policy
```

The new policy is *not* activated automatically if one is already
active — that switch is a deliberate action, not a silent overwrite,
because every prediction already made under the old policy still
references it and should stay explainable. Activate the new one via the
interactive docs at `http://127.0.0.1:8000/docs`:
`GET /api/v1/policies` to find its id, then
`POST /api/v1/policies/{id}/activate` (as `admin`). From that point on,
the confidence the dashboard displays and the threshold at which the
system abstains are the values you justified experimentally, and every
new prediction records exactly which policy id produced it.

---

## Email intake

```bash
cp .env.example .env       # then fill in the credentials
```

Gmail needs an App Password and IMAP enabled. Keep `UNIFLOW_SEND_ACK=0`
while testing so you do not email real people.

There is no separate poller process to run. Restart the API
(`uvicorn uniflow.api.main:app --reload`) with those variables set, and
its scheduler picks up an `email_intake` job automatically — polling
IMAP, classifying and routing each unseen message through the exact same
service the web form uses, and queuing an acknowledgement through the
outbox once `UNIFLOW_SEND_ACK=1`.

---

## The human-in-the-loop review log

Every Accept or Correct an officer makes in a ticket's **Review** tab
appends a row to `review_events` — recording which prediction was
judged, the reviewer, and the correction if there was one. There is no
feedback-retraining pipeline in this project to consume that log
automatically; the schema and `uniflow.domain.review.unconsumed()` /
`mark_consumed()` are what one would build on if that becomes worth
doing later.

---

## Tests

```bash
python -m pytest tests/ -v
```

43 tests covering SLA arithmetic, access control, optimistic locking,
the review log, calibration and selective-prediction mathematics,
McNemar, dataset splitting and email parsing. Every database test runs
against a throwaway SQLite file created fresh per test — nothing touches
`data/uniflow.db`. They deliberately avoid loading model weights, so
they run in under a minute.

---

## Known limitations — state these in your report

- **No migration tool.** Schema changes are applied with
  `Base.metadata.create_all()`, which only creates tables that do not
  exist yet — it does nothing at all to a table that is already there,
  not even add a column, let alone rename or retype one. Changing a
  model's columns means dropping the database and re-running the seed
  script. This is a deliberate trade for simplicity: nothing here is
  production data that a migration would need to preserve, since
  `uniflow.scripts.seed` fully repopulates a fresh database from the
  model's own label files. A system holding real accumulated data would
  need a migration tool.
- **PostgreSQL only — no SQLite fallback in the application.** Every
  model uses portable column types and the job lock always uses
  `pg_try_advisory_lock`, so there is no dual-database branching to
  maintain. The one exception is the test suite, which deliberately
  builds its own throwaway SQLite engine per test (see
  `tests/test_uniflow.py`'s `db_session` fixture) so `pytest` needs no
  database server running at all.
- **The OOD probe set is synthetic** unless you supply
  `data/ood_samples.csv`. It demonstrates the mechanism; real
  out-of-scope mail collected from the inbox would be much stronger
  evidence.
- **Duplicate detection scans stored embeddings from the lookback
  window in memory.** The transformer runs once per submission rather
  than once per candidate (the original design's real cost), but the
  similarity scan itself is still linear. At real scale this needs a
  vector index — PostgreSQL with pgvector's HNSW index is a drop-in
  target since the switch to PostgreSQL is already one setting away.
- **English only.** For a Sri Lankan university, Sinhala, Tamil and
  code-mixed inquiries are a genuine coverage gap — and the most
  promising direction for further work.
- **Demo credentials are committed** so the project is runnable by a
  marker. Every seeded account requires a password change on first use,
  and seeding is skipped in a production environment — but this is still
  not a deployment posture.
- **The scheduler runs inside the API process**, guarded by a database
  lock rather than a real job queue. That is fine for the one process
  this project runs; running several API instances behind a load
  balancer would need a proper broker instead.

---

## Memory note

The two models need over a gigabyte of resident memory together, but
only in the API process — the React frontend is pure JavaScript, so it
costs nothing to leave open beside it. On a machine with little free
RAM, PyTorch allocation failures still surface on Windows as an *access
violation* rather than a clean error.
The evaluation pipeline mitigates this by loading one task's model at a
time and caching logits to disk; if you still hit it, close other
applications and re-run — completed stages are not recomputed.
