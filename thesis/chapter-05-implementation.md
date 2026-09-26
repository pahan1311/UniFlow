# Chapter 05 — Implementation / Designing

## 5.1 Implementation Approach

This section describes the steps followed in implementing the system specified in Chapter 04: the core
routing and explainability algorithms, the request-level workflow they sit inside, the technology choices
behind both, and — in Section 5.2 — the specific implementation attempts that carry the research
contribution, each shown with the real code that runs in production rather than a simplified illustration of it.

### 5.1.1 Algorithms

**Algorithm 1 — Calibrated, Selective Routing Decision** (`uniflow/ml/inference.py:classify_inquiry`)

```
ALGORITHM ClassifyAndRoute(raw_text, policy)
    cat_logits  <- CategoryEncoder(raw_text)
    urg_logits  <- UrgencyEncoder(raw_text)

    // Temperature scaling (Guo et al., 2017): rescale, do not retrain
    cat_probs <- softmax(cat_logits / policy.category_temperature)
    urg_probs <- softmax(urg_logits / policy.urgency_temperature)

    category   <- argmax(cat_probs);  confidence <- max(cat_probs)
    urgency    <- argmax(urg_probs)

    // Energy-based OOD score (Liu et al., 2020)
    energy <- -T * logsumexp(cat_logits / T)

    IF policy.energy_threshold is set AND energy > policy.energy_threshold THEN
        decision <- OUT_OF_SCOPE ; queue <- Triage
    ELSE IF confidence < policy.category_tau THEN
        decision <- TRIAGE ; queue <- Triage
    ELSE
        decision <- AUTO ; queue <- category
    END IF

    RETURN (category, urgency, confidence, energy, decision, queue)
END ALGORITHM
```

This is a direct transcription of the real function; the only change is dropping variable names specific to
Python (`cat_id`, `urg_id`) in favour of the mathematical quantities they hold. Two properties of the real
implementation are easy to lose in a simplified version and are worth stating explicitly: the OOD check is
evaluated *before* the confidence check, so an inquiry that is both unfamiliar and happens to score a high
raw confidence is still sent to Triage rather than auto-routed; and `thresholds` and `calibration` are function
arguments, not constants read from disk inside the function, so every prediction row records exactly which
`RoutingPolicy` version produced its decision (Chapter 04, Section 4.4.2) — a routing decision from a stale
threshold in memory cannot silently appear.

**Algorithm 2 — Integrated Gradients Attribution** (`uniflow/ml/explain.py:integrated_gradients`)

```
ALGORITHM IntegratedGradients(text, model, task, steps = 32)
    ids        <- Tokenize(text)
    e          <- Embed(ids)                     // actual input embedding
    e'         <- Embed(Baseline(ids))            // every ordinary token -> [PAD]
    target     <- argmax(Forward(e))               // explain the predicted class

    accumulated <- 0
    FOR k = 1 TO steps
        alpha  <- (k - 0.5) / steps                // Riemann midpoint rule
        point  <- e' + alpha * (e - e')
        grad   <- d Forward(point)[target] / d point
        accumulated <- accumulated + grad
    END FOR

    attribution <- SumOverEmbeddingDim((e - e') * accumulated / steps)
    RETURN MergeWordpieces(Tokens(ids), attribution)   // per-token signed score
END ALGORITHM
```

Two implementation details matter enough to keep in the pseudocode rather than abstract away. First, the
baseline replaces every ordinary token with `[PAD]` but keeps special tokens (`[CLS]`, `[SEP]`) in their
original positions, so the interpolated sequence stays structurally valid at every step rather than becoming a
sequence of pure padding the model was never trained to score. Second, attribution is computed per
sub-word token and then summed back into whole words (`MergeWordpieces`) before it ever reaches the
reviewer, because a department officer has no reason to know that a transformer tokenizer split "certificate"
into "cert" and "##ificate."

**Algorithm 3 — Optimistic Concurrency Update** (`uniflow/domain/workflow.py:_bump`)

```
ALGORITHM OptimisticUpdate(inquiry_id, expected_version, changes)
    rows <- EXECUTE
        UPDATE inquiries
        SET version = expected_version + 1, changes...
        WHERE id = inquiry_id AND version = expected_version

    IF rows = 0 THEN
        RAISE Conflict409(expected_version, ActualVersionOf(inquiry_id))
    END IF
    RETURN Success
END ALGORITHM
```

The concurrency control is the `WHERE ... AND version = expected_version` clause and nothing else — there
is no separate read-then-compare step that could itself race. If two officers submit conflicting updates to
the same ticket, the database guarantees exactly one `UPDATE` matches a row and the other returns
`rowcount = 0`, which is what `_bump` turns into the HTTP 409 response described in Chapter 04 (NFR-5).

### 5.1.2 Workflow as a Block Diagram Series

The single end-to-end pipeline shown in Chapter 01 (Figure 1) is broken out here into the four sequential
blocks it is actually implemented as, since each block is a separately testable unit with its own module
boundary.

```
Block 1: Intake                    Block 2: Classify & Calibrate
+----------------------+            +---------------------------+
| triage.submit()        |          | inference.classify_inquiry |
| - validate non-empty   | -------> | - get_logits (category,    |
| - dedupe check          |          |   urgency)                 |
| - load active policy    |          | - calibrated_probs (T)     |
+----------------------+            | - energy_score             |
                                      | -> decision                |
                                      +---------------------------+
                                                    |
                                                    v
Block 4: Persist & Notify           Block 3: Route
+----------------------+            +---------------------------+
| - INSERT Inquiry,       | <------- | _department_for(label,    |
|   Prediction,           |          |   decision)                |
|   InquiryEmbedding,     |          | - AUTO -> predicted dept    |
|   InquiryEvent          |          | - else  -> Triage dept      |
| - queue OutboundMessage |          +---------------------------+
|   (if send_ack)         |
+----------------------+
```
*Figure 14. The intake pipeline as four blocks, matching the module boundaries in `uniflow/domain/triage.py`
and `uniflow/ml/inference.py` — this is the same pipeline as Chapter 01's Figure 1, redrawn at the
granularity the code is actually organized at.*

### 5.1.3 Technology Selection

#### 5.1.3.1 Programming Language

| Language | Used For | Justification |
|---|---|---|
| Python 3.13 | Backend API, domain logic, ML inference, evaluation pipeline | One language across the classifier, the calibration/OOD/explainability layer, and the service that serves them avoids a serialization boundary between "the model" and "the system using it" — `classify_inquiry` runs in the same process as the API request that calls it. |
| TypeScript ~6 | Frontend dashboard | Static typing catches a mismatch between the frontend and the API contract at build time; paired with a client generated from the API's own OpenAPI schema (Section 5.1.3.3), a renamed or retyped backend field becomes a compiler error in the frontend rather than a runtime bug discovered by a department officer. |

#### 5.1.3.2 Libraries Used

| Library | Role | Justification |
|---|---|---|
| PyTorch, Transformers | Model loading and inference | Required for the fine-tuned DistilBERT/BERT-family classifiers and for computing gradients directly, which Integrated Gradients needs. |
| scikit-learn, SciPy | Baseline classifiers, McNemar's test, bootstrap CIs | Provides the interpretable baseline models the transformer is compared against and the significance-testing primitives Chapter 06 relies on. |
| Matplotlib | Evaluation figures | Renders confusion matrices, reliability diagrams, and risk-coverage curves as static images suitable for the thesis and the evaluation report. |
| Pandas, NumPy | Dataset loading and manipulation | Handles the CSV-based corpus (Chapter 03) and vectorised metric computation. |
| FastAPI, Uvicorn | HTTP API | See Section 5.1.3.3. |
| SQLAlchemy 2.0, psycopg | ORM and PostgreSQL driver | A typed, declarative ORM keeps the 13-table schema (Chapter 04, Figure 7) as Python classes with real relationships rather than hand-written SQL scattered across the codebase; `psycopg` is the maintained PostgreSQL driver for the `postgresql+psycopg://` dialect. |
| Pydantic, pydantic-settings | Request/response validation, typed configuration | Every API request and response is a validated schema, and every environment-variable-driven setting (Chapter 01, Table 1.9.2) is a typed field with a documented default rather than an untyped `os.environ.get`. |
| APScheduler | In-process background jobs | Runs the SLA sweep, outbox drain, email intake, and retention jobs without a separate message broker or worker process — a deliberate minimal-infrastructure choice (Chapter 04, NFR-10). |
| argon2-cffi | Password hashing | Argon2id is the password-hashing competition winner and the current OWASP recommendation; the legacy PBKDF2 path exists only to verify pre-existing hashes during migration. |
| PyJWT | Access token issuance/verification | A minimal, widely-audited JWT implementation for the short-lived access token described in Chapter 04 (NFR-3). |
| TanStack Query | Frontend server-state caching | Removes hand-written loading/error/cache-invalidation state from every page component; a mutation (e.g. accepting a review) can declaratively invalidate the queue query that should now show the updated ticket. |
| openapi-typescript / openapi-fetch | Generated, typed API client | Generates TypeScript types directly from the backend's own OpenAPI schema (`npm run gen:api`), so the frontend's understanding of the API can never silently drift from what the backend actually returns. |
| React Router | Client-side routing | Implements the route table and the `RequireAuth`/`RequireAdmin` guards described in Chapter 04 (Section 4.4.5's frontend counterpart). |

#### 5.1.3.3 Backend and Frontend Frameworks

| Framework | Justification |
|---|---|
| FastAPI (backend) | Native async support, automatic OpenAPI schema generation (which the frontend's typed client depends on directly), and Pydantic-based request validation with no separate serialization layer to keep in sync. |
| React + Vite (frontend) | React's component model suits a dashboard built from many independently reusable pieces (ticket cards, queue tables, chart primitives); Vite's dev server proxy gives the development environment the same same-origin behaviour as the production build (Chapter 01), avoiding a class of CORS-only-in-dev bugs. Chosen deliberately over a meta-framework (e.g. Next.js): the system already has one backend serving one API, so file-based routing and server-side rendering would add framework surface area without solving a problem this deployment has. |

## 5.2 Selected Implementation Attempts

This section covers four implementation attempts that carry the research contribution — the parts of the
system where a design decision, not just routine CRUD, was required. Each is shown with the real code, the
reasoning documented in the codebase itself, and the evidence that confirms it behaves as intended.

### 5.2.1 The Auto-Route/Abstain Decision

This is the algorithmic core of the thesis: the point where a calibrated confidence score and an
out-of-distribution score jointly decide whether a human ever needs to look at an inquiry before it reaches a
department queue.

```python
if energy_threshold is not None and cat_energy > energy_threshold:
    decision = RoutingDecision.OUT_OF_SCOPE
    queue = TRIAGE_NAME
elif category_conf < tau:
    decision = RoutingDecision.TRIAGE
    queue = TRIAGE_NAME
else:
    decision = RoutingDecision.AUTO
    queue = category_label
```
*(`uniflow/ml/inference.py:classify_inquiry`, condensed; the full function also builds the human-readable
`routing_reason` string shown to reviewers.)*

**Executional evidence.** The test suite exercises the calibration and thresholding mathematics directly
against synthetic logits with known properties, independent of the constructed corpus, so the algorithm's
correctness is not entangled with dataset quality: `test_temperature_scaling_preserves_predictions` and
`test_temperature_above_one_reduces_confidence` confirm calibration changes confidence without changing
the predicted class; `test_choose_tau_respects_the_target_risk` and
`test_choose_tau_reports_failure_when_the_target_is_unreachable` confirm threshold selection behaves
correctly at both a reachable and an unreachable target risk; `test_energy_is_lower_for_confident_logits` and
`test_energy_uses_magnitude_not_just_shape` confirm the energy score actually distinguishes a confidently
peaked distribution from a uniformly low-magnitude one, which is the property that lets it catch an
out-of-scope message a plain softmax confidence would not. Real classification performance, calibration
error, and risk-coverage numbers measured against the constructed corpus (rather than these unit-level
mathematical checks) are reported in Chapter 06.

### 5.2.2 Per-Token Explanation with Integrated Gradients

The Explanation tab in the ticket detail view (Chapter 01, rich picture) is backed by the cached attribution
computed by `explain.py`. Its UI presents the merged, whole-word attribution scores as a heat-mapped
sentence rather than a raw score table:

```
+---------------------------------------------------------------+
|  Ticket INQ-2026-004881          [ Workflow | Review | Explanation ] |
+---------------------------------------------------------------+
|  Predicted: Examinations (High)     Confidence: 91%             |
|                                                                   |
|  "I  cannot  access  the  [exam portal]++  and  my  [paper]+     |
|   is  [tomorrow]+++ , this  is  urgent"                          |
|                                                                   |
|   (+++ strongest positive attribution   -- negative attribution) |
|                                                                   |
|  Keyword-heuristic agreement: 4/6 top words overlap (Jaccard 0.67)|
+---------------------------------------------------------------+
```
*Figure 15. Explanation tab wireframe. Attribution strength is rendered as highlight intensity in the actual
React component; the plain-text `+`/`-` markers above stand in for that highlighting here.*

`keyword_agreement()` is worth calling out specifically as a design decision rather than a cosmetic addition:
it compares the model's own top-attributed words against the pre-existing hand-authored keyword heuristic
(`uniflow/ml/keywords.py`, originally the system's only explanation mechanism) and reports the Jaccard
overlap. This turns what could have been an unfalsifiable "the model looked at reasonable words" claim into
a reportable number — Chapter 06 evaluates this agreement rate rather than asserting it.

### 5.2.3 Optimistic Concurrency on Ticket Updates

The bug this fixes is documented directly in `workflow.py`'s own module docstring, which is worth quoting
because it states the failure mode precisely: the previous implementation "read `queue_status`,
`first_response_at` and `resolved_at`, computed the new values in Python, then wrote them back — with no
transaction and no guard. Two officers acting at the same time silently overwrote each other." The fix
replaces that read-compute-write sequence with a single conditional `UPDATE`:

```python
result = session.execute(
    sa.update(Inquiry)
    .where(Inquiry.id == inquiry.id, Inquiry.version == expected_version)
    .values(version=expected_version + 1, updated_at=datetime.now(timezone.utc))
)
if result.rowcount == 0:
    raise ConflictError(...)
```

**Executional evidence.** `test_stale_write_is_rejected_by_optimistic_locking` constructs exactly the race
condition the docstring describes — two updates against the same inquiry, the second using a version number
already superseded by the first — and confirms the second raises `ConflictError` rather than silently
succeeding. `test_route_from_triage_logs_a_correction` and `test_review_events_recorded_and_consumable`
confirm the same conditional-update path correctly distinguishes an `ACCEPT` from a `CORRECT` action and
records it against the right prediction, so the concurrency fix did not regress the review-logging behaviour it
shares a code path with.

### 5.2.4 Automated-Sender Filtering in Email Intake

Unlike the previous three attempts, this one was not a designed-in feature; it was added in response to a real
incident during development, and is reported here as such rather than presented as if it had been planned
from the start. During a live verification round, the IMAP poller ingested seven automated account-security
notification emails from an email provider as though they were genuine student inquiries, because nothing in
the intake path distinguished a message's sender from a student's. The fix adds a sender check before a
message ever reaches classification:

```python
def is_automated_sender(address: str) -> bool:
    local_part = address.split("@", 1)[0].lower()
    return any(pattern in local_part for pattern in _AUTOMATED_SENDER_PATTERNS)
```
*(`uniflow/jobs/email_intake.py`; `_AUTOMATED_SENDER_PATTERNS` covers common no-reply/notification
conventions.)* A message from a matched sender is marked `status: "skipped_automated_sender"` and never
reaches `triage.submit()`, so it produces no `Inquiry`, no `Prediction`, and no classification cost.

**Executional evidence.** `test_automated_senders_are_not_treated_as_students` asserts that a message from
a matched address is skipped before classification. The incident itself is the strongest evidence of why the
check exists: the seven previously-ingested notification emails were manually identified and deleted from the
database, and no further automated-sender inquiries have been ingested since the filter was deployed.

## 5.3 Chapter Summary

This chapter described how the system specified in Chapter 04 was actually built: the calibrated,
selectively-abstaining routing algorithm and the Integrated Gradients explanation algorithm, expressed as
pseudocode faithful to the real implementation; the intake pipeline redrawn as the four blocks it is actually
organized into; and a justified technology stack spanning one language on the backend and a statically-typed,
schema-generated frontend. Section 5.2 went a level deeper into four implementation attempts that carry the
thesis's contribution — the auto-route/abstain decision, per-token explanation, optimistic concurrency, and
automated-sender filtering — each shown with its real code, its rationale as documented in the codebase, and
the specific test or incident that provides evidence it works. One of the four, automated-sender filtering, is
reported honestly as a fix discovered through a real incident rather than a feature anticipated in advance,
consistent with this thesis's practice of describing what was actually built rather than smoothing execution
into a plan it did not originally follow. The next chapter presents the testing strategy and the evaluation
results measured against the constructed corpus.
