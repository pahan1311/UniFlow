# Chapter 06 — Testing and Evaluation

## 6.1 Chapter Overview

This chapter presents the system's testing strategy and the evaluation results measured against the
constructed corpus (Chapter 03). Two separate testing efforts run through this project, deliberately kept
independent of each other. The first is a pytest suite of 43 tests exercising the domain logic, access control,
concurrency handling, and the mathematics behind calibration and selective prediction, run against a
throwaway SQLite database with no transformer model loaded, so it runs in seconds and is safe to run on
every change. The second is the `evaluation/` pipeline, which loads the actual fine-tuned transformer and
scores it against the real, held-out test split, producing the classification, calibration, selective-prediction,
out-of-distribution, and inference-cost results this chapter reports in Section 6.3. Section 6.2 presents ten
representative test cases from the first effort in full, with the remaining tests summarized in Appendix D.
Section 6.4 reviews which testing strategies were used, and, where a strategy from the template's checklist
was deliberately not used, why.

## 6.2 Test Plan and Test Cases

Every test case below names the real test function it corresponds to in `tests/test_uniflow.py`, runs against
a throwaway SQLite database created fresh for that test (`db_session` fixture), and was passing at the time of
writing. The full suite of 43 tests is summarized by area in Appendix D.

### 6.2.1 Non-Functional Testing

| ID | Test | Objective | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|---|
| TC-02 | `test_stale_write_is_rejected_by_optimistic_locking` | Confirm two concurrent writers to the same ticket cannot silently overwrite each other (NFR-5, Chapter 04). | Two `update_status` calls against the same inquiry, both supplying `expected_version=1`; the first changes status to `IN_PROGRESS`, the second (still using version 1) tries to change it to `RESOLVED`. | The second call raises `ConflictError` rather than succeeding. | `ConflictError` raised on the second call. | Pass |
| TC-03 | `test_login_locks_the_account_after_repeated_failures` | Confirm brute-force login attempts are locked out after the configured threshold (NFR-3). | `max_failed_logins` (5) consecutive `login()` calls with a wrong password, then one call with the correct password. | Every wrong-password attempt raises `AuthenticationError`; the correct-password attempt afterward also raises `AuthenticationError` with a message containing "locked". | All 5 wrong attempts raised `AuthenticationError`; the correct-password attempt also raised `AuthenticationError` matching "locked". | Pass |
| TC-04 | `test_visible_departments_scoped_to_own_plus_triage` | Confirm an officer's visible-department set is exactly their own department plus Triage, and an administrator has no restriction (NFR-2). | `visible_department_ids()` called for an officer principal in "Examinations" and for an administrator principal, with three departments (Triage, Examinations, Admissions) present. | Officer's scope equals `{Triage, Examinations}`; administrator's scope is `None` (unrestricted). | Officer scope was `{Triage.id, Examinations.id}`; administrator scope was `None`. | Pass |
| TC-05 | `test_can_read_denies_other_departments` | Confirm an officer cannot read a ticket outside their department, but can read their own and any Triage ticket (NFR-2). | `can_read()` called by an Examinations officer against three tickets: one in Examinations, one in Admissions, one in Triage. | `True` for the Examinations and Triage tickets; `False` for the Admissions ticket. | Examinations ticket: `True`. Triage ticket: `True`. Admissions ticket: `False`. | Pass |

### 6.2.2 Functional Testing

| ID | Test | Objective | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|---|
| TC-01 | `test_sla_due_dates_follow_urgency` | Confirm the service-level deadline is computed correctly from urgency (FR-4). | `sla.due_at()` called with `Urgency.HIGH` and `Urgency.LOW`, both created at 2026-01-01 09:00 UTC. | High urgency due 4 hours later (13:00); Low urgency due 72 hours later (2026-01-04 09:00). | High: 2026-01-01 13:00 UTC. Low: 2026-01-04 09:00 UTC. | Pass |
| TC-06 | `test_route_from_triage_logs_a_correction` | Confirm routing an abstained ticket out of Triage updates the ticket and logs exactly one review event (FR-8). | `workflow.route()` called on a Triage ticket, routing it to Examinations with `expected_version=1`. | Ticket's `department_id` becomes Examinations and `routing` becomes `AUTO`; exactly one `ReviewEvent` is recorded for the ticket. | Department and routing updated as expected; exactly one `ReviewEvent` row found. | Pass |
| TC-07 | `test_email_source_ref_uniqueness_is_enforced` | Confirm the same email message cannot be ingested twice (FR-1, database-level enforcement). | Two inquiries created with `source=EMAIL` and the identical `source_ref` (`<msg-1@example.com>`). | The second insert raises `IntegrityError` from the `UNIQUE(source, source_ref)` constraint. | `IntegrityError` raised on the second insert. | Pass |
| TC-08 | `test_automated_senders_are_not_treated_as_students` | Confirm known automated/no-reply senders are filtered before classification (FR-11); the regression test for the real incident described in Chapter 05. | `is_automated_sender()` called with `no-reply@accounts.google.com`, `notifications@github.com`, `MAILER-DAEMON@uni.edu`, `student@uni.edu`, `None`, and `"not-an-address"`. | `True` for the three automated addresses; `False` for the student address, `None`, and the malformed string. | Matched exactly as expected for all six inputs. | Pass |
| TC-09 | `test_temperature_scaling_preserves_predictions` | Confirm temperature scaling never changes which class is predicted (the property that makes it safe to apply after training, Chapter 02). | `calibrated_probs()` applied to the same fixed random logits (seed 0, 64 examples x 7 classes) at temperatures 0.5, 1.5, 3.0, and 10.0. | The `argmax` predicted class at every tested temperature equals the `argmax` at temperature 1.0 (no scaling). | Predicted class identical across all four temperatures for all 64 examples. | Pass |
| TC-10 | `test_choose_tau_respects_the_target_risk` | Confirm the selective-prediction threshold selection reaches a specified target risk at usable coverage. | 2,000 synthetic confidence/correctness pairs (confidence ~ Uniform[0.3, 1.0], correctness ~ Bernoulli(confidence)); `choose_tau()` called with `target_risk=0.15`. | The target is reported reachable, and evaluating the corpus at the chosen τ yields a selective risk at or below 15%. | Target reachable; selective risk at chosen τ was at or below 0.15. | Pass |

## 6.3 Evaluation Results

The evaluation pipeline (Chapter 03, Section 3.5.5) was run against the constructed corpus of 10,782
inquiries (8,879 train / 960 validation / 943 test, Chapter 03), using the real train/validation/test assignment
the models were fine-tuned on rather than a freshly generated split — the pipeline's own leakage check
confirmed this (`leakage_risk: false`), so the figures below are not inflated by the transformer having already
seen a test-set inquiry during training.

**A methodological note worth recording here.** The first run of this pipeline against the real corpus
produced classification and significance numbers that were, on inspection, identical to an earlier pipeline
smoke-test run on synthetic data and covered only 70 of the 943 test inquiries. The cause was a real defect in
`evaluate.py` (`from config import MODELS_DIR`, referencing a module that does not exist in this codebase,
where every other evaluation script correctly imports `MODELS_DIR` from `policy_files`); the stage crashed
before writing new results, and the report-assembly step silently reused stale files from a previous run
instead of failing loudly. The bug was fixed and the stage re-run before any number below was accepted.
This is reported because it is a genuine methodological risk this research encountered and corrected — an
evaluation pipeline can produce a plausible-looking but wrong report exactly as easily as the model it
evaluates can produce a plausible-looking but wrong prediction, and both need to be checked rather than
trusted on inspection alone.

### 6.3.1 Classification Performance

| Task | System | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| Category | majority-class baseline | 0.0764 | 0.0203 | 0.0108 |
| Category | TF-IDF + Logistic Regression | 0.9852 | 0.9865 | 0.9851 |
| Category | TF-IDF + Linear SVM | 0.9926 | 0.9924 | 0.9926 |
| Category | **Dual transformer** | 0.9936 | **0.9906** | 0.9935 |
| Category | *(transformer macro-F1, 95% bootstrap CI)* | | *[0.9824, 0.9969]* | |
| Urgency | majority-class baseline | 0.3319 | 0.1661 | 0.1654 |
| Urgency | TF-IDF + Logistic Regression | 0.9459 | 0.9451 | 0.9451 |
| Urgency | TF-IDF + Linear SVM | 0.9173 | 0.9152 | 0.9153 |
| Urgency | **Dual transformer** | 0.9809 | **0.9808** | 0.9808 |
| Urgency | *(transformer macro-F1, 95% bootstrap CI)* | | *[0.9714, 0.9894]* | |

*[Insert `confusion_category.png` and `confusion_urgency.png` from `evaluation/results/` here.]*

### 6.3.2 Significance (McNemar's Exact Test)

| Task | Baseline | Transformer-only correct (b) | Baseline-only correct (c) | p | Verdict |
|---|---|---|---|---|---|
| Category | majority | 871 | 6 | 1.24e-249 | Significant |
| Category | TF-IDF + LR | 14 | 6 | 0.1153 | Not significant |
| Category | TF-IDF + SVM | 7 | 6 | 1.000 | Not significant |
| Urgency | majority | 627 | 15 | 9.45e-164 | Significant |
| Urgency | TF-IDF + LR | 39 | 6 | 5.42e-07 | Significant |
| Urgency | TF-IDF + SVM | 65 | 5 | 2.22e-14 | Significant |

The transformer's advantage over the majority-class baseline is significant for both tasks, as expected. For
category classification, the transformer is **not** significantly different from either TF-IDF baseline (p =
0.12 and p = 1.0) — the templated, scenario-based construction of the corpus (Chapter 03) gives strong
enough lexical cues per category that a linear bag-of-words model reaches essentially the same performance
as a fine-tuned transformer, and this is reported as an honest finding rather than smoothed into a claimed
transformer advantage the data does not support. For urgency classification, the transformer is significantly
better than both baselines (p = 5.4e-07 and p = 2.2e-14): urgency appears to depend on context in a way a
bag-of-words representation captures less completely than category does, consistent with the Chapter 02
discussion of why urgency, more than category, benefits from contextual representation (e.g. distinguishing
"exam" appearing routinely from "exam" combined with "today" and "cannot access").

### 6.3.3 Confidence Calibration (Temperature Scaling)

| Task | T | ECE before | ECE after | NLL before | NLL after | Mean confidence before | after |
|---|---|---|---|---|---|---|---|
| Category | 0.630 | 0.0119 | 0.0052 | 0.0284 | 0.0284 | 0.989 | 0.998 |
| Urgency | 1.569 | 0.0174 | 0.0140 | 0.0806 | 0.0720 | 0.991 | 0.975 |

Accuracy is unchanged by temperature scaling by construction (Chapter 02); only the reported confidence
moves. The fitted category temperature (0.630, below 1) sharpens an already well-behaved confidence
distribution rather than correcting overconfidence, while the urgency temperature (1.569, above 1) softens a
more overconfident raw distribution — both directions are legitimate outcomes of the same method, and
which direction is needed is itself something that has to be measured per task rather than assumed. Expected
Calibration Error improves for both tasks (category 0.0119 to 0.0052; urgency 0.0174 to 0.0140).

*[Insert `reliability_category.png` and `reliability_urgency.png` from `evaluation/results/` here.]*

### 6.3.4 Selective Routing (Risk-Coverage)

| Task | τ | Coverage | Selective risk | Risk if fully automated | Misroutes avoided | AURC |
|---|---|---|---|---|---|---|
| Category | 0.554 | 100.0% | 0.64% | 0.64% | 0 | 0.0000 |
| Urgency | 0.521 | 99.8% | 1.91% | 1.91% | 0 | 0.0014 |

τ is selected on the validation split against a target selective risk and then applied unchanged to the test
split. Because the classifier is already highly accurate on this corpus (Section 6.3.1), the validated τ ends up
low enough that selective risk and full-automation risk coincide almost exactly at very high coverage — the
abstention mechanism found little it needed to abstain on here, which is a direct consequence of the corpus
being easier to classify than a genuine, noisy mailbox is likely to be (Chapter 07 returns to this as a
limitation). The mechanism's behaviour under a harder, noisier distribution is better evidenced by the
out-of-distribution results below than by this risk-coverage curve alone.

*[Insert `risk_coverage_category.png` and `risk_coverage_urgency.png` from `evaluation/results/` here.]*

### 6.3.5 Out-of-Distribution Detection (Energy Score)

| Metric | Value |
|---|---|
| AUROC | 0.9898 |
| False positive rate at 95% true positive rate | 0.0000 |
| Energy threshold | -5.1496 |
| Out-of-scope messages caught | 100.0% |
| Genuine inquiries diverted to Triage (false positive cost) | 4.5% |
| Probe set | 20 items — **synthetic**, standing in for genuine out-of-scope mail |

The energy score separates the 20-item synthetic out-of-scope probe set from genuine inquiries almost
perfectly (AUROC 0.99, zero false positives at 95% true-positive rate). The probe set's synthetic status is
carried over honestly from the evaluation script's own labelling rather than presented as if it were real
out-of-scope mail; Chapter 07 recommends replacing it with genuine out-of-scope messages (e.g. spam,
unrelated correspondence, or inquiries about a different institution entirely) as future work, since a
synthetic probe set — almost by construction — is easier to separate than a real one would be.

*[Insert `ood_energy.png` from `evaluation/results/` here.]*

### 6.3.6 Inference Cost

| Backend | Parameters (M) | Disk (MB) | Forward passes per inquiry | Median latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|
| Dual | 176.44 | 674.5 | 2 | 364.4 | 466.1 |

Measured on CPU (4 threads), matching the deployment profile in Chapter 01's resource requirements
(no GPU is assumed at inference time — fine-tuning, not serving, is what used the cloud GPU). A median
latency of 364 ms per inquiry is comfortably inside the time budget of a synchronous HTTP request-response
cycle (Chapter 04, NFR-7), including for the email-intake job, which processes messages on a 60-second
poll interval rather than under interactive latency pressure.

## 6.4 Review of Testing Strategies

| Strategy | Used? | Justification |
|---|---|---|
| Unit / integration testing (pytest) | Yes | The primary strategy for domain logic, access control, and ML mathematics — fast, deterministic, and independent of dataset or model quality, so a routing-logic regression is caught without ever loading a transformer. |
| Model evaluation (held-out test set, baselines, significance testing) | Yes | The primary strategy for the ML contribution itself — accuracy alone would not support the calibration or selective-prediction claims made in Chapters 02 and 05, so this project treats classification metrics, calibration metrics, and selective-prediction metrics as three separate, necessary evaluation strategies rather than one. |
| Database-constraint testing | Yes | Used specifically for invariants a unit test cannot verify by calling a Python function alone (e.g. `UNIQUE(source, source_ref)`, TC-07), since the constraint is enforced by PostgreSQL/SQLite itself, not by application code that could be bypassed. |
| Regression testing | Yes | TC-08 is a regression test written directly in response to the real automated-sender incident (Chapter 05); this is the project's practice of turning an incident into a permanent test rather than only a one-time fix. |
| Manual / exploratory testing | Yes, informally | Used throughout development to verify HTTP behaviour directly (issuing real requests and inspecting the resulting database rows), as described in Chapter 04's operationalization process, rather than trusting code review alone. |
| Load / stress testing | No | Not attempted. The system has not been exercised under concurrent multi-user HTTP load, so its behaviour under realistic institutional traffic volume is not evaluated in this thesis and is named as a limitation in Chapter 07. |
| Security penetration testing | No | Not attempted; the auth and access-control tests in this suite verify the intended behaviour of the implemented controls (lockout, department scoping, token rotation), not resistance to a deliberate attacker probing for unintended behaviour. |
| Usability testing with real staff | No | The dashboard has not been evaluated with actual department officers; its design follows the requirements in Chapter 04, but a genuine usability assessment would require institutional access this undergraduate project did not have. |
| Cross-institution / external validity testing | No | The corpus (Chapter 03) was constructed for this specific university-category schema; how the trained classifiers perform on a different institution's inquiry categories or phrasing conventions is untested and out of scope. |

Not every strategy the template lists applies to a project of this kind and scope, and the table above is meant
to show that omission was a decision, not an oversight: load testing and penetration testing in particular
would require infrastructure and authorization beyond an undergraduate research project, and are named
explicitly in Chapter 07 as recommended future work rather than silently absent.

## 6.5 Chapter Summary

This chapter presented ten representative test cases from the 43-test suite, split across non-functional
concerns (concurrency, account lockout, department-scoped access control) and functional concerns (SLA
computation, routing corrections, email deduplication, automated-sender filtering, and the mathematical
correctness of calibration and selective prediction). It then reported the evaluation pipeline's results against
the real, constructed corpus: a dual transformer statistically indistinguishable from a linear baseline on
category classification but significantly better on urgency classification, well-calibrated confidence scores,
a selective-prediction mechanism operating near-perfectly on this corpus's difficulty level, and
near-perfect separation of a synthetic out-of-distribution probe set. It also reported, rather than
concealed, a real defect discovered in the evaluation tooling itself during this process, and the strategies
this project did not attempt alongside the reasoning for each omission. The next and final chapter reflects on
these results against the research objectives, states the project's limitations honestly, and sets out
recommendations for future work, including voice/ASR intake, feedback retraining, and the load, security,
and usability testing this chapter identified as not yet performed.
