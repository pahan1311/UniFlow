# Chapter 07 — Concluding Remarks

## 7.1 Accomplishment of the Research Objectives

Each research objective from Chapter 01 is assessed here by triangulation: literature evidence (Chapter 02),
implementation evidence (Chapters 04-05), and testing evidence (Chapter 06), rather than by a single line of
support. An objective is judged met only when at least two of these three independently point the same way.

**Objective 1 — To identify university inquiry categories, urgency criteria, and suitable secondary datasets.**
Met. Chapter 02 reviewed the domain and the available secondary datasets; Chapter 03 defined seven
categories and three urgency levels and constructed a 10,782-row corpus combining relabelled public
sources with author-constructed inquiries against that schema; Chapter 06's evaluation confirms the schema
is learnable, with the classifier reaching 0.99 category macro-F1 and 0.98 urgency macro-F1 on held-out
data drawn from it.

**Objective 2 — To analyze existing AI, NLP, ticket classification, support-desk prioritization, calibration,
and selective-prediction studies.** Met. Chapter 02's literature review covers all of these areas and — beyond
the interim report's scope — extends into calibration, out-of-distribution detection, and explainability
literature once building the system showed these were necessary, not optional reading. The research gap this
analysis produced (Section 2.6) is the gap this thesis's artifact actually closes, which is the intended purpose
of this objective rather than literature review as an end in itself.

**Objective 3 — To design and develop a system that preprocesses inquiry text, classifies category and
urgency, calibrates confidence, and abstains to a human-reviewed queue when confidence or
out-of-distribution evidence indicates the prediction should not be trusted.** Met. Chapter 04 specified this
system and Chapter 05 showed the real code implementing it — the calibrated, dual-gated routing decision
(Section 5.2.1), per-token explanation (Section 5.2.2), optimistic concurrency (Section 5.2.3), and
automated-sender filtering (Section 5.2.4) are not aspirational descriptions but working code with passing
tests (Chapter 06, Section 6.2).

**Objective 4 — To evaluate the proposed system using classification, calibration, and selective-prediction
metrics, comparing against baseline models.** Met, with an honest qualification. Chapter 06 reports
classification accuracy/macro-F1/significance, Expected Calibration Error, risk-coverage/AURC, and
energy-score AUROC, all against the real held-out corpus and all compared against classical baselines with
McNemar's exact test. The qualification is that the transformer's advantage over the strongest baseline is
statistically significant for urgency but not for category (Section 6.3.2) — evaluation was carried out
completely and honestly, and it does not uniformly favour the more complex model, which this thesis treats
as a finding to report rather than a result to explain away.

## 7.2 Problems Encountered

Six problems shaped this project's execution, each already described where it occurred and drawn together
here as a connected account of what was actually difficult.

**Public datasets did not fit the university schema.** As anticipated in the interim report's specific-problem
statement, the public secondary sources reviewed in Chapter 02 used categories and urgency conventions that
did not map cleanly onto university administrative categories, which is why the final corpus (Chapter 03)
became a deliberate mix of relabelled public data and author-constructed scenario-based inquiries rather than
a public dataset used as-is.

**The first fine-tuned classifier's confidence was not trustworthy.** Evaluating the transformer's raw
confidence during the DSRM Evaluation stage (Chapter 03, Section 3.3) showed it was miscalibrated in the
way Guo et al.'s literature predicts, which sent the project back into Design and Development to add
temperature scaling, the τ threshold, and energy-based out-of-distribution detection before evaluation could
be trusted — an unplanned iteration the interim schedule did not budget for.

**The original prototype architecture could not support the access-control and audit requirements the
system turned out to need.** Once department-scoped access control, an append-only audit trail, and
optimistic concurrency were identified as requirements (Chapter 04), the Streamlit/SQLite prototype was
rebuilt on FastAPI, PostgreSQL, and React (Chapter 03, Section 3.6.1) — a larger engineering undertaking
than originally scheduled, running in parallel with model refinement rather than after it.

**A live automated-sender email incident exposed a missing safeguard.** During development, the IMAP
poller ingested seven automated account-security notification emails as if they were student inquiries
(Chapter 03, Section 3.6.2; Chapter 05, Section 5.2.4), which was not anticipated until it happened and had
to be fixed and turned into a permanent regression test (Chapter 06, TC-08) rather than a one-time cleanup.

**Windows and CPU-only inference imposed real engineering constraints.** Running two transformer models
in the same process on a memory-constrained laptop required deliberately releasing each model before
loading the next (Chapter 05's `logits_cache.py`), because a failed PyTorch allocation on Windows surfaces
as an opaque access violation rather than a catchable `MemoryError` — a constraint documented directly in
the evaluation tooling's own comments and one this thesis's evaluation pipeline was written around rather
than worked past.

**The evaluation pipeline itself contained a defect that produced a plausible but wrong report.** A broken
import in `evaluate.py` caused a stage to fail silently while the report-assembly step reused stale results from
an earlier synthetic-data run (Chapter 06). This was caught only because the resulting numbers were checked
against what the underlying test-set size should have been, not because the report looked wrong on its face
— it did not.

## 7.3 Self-Reflection

### 7.3.1 Ideology About the Research Carried Out

This research is built on a specific position about what an AI system should be trusted to decide on its own:
that a classifier's raw output is not a decision, and that the gap between "the model produced a label" and
"it is safe to act on that label without a human" has to be closed explicitly, with evidence, rather than assumed
away by reporting high accuracy and moving on. That position shaped almost every design decision in this
thesis — calibration, selective abstention, out-of-distribution detection, per-token explanation, and an
append-only audit trail all exist because the alternative (auto-routing every inquiry on raw confidence) was
judged unacceptable for a system making decisions that affect a real student's deadline, even though it would
have been considerably simpler to build and would have looked no worse in a plain accuracy table.

### 7.3.2 Benefits Gained

This project required working competently across a wider stack than a purely ML-focused undergraduate
project typically demands: fine-tuning and calibrating transformer classifiers; implementing selective
prediction and out-of-distribution detection from their source papers rather than from a library call;
designing and building a role-based, auditable web service in FastAPI and PostgreSQL; and building a typed,
schema-generated React frontend consuming it. Beyond the technical skills, the research-methodology
discipline of Design Science Research — treating the system itself as the unit of evaluation, and being
answerable to literature, testing, and honest reporting simultaneously rather than any one of them alone — is
a way of working this thesis's author had not practised at this scale before.

### 7.3.3 Learning Curves

Several concepts required real study before they could be implemented correctly rather than merely invoked:
temperature scaling's guarantee that it preserves the predicted class (verified directly in TC-09 once
understood, not merely assumed); why an energy score, and not a softmax confidence, is needed for
out-of-distribution detection (Section 2.5.1); the mechanics of Integrated Gradients' path-integral formulation
well enough to implement it directly rather than call a library function; and the concurrency-control
reasoning behind optimistic locking, which required understanding a real bug (the "lost update" described in
`workflow.py`'s own docstring) before the fix made sense as anything other than boilerplate. In each case, the
learning curve was steepest exactly where the interim report had least anticipated needing it — which is
itself consistent with this thesis's broader finding that a system's real requirements are discovered by
building it, not fully known in advance.

## 7.4 Business Insight

### 7.4.1 Real-World Application Possibilities

The calibrated, selectively-abstaining triage pattern this thesis implements is not specific to university
inquiries; it generalizes to any service-desk setting where misrouting a request has a real cost and a human
reviewer is available as a backstop — government citizen-service portals, healthcare administrative
scheduling, and commercial customer-support ticketing are all structurally similar problems. Within higher
education specifically, a realistic adoption path would start narrow rather than institution-wide: pilot the
system on one or two high-volume departments (Examinations and Fees and Payments are natural
candidates, given their deadline sensitivity), measure the τ-driven automation coverage and misroute rate
against real traffic for a full semester, and only then expand department coverage — mirroring the
department-scoped access model the system already implements rather than requiring a redesign to support a
staged rollout. The value proposition to an administration is concrete and auditable rather than speculative:
faster first response to deadline-sensitive inquiries, a stored audit trail usable for compliance and quality
review, and a quantified (not assumed) automation rate that leaves the harder cases explicitly with staff
rather than hidden inside an opaque "AI handled it" claim.

## 7.5 Future Recommendations

The following recommendations are not a generic wish list; each responds to a specific limitation this thesis
identified and reported rather than concealed.

1. **Implement voice/ASR intake.** Descoped from this project's timeframe (Chapters 01-03) but architecturally
   accounted for: a transcript from an ASR component would enter the existing pipeline at the same
   "cleaned text" point a written inquiry does, requiring no change to classification, calibration, or routing
   logic. Radford et al.'s and Southwell's work on ASR robustness (Chapter 02) should be revisited for
   transcript-error handling before this is built.
2. **Build the feedback-retraining pipeline the review log already supports.** `ReviewEvent` rows and the
   `TrainingRun` table (Chapter 04) exist specifically to support this, but nothing currently consumes them.
   A concrete next step is a scheduled job that assembles corrected labels into a new training run once enough
   accumulate, with the same evaluation pipeline (Chapter 06) re-run before any retrained model is promoted.
3. **Replace the synthetic out-of-distribution probe set with genuine out-of-scope messages.** Section 6.3.5's
   0.99 AUROC was measured against 20 synthetic items; a probe set of real spam, misdirected mail, or
   inquiries about an unrelated institution would be a materially harder and more honest test of the energy-score
   gate.
4. **Carry out load, security, and usability testing.** Named in Chapter 06 as strategies this undergraduate
   project did not attempt: concurrent-traffic load testing against realistic institutional volume, a security
   review of the authentication and access-control implementation by someone other than its author, and a
   usability evaluation with real department officers rather than the researcher acting as the only tester.
5. **Validate against genuine, ethically-approved institutional data.** The corpus constructed for this thesis
   (Chapter 03) deliberately avoided real student data; a pilot using anonymized real inquiries, under proper
   ethical approval, is the strongest remaining test of whether the corpus's templated construction
   (Section 6.3.4's discussion of why selective prediction found little to abstain on) generalizes to genuinely
   noisy, unscripted student writing.
6. **Test cross-institution transfer.** Whether the trained classifiers, or the calibration and OOD thresholds fitted
   for them, transfer to a different university's category set and phrasing conventions without retraining from
   scratch is untested and would materially affect how reusable this artifact is beyond the institution it was
   built for.

## 7.6 Chapter Summary

This chapter closed the thesis by assessing all four research objectives as met, through triangulated evidence
across the literature, implementation, and testing chapters rather than any one alone, while reporting the one
qualification the evidence actually supports: a statistically significant transformer advantage for urgency
classification but not for category classification. It gave an honest account of six problems actually
encountered during execution, from dataset-schema mismatch through to a defect discovered in the
evaluation tooling itself, and reflected on the research ideology, skills, and learning curves this project
required. It argued that the calibrated, selectively-abstaining triage pattern implemented here has application
beyond university administration, and set out six future recommendations, each tied to a specific,
previously-identified limitation rather than offered as generic future work. Together with Chapters 01
through 06, this concludes the account of an AI-based university inquiry classification and urgency
prioritization system that was not only designed, but built, tested, and evaluated against evidence the thesis
reports in full.
