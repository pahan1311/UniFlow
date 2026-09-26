# Chapter 03 — Methodology

## 3.1 Research Paradigm

This research adopted a pragmatic paradigm supported by Design Science Research. Pragmatism was suitable
because the research is concerned with solving a real institutional problem using an artifact and evaluating
whether the artifact works in practice; the study did not rely on theory or observation alone, but combined
literature, dataset construction, model development, prototype design, and empirical evaluation into one
iterative process.

Design Science Research is appropriate because the main output is an IT artifact: an AI-based inquiry
classification and urgency prioritization system. Hevner et al. explain that knowledge in design science is
gained through building and applying an artifact to a problem domain [12]. Peffers et al. further present a
Design Science Research Methodology that includes problem identification, objectives, design and
development, demonstration, evaluation, and communication [13]. These stages structured the research as
planned in the interim report, and — as Section 3.6.1 discusses — the methodology's own provision for an
iterative feedback loop between evaluation and design turned out to be necessary rather than a formality:
evaluating the first fine-tuned classifier surfaced the overconfidence problem described in Chapter 02, which
sent the project back through Design and Development before a trustworthy Evaluation stage was possible.

## 3.2 Research Approach

The study used a mixed deductive and inductive approach. The deductive aspect came from applying
established theories, methods, and metrics from NLP, machine learning, and Design Science Research: that
text classification models can learn patterns from labelled data, and that model performance and reliability
can be measured using precision, recall, F1-score, Expected Calibration Error, and risk-coverage analysis.
The inductive aspect appeared during data construction and error analysis, where patterns in the constructed
inquiry corpus — and in the classifier's failure modes once it was evaluated — shaped decisions that the
interim proposal had not anticipated, most notably the addition of a calibration and selective-prediction layer.

The approach remained pragmatic in the sense that method selection was based on research usefulness rather
than allegiance to one paradigm. Classical machine learning models were retained as baselines because they
are interpretable and efficient; transformer models were adopted because they are effective for contextual
language understanding; and human review was retained throughout because university administrative
decisions require accountability and may involve sensitive student circumstances.

## 3.3 Research Strategy

The research strategy was Design Science Research Methodology (DSRM). The implemented system is the
artifact, and its evaluation is central to the research contribution. The DSRM stages were applied as follows:

1. **Problem Identification**: Identify inefficiencies in manual university inquiry handling and the risk of
   delayed urgent inquiries.
2. **Define Objectives**: Establish system objectives for inquiry category classification, calibrated urgency
   prediction, selective abstention, and staff review.
3. **Design and Development**: Design the architecture, dataset schema, preprocessing pipeline, model
   pipeline, calibration and selective-prediction layer, and the role-based dashboard.
4. **Demonstration**: Apply the implemented pipeline to the constructed inquiry corpus to show how
   classification, calibration, and prioritization work end to end.
5. **Evaluation**: Compare baseline and transformer models using classification, calibration, and
   selective-prediction metrics, with significance testing.
6. **Communication**: Present the methodology, results, discussion, limitations, and recommendations in
   the interim and final thesis reports.

Unlike the interim proposal, which presented these stages as a forward-only pipeline, Stage 5 fed back into
Stage 3 once during execution: initial evaluation showed the fine-tuned transformer's raw confidence scores
were unusable as a routing signal (Chapter 02), which meant Design and Development had to be revisited to
add temperature-scaling calibration, the τ-threshold selective-prediction decision, and energy-based
out-of-distribution detection before Evaluation could be trusted. This is the iterative feedback loop the
interim proposal's DSRM diagram already allowed for (Figure 4, Section 3.5.1), and it is reported here as
something the methodology actually exercised, not only planned for.

## 3.4 Fact Collection Mechanisms

Fact collection used four mechanisms: literature review, secondary dataset review, dataset construction, and
prototype evaluation. The literature review focused on peer-reviewed and credible sources related to AI
chatbots in education, support-ticket classification, urgency detection, transformer-based NLP, confidence
calibration, out-of-distribution detection, and explainability. A PRISMA-inspired search-and-screening
process was used to keep the review transparent, although this undergraduate research does not claim to be a
full systematic review [14].

The dataset strategy combined secondary public datasets with author-constructed inquiries. The public
sources identified in the interim proposal — university-related query datasets and IT/customer-support ticket
datasets from Kaggle, Mendeley, and Zenodo — provided realistic phrasing and structural variety, but, as
anticipated, their category and urgency conventions did not map cleanly onto the university-specific schema
this research needed. Rows drawn from these sources were relabelled against the project's own category and
urgency definitions (Section 3.4.1); rows and phrasing variants that public sources did not adequately cover
for a given category/urgency/scenario combination were supplemented with author-constructed inquiries
following the same schema, so that the final corpus is a deliberate mix rather than a single public dataset used
as-is. This directly implements the "mapping and validation process for adapting secondary datasets to the
university inquiry context" that Chapter 01's specific-problem statement identifies as a discipline-specific
challenge.

The constructed corpus totals approximately 8,000 labelled inquiries for training, with separate,
non-overlapping validation and test files held out for model selection and final evaluation respectively. Each
row carries the following fields:

| Field | Description |
|---|---|
| Query_ID | Unique identifier for the inquiry row; a suffixed variant (e.g. `404869_A`) marks a phrasing variant generated from the same underlying scenario. |
| Student_Query | The inquiry text, as a student or applicant might plausibly phrase it. |
| Urgency_Label | The ground-truth urgency: High, Medium, or Low. |
| University_Category | The ground-truth department-facing category (Admissions, Examinations, Fees and Payments, Timetable and Scheduling, Academic Documents, Learning Platform / IT Access, or General Administration). |
| Scenario_Family | A short code (e.g. `DOC05A`, `TIM01B`) grouping phrasing variants that were generated from the same underlying scenario, kept to support scenario-level error analysis and to trace a variant back to the situation it represents. |
| Primary_Intent | A short human-readable label for the specific situation the inquiry represents (e.g. "digital certificate", "class timetable clash"), finer-grained than the category and useful for qualitative error review. |

Scenario_Family and Primary_Intent are refinements over the data dictionary proposed in the interim report
(Appendix B): they were added during dataset construction once it became clear that category and urgency
labels alone were too coarse to audit whether misclassifications clustered around specific real-world
situations (for example, whether "digital certificate" and "certified copies" inquiries were being confused
with each other more than with unrelated Academic Documents inquiries).

### 3.4.1 Label Schema

The label schema retained from the interim proposal has two output dimensions. The first is inquiry category,
matching the university's own department structure. The second is urgency level: High, Medium, or Low.

| Urgency Label | Definition | Example Inquiry Type |
|---|---|---|
| High | Immediate or near-deadline issue with academic, financial, or administrative consequences. | Cannot access exam portal today; payment issue blocking registration; urgent document required before deadline. |
| Medium | Requires staff action within a reasonable period but does not indicate immediate harm. | Request for fee confirmation, timetable correction, admission document clarification for the upcoming week. |
| Low | General inquiry, information request, or non-time-sensitive clarification. | Programme information, general scholarship inquiry, future intake dates, campus facilities inquiry. |

## 3.5 Research Methodology Execution Workflow

The methodology execution workflow followed the DSRM stages while adding dataset management and model
evaluation activities, with the iterative feedback loop described in Section 3.3 shown explicitly in Figure 4.

```
        Design Science Research Methodology Execution Workflow

  1. Problem      2. Objectives    3. Literature &    4. Design &
  Identification  Definition       Dataset Review      Development
       |               |                 |                  |
       +---------------+-----------------+------------------+
                                                              |
                                                              v
       +--------------------------------------------------- 6. Evaluation &
       |                                                     Communication
       |     5. Demonstration & Testing  <-------------------+
       |               ^
       |               |
       +---------------+
   Iterative feedback: calibration/selective-prediction gap found during
   evaluation sent the project back to stage 4 before evaluation was trusted.
```
*Figure 4. Research methodology execution workflow, with the feedback path actually exercised during this
project highlighted.*

| Stage | Execution Activity | Research Output |
|---|---|---|
| Problem Identification | Reviewed the practical inquiry-handling problem, the interim proposal, and supervisor feedback. | Defined problem background, problem statement, and research question (Chapter 01). |
| Relevance Justification | Applied recent literature and higher-education workflow analysis. | Justified need for AI-assisted, calibration-aware inquiry triage. |
| Comparative Analysis and Gap Justification | Compared chatbots, ticket classifiers, transformer models, and calibration/OOD/explainability literature. | Research gap: lack of a university-specific, calibration-aware, auditable urgency-prioritization artifact (Chapter 02). |
| Define and Finalize Objectives | Converted the problem into research-specific objectives. | Four objectives covering identification, analysis, design/development, and evaluation (Chapter 01). |
| Design / Development / Data Management | Constructed the labelled corpus, trained baseline and transformer models, added calibration/selective-prediction/OOD, built the FastAPI/PostgreSQL/React system. | Implemented artifact and documented data pipeline (Chapters 04-05). |
| Evaluation and Communication | Evaluated model, calibration, and selective-prediction metrics; communicated findings. | Evaluation results, limitations, and recommendations (Chapters 06-07). |

### 3.5.1 Problem Identification

The initial problem was identified from the interim research proposal and the practical workflow of university
inquiry handling: manual inquiry processing does not automatically separate urgent inquiries from routine
ones, delaying time-sensitive matters and increasing administrative burden.

### 3.5.2 Relevance Justification

The relevance of the problem is justified by the growing use of AI support systems in education and the
proven value of automated ticket classification in service domains. The research is relevant to MIS because it
improves administrative information flow, to Software Engineering because it produces a prototype artifact,
and to Data Science because it evaluates predictive models — including their calibration and reliability, not
only their accuracy — on labelled text data.

### 3.5.3 Comparative Analysis and Gap Justification

The literature review (Chapter 02) compares AI chatbots, ML-based ticket classifiers, transformer models,
and — extending the interim review — confidence-calibration, selective-prediction, and out-of-distribution
detection literature. This comparison confirmed that existing approaches are useful but not directly sufficient:
none address whether a classifier's own confidence score can be trusted enough to act on without a human
check. The research gap was accordingly refined during execution, from "a domain-adapted classification and
prioritization artifact" in the interim proposal to "a domain-adapted, calibration-aware, selectively-abstaining
artifact with an auditable human-review layer."

### 3.5.4 Define and Finalize Objectives

The interim objectives (identifying categories/datasets, analyzing existing technologies, designing/developing
the artifact, and evaluating it) were retained, with the design-and-development and evaluation objectives
extended in scope once calibration and selective prediction were identified as necessary, as reflected in the
final objectives stated in Chapter 01.

### 3.5.5 Design, Development, Data Management and Handling

Data management began with the dataset construction described in Section 3.4: candidate public sources were
reviewed for license, fields, and label quality; text was cleaned while preserving urgency-bearing language;
and the unified schema (Query_ID, Student_Query, Urgency_Label, University_Category, Scenario_Family,
Primary_Intent) was applied consistently across the combined corpus, with training, validation, and test
inquiries kept in separate files so that no inquiry used for model selection or final evaluation was also used for
training.

Model development used the shared-or-dual transformer design discussed in Chapter 02, fine-tuned externally
on a cloud GPU (Section 1.9.1) and evaluated through a dedicated evaluation pipeline covering baseline
comparison, calibration, selective-routing risk-coverage analysis, out-of-distribution detection, and inference
cost. That pipeline was first exercised end-to-end on a clearly-marked synthetic smoke-test dataset — generated
specifically to have no evidentiary value — to confirm every stage executed correctly and every figure
rendered before it was ever pointed at the real, constructed corpus; scores produced on the synthetic set are
excluded from Chapter 06 by construction, not merely by convention. The service layer, dashboard, and
review workflow were built as described in Chapter 02's design analysis and are detailed in full in Chapters
04 and 05.

### 3.5.6 Evaluation and Communication

Evaluation uses accuracy, precision, recall, macro F1-score, confusion matrices, and class-level analysis, with
High-urgency recall treated as a safety-oriented measure, exactly as planned in the interim report. This is
extended with Expected Calibration Error, risk-coverage and area-under-risk-coverage analysis for the
selective-prediction claim, energy-score AUROC for out-of-distribution detection, and McNemar's exact test
with bootstrap confidence intervals for the baseline-versus-transformer comparison — all reported in full
against the real constructed corpus in Chapter 06. Communication occurred through the interim report,
supervisor discussions, this final thesis, and the review presentation.

## 3.6 Project Management Methodology

A hybrid Scrum-Kanban methodology was selected and followed. Scrum was suitable because the project
divided into short research-and-development iterations — literature review, dataset construction, baseline
modelling, transformer modelling, calibration/OOD integration, service-layer development, frontend
development, and evaluation. Kanban was useful because research tasks changed based on supervisor
feedback, dataset construction progress, and evaluation findings; a visual Kanban board tracked tasks under
To Do, In Progress, Review, and Completed.

This hybrid approach suited an undergraduate research project because it gave structure without becoming
overly heavy. Weekly or bi-weekly goals were aligned with supervisor meetings, while Kanban accommodated
the unplanned calibration/OOD work item and the mid-project decision to rebuild the service layer on FastAPI
and PostgreSQL rather than extend the original Streamlit/SQLite prototype, both of which a strictly sequential
Scrum-only plan would have handled less gracefully.

### 3.6.1 Project Timeline

The timeline below is the schedule set at the interim stage, organized from problem finalization to final
submission. It is retained here as the planned schedule against which execution is compared, rather than
rewritten with invented dates for every activity.

| Period | Planned Activity |
|---|---|
| February - March 2026 | Proposal refinement, problem background, research question, objectives, and initial supervisor review. |
| March - May 2026 | Literature review, dataset search, dataset suitability analysis, and research gap finalization. |
| May - June 2026 | Dataset mapping, cleaning, annotation guide, baseline model design, and interim report completion. |
| June - July 2026 | Baseline model implementation, transformer fine-tuning, and prototype design. |
| July - August 2026 | Evaluation, error analysis, comparison, dashboard improvement, and draft thesis writing. |
| August - September 2026 | Final thesis refinement, formatting, supervisor feedback incorporation, viva preparation, and final submission. |

Execution broadly followed this schedule, with two deviations driven by the DSRM feedback loop described in
Section 3.3. First, evaluating the fine-tuned transformer's confidence scores during the July-August evaluation
window showed they were miscalibrated and unsafe to route on directly, which inserted an unplanned
calibration, selective-prediction, and out-of-distribution-detection work item between model development and
final evaluation rather than treating evaluation as a single terminal step. Second, the service architecture
originally planned around the interim prototype's Streamlit interface and SQLite storage was rebuilt on
FastAPI, PostgreSQL, and a React dashboard once department-scoped access control, optimistic concurrency,
and an auditable review log were identified as requirements a single-process prototype could not satisfy; this
ran in parallel with, rather than after, model refinement. Voice/ASR integration, originally scheduled inside
the June-July window, was descoped for the reasons given in Chapter 01 and carried forward to Chapter 07's
future-work recommendations instead of being executed.

### 3.6.2 Ethical Considerations

Ethical considerations remain relevant because university inquiries — real ones — may contain personal
information, academic status, financial information, and sensitive student circumstances, even though the
following principles describe how this specific project's data sources kept that risk low rather than how a
production deployment would need to manage real institutional data:

- No real student inquiry records were used to train or evaluate the models. The constructed corpus
  (Section 3.4) combines relabelled public secondary data with author-constructed scenario-based inquiries;
  no row in it originates from a real student's actual submission, so no anonymization step was exercised
  against genuine personal data during model development.
- The web-form and email-intake channels used during development were exercised with the researcher's
  own test submissions and, during one verification round, inadvertently ingested automated
  account-security notification emails from an email provider — not real third-party student inquiries. That
  incident is what motivated the automated-sender filtering described in Chapters 01 and 05; it did not
  expose any student's personal data, since no student had used the system at that point.
- The system does not make final decisions about student requests, eligibility, penalties, or policy
  interpretation; it assists administrative staff, who remain responsible for every review outcome.
- Dataset sources, construction method, labelling criteria, and limitations are documented transparently in
  Section 3.4 and Chapter 09 (Appendices) rather than asserted without evidence.
- Potential model bias remains a live concern precisely because part of the corpus is author-constructed:
  scenario templates reflect the researcher's own assumptions about how students phrase requests, and this
  is reported as a limitation in Chapter 07 rather than treated as resolved.
- Should genuine, anonymized institutional inquiry data become available for future work (Chapter 07), the
  principles set out in the interim report continue to apply unchanged: no personally identifiable
  information without ethical approval, and identifiers removed or replaced with neutral placeholders before
  any such data is used.
- Model outputs are presented to staff as recommendations with confidence scores, not as ground truth,
  consistent with the calibrated, selectively-abstaining design described in Chapter 02.

## 3.7 Chapter Summary

This chapter described the methodology actually followed for this research, building on the pragmatic,
Design-Science-Research strategy set out in the interim report. The methodology included literature review,
dataset construction combining public secondary sources with author-generated inquiries, model development,
prototype design, evaluation, and communication, executed through a hybrid Scrum-Kanban project
management approach. Two points depart materially from the interim plan and are reported honestly here
rather than smoothed over: the DSRM evaluation stage fed back into design and development once because the
first fine-tuned classifier's confidence scores proved unusable without calibration, and the service architecture
was rebuilt around FastAPI, PostgreSQL, and React once auditability and access-control requirements
outgrew the original prototype. Ethical considerations were kept low-risk in practice by not training or
testing on real student data, while the principles that would govern real institutional data remain documented
for any future work that uses it. The next chapter specifies the system's functional and non-functional
requirements as implemented.
