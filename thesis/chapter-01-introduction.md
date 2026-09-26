# Chapter 01 — Introduction

## 1.1 Chapter Overview

This chapter introduces the research area, the practical problem that motivates the study, and the
system that was designed, implemented, and evaluated to address it. It explains the problem
background, defines the general and specific problem statements, presents the main research
question and supporting sub-questions, and outlines the research aim and objectives. It also
presents a rich picture of the implemented solution workflow, the resource requirements the
project actually used, the project scope as it stands at final submission, and a summary of how the
study is positioned within Software Engineering, Data Science, and Management Information
Systems contexts.

Since Interim Report 02, the research has progressed from a proposed prototype design to a fully
implemented, tested, and evaluated web application — UniFlow — comprising a calibrated
dual-transformer classification pipeline, a PostgreSQL-backed FastAPI service layer enforcing
department-scoped access control and optimistic concurrency, and a React dashboard used by
department officers and administrators. Sections 1.8 through 1.10 have been updated from the
interim version to describe this implemented system rather than the originally proposed one;
Sections 1.2 through 1.7 remain substantively unchanged because the underlying problem,
motivation, and objectives that justified the work have not changed.

## 1.2 Problem Background

Modern universities operate in a highly service-oriented environment where students and
applicants expect timely answers regarding admissions, examinations, fees, deadlines, timetables,
academic documents, course registration, learning platforms, and administrative procedures.
Universities receive a high volume of inquiries from prospective students, current students,
parents, and external stakeholders through different communication channels, including
text-based and telephone-based inquiries. Many institutions still handle these inquiries through
manual email, help desk, or ticketing workflows that process messages mainly in chronological
order rather than by urgency.

This creates a practical service-management problem. A routine request about general programme
information and a time-critical examination issue may enter the same queue, although their
academic consequences are very different. If urgent inquiries are not recognized early, students
may miss deadlines, fail to resolve examination-related problems, or experience unnecessary
stress. Administrative staff also spend time manually reading, categorizing, and forwarding
inquiries before resolving them, which increases workload and can reduce the consistency of
service quality.

AI and Natural Language Processing provide a technically suitable foundation for addressing this
problem because they enable computational analysis of unstructured inquiry text. Recent
education research shows that AI chatbots can provide immediate support, explanations, and
additional resources for learners and educators, while also raising concerns about reliability,
accuracy, and ethics [1]. In the service-management domain, ticket classification and prioritization
systems have demonstrated that machine learning and deep learning can reduce manual effort by
assigning categories and priority levels to incoming tickets [2], [3].

The research therefore focuses on an AI-assisted triage artifact rather than a fully autonomous
response system. The implemented system classifies the topic of each inquiry, predicts its urgency
level, and presents the result to administrative staff — ordered by priority — for review and
response. A closed-set classifier that is forced to answer every time will confidently misroute the
inquiries it understands least, so the system additionally **abstains** to a shared human-reviewed
queue when its calibrated confidence in a category prediction is below a validated threshold, or
when the inquiry does not resemble any department it was trained on. Voice inquiries were
scoped into the original proposal as a channel extension converted to text via an existing
automatic-speech-recognition component before entering the same pipeline [4], [5]; this extension
was not implemented within the timeframe of the project and is carried forward as future work
(Section 1.10, Chapter 07).

## 1.3 Problem Statement

### 1.3.1 General Problem

The general problem is that manual university inquiry handling is inefficient when inquiry volume
increases and messages differ in urgency. Traditional queue-based processing does not
automatically distinguish between high-impact, deadline-sensitive inquiries and routine
informational requests. This can delay critical responses, increase administrative workload, and
negatively affect student satisfaction, institutional service quality, and the perceived reliability of
university support services.

The problem is not only operational but also managerial. Universities must allocate limited
administrative resources while maintaining fairness, transparency, and responsiveness. A purely
manual process depends heavily on staff availability and individual judgment. A purely automated
system may respond quickly but may fail when the issue is sensitive, ambiguous, or requires
institutional policy interpretation. Therefore, the research problem requires a balanced
AI-supported workflow that identifies priorities while keeping human staff in control of final
decisions — and, critically, that is honest about when its own classification is not reliable enough
to act on unsupervised.

### 1.3.2 Specific Problem

The specific problem is the limited availability of domain-specific AI solutions that classify
university inquiries and predict urgency levels in a way that can be integrated into higher
education administrative workflows, with the access control, auditability, and service-level
guarantees a real administrative deployment requires. Existing NLP systems are often designed for
generic customer support, IT service desks, or commercial complaint management rather than
university-specific inquiry categories such as admissions, examinations, fees, timetables, academic
documentation, and learning platform access.

A second discipline-specific challenge is data suitability. Publicly available support-ticket
datasets can provide useful structures such as text, category, priority, or department, but their
language, labels, and urgency criteria are not directly aligned with higher education
processes [3]. University-focused datasets exist for student questions and query classification,
but they may not always include explicit urgency labels. As a result, this research defines a
mapping and validation process for adapting secondary datasets to the university inquiry context.

A third, implementation-level challenge — one that only became apparent once the system moved
from a proposed design to a working artifact — is that a classifier's raw confidence score is not, by
itself, trustworthy enough to drive an unsupervised routing decision: fine-tuned transformers are
known to be systematically overconfident, and closed-set classifiers cannot recognize inputs that
do not resemble anything in their training distribution. Any deployable inquiry-routing system
therefore needs a calibration and selective-prediction layer on top of the classifier itself, not just
the classifier.

The research gap can be summarized as follows: there is a need for a validated AI-based system
design that combines university inquiry category classification, calibrated urgency prediction,
selective abstention when confidence is insufficient, secondary dataset adaptation, and
human-in-the-loop administrative review inside an access-controlled, auditable service — not
merely a classification model evaluated in isolation. This gap is directly addressed by the
implemented artifact and its evaluation plan.

## 1.4 Research Question

**Main Research Question:** How can an AI-based NLP system be designed, implemented, and
evaluated to automatically classify university inquiries, predict urgency levels, and selectively
abstain from routing decisions it is not confident in, in order to improve response prioritization
within higher education administrative workflows?

Supporting research sub-questions are as follows:

- How can university inquiry categories and urgency levels be defined in a way that is
  meaningful to administrative staff and suitable for supervised machine learning?
- Which secondary/public datasets are most suitable for training, validating, or externally
  testing a university inquiry classification and urgency prediction model?
- How effectively do baseline machine learning models compare with transformer-based NLP
  models for inquiry category classification and urgency-level prediction?
- How should confidence calibration and a selective-prediction threshold be derived so that the
  system abstains from auto-routing when it is likely to be wrong, without abstaining so often
  that automation coverage becomes negligible?
- How should the system be evaluated to ensure that urgent inquiries are detected reliably while
  reducing unnecessary false urgent predictions, and that abstained inquiries are still handled
  correctly by a human reviewer?

The sub-question on voice-based inquiry incorporation that appeared in the interim proposal is
retained as a **future-work question** (Chapter 07) rather than a question this research answers,
since automatic speech recognition was not implemented within the project timeframe.

## 1.5 Research Motivation

The motivation for this research comes from the practical need to improve the speed, fairness, and
consistency of university inquiry handling. In a higher education environment, administrative
delays can directly affect students because deadlines, examinations, document submissions, and
payments are often time-sensitive. A student with an examination issue requires faster attention
than a student asking for general information about future intake dates. However, both messages
may look similar in a manual inbox until they are read by staff.

The academic motivation is to apply AI, NLP, and Software Engineering principles to a real
institutional problem. The study provides an opportunity to investigate text classification, urgency
detection, confidence calibration, selective prediction, dataset adaptation, model comparison,
service architecture design, and ethical AI use in a higher education setting. The work also aligns
with recent interest in AI-powered support systems in universities, where automated tools may
assist students and staff but must be designed carefully to address reliability, privacy, and
governance concerns [1], [6].

## 1.6 Research Aim

The aim of this research is to design, develop, and evaluate an AI-based system that automatically
classifies university inquiries into administrative categories, predicts urgency levels, and
selectively abstains to a human-reviewed queue when its confidence is insufficient, so that staff
can prioritize responses more effectively while maintaining human oversight and accountability
for every routing decision.

## 1.7 Research Objectives

The research objectives are written as research-specific objectives aligned with the implemented
artifact and its evaluation process:

- **To identify** university inquiry categories, urgency criteria, and suitable secondary datasets
  that can support the development of an AI-based inquiry prioritization system.
- **To analyze** existing AI, NLP, ticket classification, support-desk prioritization, calibration,
  and selective-prediction studies in order to justify the research gap and select appropriate
  technologies.
- **To design and develop** a system that preprocesses inquiry text, classifies inquiry categories,
  predicts urgency levels, calibrates the resulting confidence scores, and abstains to a
  human-reviewed queue when calibrated confidence or out-of-distribution evidence indicates
  the prediction should not be trusted.
- **To evaluate** the proposed system using classification metrics such as accuracy, precision,
  recall, F1-score, and confusion matrices; calibration metrics such as Expected Calibration
  Error; and selective-prediction metrics such as risk–coverage curves, comparing performance
  against baseline models.

## 1.8 Rich Picture of the Implemented Solution

The implemented solution follows a human-assisted AI workflow, refined from the interim
proposal once the system was actually built and integrated end to end. Incoming inquiries
currently originate from students through two channels: a web submission form used directly by
administrative staff, and an automated IMAP mailbox poller that fetches, parses, and de-duplicates
unseen email messages on a fixed schedule, filtering out messages from automated/no-reply
senders before they reach the classification pipeline. Both channels converge on the same intake
service, so there is exactly one code path from "raw inquiry text arrives" to "a routing decision is
recorded" regardless of which channel it came in on.

The cleaned text is passed through a shared or dual transformer backbone that produces two
outputs: a **category** prediction, used for department routing, and an **urgency** prediction
(High / Medium / Low), used to compute a response-time deadline. Both outputs are passed
through a **temperature-scaling calibration** step before being read as confidence scores. The
category confidence is then compared against a validated threshold (τ): if it is at or above τ *and*
the message's energy score does not indicate it is out-of-distribution for every known department,
the inquiry is **auto-routed** directly into that department's queue; otherwise it is held in a shared
**Triage** queue for a human to place. Every inquiry, in either path, is assigned a service-level
deadline derived from its predicted urgency and is inserted into its destination queue **ordered by
priority** — most urgent, and soonest-due, first — rather than by arrival time.

Administrative staff (scoped to their own department plus Triage) and administrators (with
visibility across all departments) review inquiries through a web dashboard. For any inquiry they
can inspect the model's per-token attribution for its prediction, accept or correct the predicted
category and urgency, and route an abstained inquiry into the correct department. Every state
change — creation, routing, assignment, status change, escalation, and resolution — is written to
an append-only timeline, and every accept/correct decision is written to a separate review log that
is available (but, at the time of writing, not yet consumed) as the training set for a future
feedback-retraining pipeline. Voice inquiries, shown as a dashed extension in Figure 1, remain a
planned but unimplemented channel; a transcript produced by an external ASR component would
enter the pipeline at the same "cleaned text" point a written inquiry does, requiring no change to
the classification, calibration, or routing logic downstream of it.

```
 Students / Staff                    Automated ingestion
┌───────────────────┐              ┌────────────────────────┐
│  Web submission    │              │  IMAP poller (60s)     │
│  form (admin-only) │              │  parse + de-dup +      │
└─────────┬──────────┘              │  automated-sender      │
          │                          │  filter                │
          │                          └───────────┬────────────┘
          └───────────────┬──────────────────────┘
                           ▼
                 Text preprocessing
                 (clean_text)
                           │
                           ▼
        ┌─────────────────────────────────┐
        │   Transformer backbone          │
        │   Category head   Urgency head  │
        └───────────┬──────────┬──────────┘
                     ▼          ▼
            Temperature-scaling calibration
                     │          │
                     ▼          ▼
         Energy-based OOD check   SLA deadline
         + τ threshold compare    (from urgency)
                     │
        ┌────────────┴────────────┐
        ▼                          ▼
  confidence ≥ τ AND          confidence < τ OR
  in-distribution              out-of-distribution
        │                          │
        ▼                          ▼
  Auto-route to                Abstain to shared
  department queue             Triage queue
        │                          │
        └────────────┬─────────────┘
                      ▼
        Priority-ordered queue (React dashboard)
        (urgency rank, then soonest deadline)
                      │
                      ▼
        Human review: accept / correct / route
        (audit timeline + review log recorded)
                      │
                      ▼
        [Future work: feedback retraining]

  - - - - - - - - - - - - - - - - - - - - - - - - -
  [Future work] Voice call → ASR transcript ─────┘  (enters at "Text preprocessing")
```
*Figure 1. Rich picture of the implemented solution workflow. Dashed elements are planned but
not yet implemented.*

## 1.9 Resource Requirements

### 1.9.1 Hardware

| Hardware Resource | Suggested Specification | Purpose |
|---|---|---|
| Research and Development Laptop/PC | Intel Core i5/i7 or AMD Ryzen 5/7, 16 GB RAM minimum, 512 GB SSD | Backend, frontend, and evaluation-pipeline development; the two loaded transformer models together require over 1 GB of resident memory. |
| Optional GPU Workstation or Cloud GPU | NVIDIA GPU with CUDA support, or a cloud GPU instance (used via Google Colab) | Used for the external fine-tuning of the category and urgency transformer models, which was performed outside the deployed system on Colab. |
| Testing Device | Standard browser-enabled laptop or desktop | Required for demonstrating the web dashboard to the supervisor or evaluators. |
| Backup Storage | External drive or secure cloud storage (Git repository) | Source code, evaluation results, and report versions. |

### 1.9.2 Software

| Software Resource | Tool / Version | Purpose |
|---|---|---|
| Python | 3.13 | Core language for the backend service, the ML/evaluation pipeline, and data cleaning. |
| PyTorch | ≥ 2.2 | Loading and running the fine-tuned transformer models for inference, calibration, and explainability. |
| Hugging Face Transformers | ≥ 4.40 | Tokenizers, model loading, and the shared/dual transformer backbone. |
| scikit-learn, SciPy | ≥ 1.4, ≥ 1.11 | Baseline classifiers (Logistic Regression, Linear SVM), McNemar's test, and bootstrap confidence intervals. |
| Matplotlib | ≥ 3.8 | Confusion matrices, reliability diagrams, and risk–coverage curves for the evaluation report. |
| FastAPI, Uvicorn | ≥ 0.111, ≥ 0.30 | The backend HTTP API, request validation, and OpenAPI schema generation. |
| SQLAlchemy, Alembic-equivalent (`create_all`) | ≥ 2.0 | The ORM layer and schema management for the 14-table PostgreSQL schema. |
| PostgreSQL, psycopg | current stable | Persistent storage for inquiries, predictions, review events, users, and routing policy. |
| Argon2-cffi, PyJWT | current stable | Password hashing and JWT-based session authentication. |
| APScheduler | ≥ 3.10 | In-process scheduling of the SLA-escalation, outbox, and email-intake background jobs. |
| React, TypeScript, Vite | React 19, TS ~6 | The department-officer and administrator dashboard, built as a single-page application. |
| TanStack Query, React Router | current stable | Server-state caching/mutation and client-side routing in the dashboard. |
| pytest | ≥ 8.0 | The automated test suite (43 tests) covering SLA arithmetic, access control, calibration and selective-prediction mathematics, and the review log. |
| Microsoft Word, Mendeley/Zotero, draw.io | — | Academic report writing, citation management, and diagrams. |

## 1.10 Project Scope

| Scope Type | Item | Elaboration |
|---|---|---|
| In Scope | Text-based university inquiries from a web form and an automated email inbox. | The implemented system accepts text input from both channels through one shared intake service and outputs category and urgency predictions. |
| In Scope | Inquiry categories reflecting real department labels: Admissions, Examinations, Fees and Payments, Timetable and Scheduling, Academic Documents, Learning Platform / IT Access, and General Administration. | These categories are the transformer's own label set and are mirrored one-to-one by department records in the system's database. |
| In Scope | Urgency labels: High, Medium, and Low. | Urgency is predicted directly from inquiry text and drives a stored service-level deadline. |
| In Scope | Confidence calibration and selective prediction (abstention). | Temperature scaling and a validated confidence threshold (τ) decide whether an inquiry is auto-routed or held for human placement. |
| In Scope | Energy-based out-of-distribution detection. | Inquiries that do not resemble any trained department are flagged and routed to human review rather than forced into the closest category. |
| In Scope | Model comparison between baseline machine learning and transformer-based approaches. | Baseline comparison, using McNemar's exact test, is used to justify model selection. |
| In Scope | A role-based, department-scoped administrative web dashboard with an audit trail. | Officers see only their own department and the shared Triage queue; every state change and human decision is recorded. |
| In Scope | Automated email intake with duplicate and automated-sender filtering. | Inquiries can arrive by email without a human re-typing them into the web form. |
| Out of Scope | Voice inquiries and any automatic-speech-recognition component. | Proposed in the interim report as a channel extension; not implemented within the project timeframe. Carried forward as future work (Chapter 07). |
| Out of Scope | Fully automated final response generation without staff review. | The artifact supports staff decision-making; it does not answer or close inquiries on its own. |
| Out of Scope | An automated feedback-retraining pipeline. | Human corrections are captured in an append-only review log, but nothing currently consumes that log to retrain the models. |
| Out of Scope | Integration with a live university production system. | The deliverable is a research prototype and evaluated artifact, not a production deployment. |
| Out of Scope | Use of personally identifiable information without approval. | Any real inquiry data must be anonymized and ethically approved before use. |

## 1.11 Chapter Summary

This chapter introduced the research problem and justified the need for an AI-based university
inquiry classification and urgency prioritization system. It explained that manual inquiry handling
can delay urgent inquiries and increase administrative workload, and that a deployable system
additionally needs a calibration and selective-prediction layer so that an overconfident or
out-of-distribution prediction does not silently misroute a student's inquiry. The chapter presented
the research question, motivation, aim, and objectives largely unchanged from the interim
proposal, then updated the rich picture, resource requirements, and project scope to describe the
system as it was actually implemented and evaluated — a calibrated, selectively-abstaining,
role-based web application rather than the originally proposed generic dashboard prototype, with
voice intake descoped to future work. The next chapter reviews the literature underpinning this
implementation, including the calibration, selective-prediction, and explainability techniques the
system relies on that were not yet discussed in the interim submission.
