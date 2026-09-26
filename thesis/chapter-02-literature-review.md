# Chapter 02 — Literature Review

## 2.1 Chapter Overview

This chapter reviews the academic and technical literature relevant to automated university inquiry
classification and urgency-based prioritization. The review is organized around six connected areas: higher
education inquiry management, AI chatbots and student support, support-ticket classification, urgency and
priority prediction, transformer-based NLP, and the confidence-calibration, selective-prediction, and
explainability techniques that the implemented system relies on. The chapter does not only summarize
literature; it critically reflects on how suitable each body of work is for the research problem, and, where the
implemented artifact departs from a body of work's assumptions, explains why.

Since Interim Report 02, the technological-analysis and reflection sections of this chapter have been
substantially extended. The interim version reviewed literature that justified a two-model category/urgency
classifier and a voice-to-text (ASR) extension. Building the system exposed a problem the interim review did
not yet address: a fine-tuned transformer's raw confidence score is not, by itself, a trustworthy basis for an
unsupervised routing decision. This chapter therefore also reviews the confidence-calibration, selective-
prediction, and out-of-distribution detection literature that the implemented calibration layer is built on, and
the explainability literature behind the per-token attribution shown to reviewing staff. The voice/ASR
literature reviewed in the interim report is retained here in reduced form, positioned as the basis for future
work (Chapter 07) rather than as literature justifying a channel that was implemented.

## 2.2 Conceptual Map of the Literature

The conceptual map shows the relationship between the main literature themes. The central problem is
AI-based university inquiry prioritization. This problem is shaped by higher education service workflows,
NLP classification techniques, urgency detection, confidence calibration and selective prediction,
explainability, and ethical governance. Two themes shift position relative to the interim report's map: dataset
adaptation is folded into the domain-overview discussion rather than treated as a standalone theme, and a new
theme — confidence calibration and selective prediction — is added as a first-class node, because it turned
out to be a precondition for any of the other themes being usable in an unsupervised routing decision. The
voice/ASR theme is kept but shown as a future-work extension rather than an implemented pipeline stage,
matching Chapter 01's rich picture.

```
                       Conceptual Map of the Literature

  Higher Education        NLP Text              Urgency / Priority
  Inquiry Management      Classification         Detection
         \                    |                     /
          \                   |                    /
           \                  |                   /
            +---- AI-Based University Inquiry ----+
            |          Prioritization              |
            +---------------+----------------------+
           /                |                 \            \
          /                 |                  \            \
  Confidence Calibration  Explainability   Ethics, Privacy   [Future work]
  & Selective Prediction  (Integrated       & Human           Voice / ASR
  (temperature scaling,   Gradients)        Oversight         extension
  energy-based OOD)
```
*Figure 2. Conceptual map of the literature. The dashed future-work node mirrors Figure 1 in Chapter 01.*

## 2.3 Domain Overview

The domain of university inquiry management includes administrative communication between the
institution and its stakeholders. Common inquiries relate to admissions, examinations, course registration,
fees, timetables, academic documents, and learning platform access. Unlike commercial customer support,
university inquiries can have academic consequences, policy constraints, and strict deadlines. Urgency
therefore cannot be treated only as emotional tone or customer frustration; it must also consider academic
impact, deadline proximity, and the need for escalation.

Recent work on AI chatbots in education shows that conversational AI can provide immediate support,
explanations, and additional resources. Labadze, Grigolia, and Machaidze reviewed 67 studies and reported
benefits for students and educators, while also emphasizing concerns about reliability, accuracy, and ethical
use [1]. This is directly relevant to the implemented system: an inquiry prioritization system may support
faster responses, but it must not create misleading decisions or remove accountability from staff. This
concern is precisely why the implemented system routes low-confidence and out-of-distribution inquiries to
a human-reviewed queue rather than resolving them automatically (Section 2.5.1).

In higher education, AI support systems are commonly discussed in the context of chatbots, personal tutoring,
retrieval-augmented question answering, and student assistance. The research reported in this thesis is
narrower and more operational: it classifies incoming inquiries and prioritizes them for staff review rather
than answering them. This positions the work closer to ticket classification and service-desk automation than
to general-purpose educational chatbots — a distinction the implemented role-based dashboard makes
concrete, since it is a staff-facing triage tool rather than a student-facing conversational agent.

## 2.4 Existing Systems, Frameworks and Designs

Existing approaches can be grouped into manual help-desk systems, rule-based routing, traditional machine
learning classifiers, deep learning classifiers, transformer-based models, and AI-assisted ticket prioritization
frameworks. Each approach was assessed for suitability against the requirements the implemented system
ended up satisfying: department-scoped access control, a stored service-level deadline, and a decision the
system is entitled to make without a human in the loop only when it is calibrated and in-distribution.

| Approach | Description | Strengths | Limitations | Suitability for the Implemented System |
|---|---|---|---|---|
| Manual email/help desk queue | Staff read and respond to messages in chronological or manually sorted order. | Simple and familiar; requires no AI infrastructure. | Slow for high volumes; urgent inquiries may be delayed; inconsistent categorization. | Rejected as a standalone approach; retained only as the informal baseline the system improves on. |
| Rule-based keyword routing | Rules detect words such as exam, payment, urgent, deadline, or admission. | Transparent and easy to implement. | Brittle when wording changes; poor contextual understanding; negation and ambiguity are hard. | Not adopted; a closed-set transformer classifier with calibration was judged more robust to wording variation than a rule set would be. |
| Classical ML classification | TF-IDF with Logistic Regression, SVM, Naive Bayes, or tree-based models [7]. | Fast, interpretable, works well on small datasets, easy to benchmark. | Depends on feature engineering and may miss context. | Retained as the baseline model family the transformer is compared against using McNemar's exact test (Chapter 06). |
| Deep learning models | CNN, RNN, and BiLSTM models learn sequence patterns from text. | Can capture local and sequential patterns. | Needs more data and tuning; may not outperform transformers on limited domain data. | Not adopted as a separate model family; superseded by the transformer baseline comparison. |
| Transformer-based models | BERT/RoBERTa/DistilBERT fine-tuned for category and urgency classification [9], [10], [11]. | Strong contextual representation; effective transfer learning. | Needs careful training, compute resources, and — as this research found — produces overconfident, uncalibrated scores that are unsafe to route on directly. | Adopted as the core classifier, but only once paired with the calibration and selective-prediction layer described in Section 2.5.1. |
| ASR-assisted voice pipeline | Phone calls are transcribed into text and processed by the same NLP model [4], [5]. | Supports multimodal inquiry channels without building a separate voice model. | ASR errors may affect downstream text classification; adds a second component to validate and maintain. | Not implemented within the project timeframe. Retained in the literature as the basis for Chapter 07's future-work recommendation, since a transcript would enter the implemented pipeline at the same preprocessing point a written inquiry does. |

Ticket-BERT is a relevant example from the incident-management domain. Liu, Benge, and Jiang introduced
a language-model approach for labeling incident tickets and highlighted challenges such as changing ticket
descriptions, time-sensitive labeling, and the need for quick adaptation to new labels [2]. These challenges
recur in university inquiry management, where inquiry categories and deadlines shift across semesters and
administrative periods.

Customer-support ticket prioritization research also supports the feasibility of the approach taken. Selvi et al.
developed an NLP-driven system for customer support ticket categorization and prioritization using machine
learning and deep learning models, reporting that BERT performed competitively for urgency prioritization
and that classical models such as Logistic Regression and SVM were strong for categorization [3]. Although
the domain is commercial customer support rather than university administration, the design principle of
category-plus-priority prediction, sitting behind a review step rather than replacing it, is exactly the shape the
implemented system takes.

None of this literature, however, addresses what happens when the classifier is simply wrong with high
confidence, or when an inquiry does not resemble anything the classifier was trained on. That gap is the
subject of Section 2.5.1 and is what most changed between the interim proposal and the implemented system.

## 2.5 Technological Analysis

### 2.5.1 Algorithmic Analysis

The algorithmic component of the implemented system divides into preprocessing, representation learning,
category and urgency classification, confidence calibration, selective prediction, out-of-distribution
detection, and evaluation. Preprocessing cleans text, normalizes casing, handles contractions, and removes
unnecessary symbols while preserving critical information such as dates, examination terms, payment
references, and deadline-related expressions — generic text cleaning is inappropriate here because it can
remove exactly the words that signal urgency or domain relevance.

Baseline representation is created using TF-IDF, which converts each inquiry into a sparse vector reflecting
the importance of words in the corpus. This representation is used with Logistic Regression, Support Vector
Machines, and Multinomial Naive Bayes as the baseline models the transformer is compared against. Baseline
models remain important because they provide a transparent, low-variance benchmark, and classical ML
models have achieved strong category-classification performance on comparable ticket data [3].

Transformer-based representation is the core advanced approach. BERT introduced bidirectional contextual
pretraining and can be fine-tuned with a small task-specific output layer for classification [9]. RoBERTa
improved BERT-style pretraining by optimizing training choices and data usage [10]. DistilBERT reduced
model size while retaining much of BERT's language-understanding performance, which matters where
inference has to run inside a request-serving API rather than a batch job [11]. The implemented system uses a
shared-or-dual transformer backbone (Figure 1, Chapter 01) that produces both a category output, used for
department routing, and an urgency output (High/Medium/Low), used to compute a service-level deadline —
whether the two heads share an encoder or are trained as separate fine-tuned models is a configuration choice
(`prefer_multitask`) rather than a fixed architectural commitment, since both configurations were kept
comparable under the same evaluation protocol.

This is also where the interim review's coverage stopped being sufficient once the system was built. Guo et
al. showed that modern neural networks, including the architectures underlying BERT-family models, are
systematically miscalibrated: their softmax confidence overstates the true probability of a correct prediction,
and this overconfidence gets worse, not better, as models get deeper and more accurate on held-out accuracy
alone [15]. A classifier that reports 98% confidence on a misclassified inquiry is not a hypothetical risk for
this system — it is the default behavior of an unadjusted fine-tuned transformer, and routing on that number
directly would silently misroute exactly the inquiries the model is least equipped to handle. Guo et al.'s
temperature-scaling method addresses this with a single learned scalar temperature applied to the logits
before the softmax, calibrated on a held-out validation set so that a stated confidence of, say, 80% is
empirically correct about 80% of the time; its simplicity, small number of parameters, and negligible effect
on the model's ranking of predictions make it well suited to a two-head classifier that needs its category and
urgency confidences calibrated independently without retraining either head. The implemented system's
calibration step, and its Expected Calibration Error evaluation metric (Chapter 06), are a direct application of
this method.

Calibration alone answers "how much should I trust this probability," not "should I act on this probability at
all." That second question is selective prediction: a classifier that is allowed to abstain rather than being
forced to output a label for every input. A calibrated confidence score is the natural signal to threshold for
this decision, and the validated threshold τ described in Chapter 01 is exactly that: an inquiry is auto-routed
only when its calibrated category confidence is at or above τ, and is otherwise held for a human to place in
Triage. This keeps the system's behavior legible in exactly the terms the domain-overview literature raised —
staff, not the model, remain accountable for the inquiries the model was not confident about [1].

A calibrated confidence threshold still assumes the inquiry resembles something the classifier was trained
on. It does not, by itself, protect against an inquiry from a department, language register, or topic the training
data never covered — a closed-set classifier will still assign such an inquiry to whichever known category is
least dissimilar, with a confidence score that calibration was never validated to interpret correctly out-of-
distribution. Liu et al. proposed an energy-based scoring function derived from a classifier's logits that is
higher (indicating unfamiliarity) for out-of-distribution inputs than for in-distribution ones, without requiring
the network to be retrained or an auxiliary outlier detector to be added, and reported that it outperforms
softmax-confidence-based OOD detection [16]. The implemented system uses this energy score as a second,
independent gate alongside the τ threshold: an inquiry is auto-routed only when its confidence clears τ and
its energy score does not flag it as out-of-distribution for every known department; either condition failing
sends it to Triage instead.

Evaluation of the classification components follows the same shape as the interim report's plan — accuracy,
precision, recall, macro F1-score, and confusion matrices, with particular attention to recall for High-urgency
inquiries, since a false negative there is the failure mode with the worst real-world consequence. The
implemented evaluation plan extends this with Expected Calibration Error (for the calibration claim),
risk-coverage curves and area-under-risk-coverage (for the selective-prediction claim — how much accuracy
is gained by abstaining on the least-confident fraction of inquiries), and McNemar's exact test with bootstrap
confidence intervals for the baseline-versus-transformer comparison, so that a reported improvement is
supported by a significance test rather than a single point estimate.

### 2.5.2 Design Analysis

The implemented architecture follows a modular, layered design so that data ingestion, preprocessing, model
inference, calibration, routing, persistence, and the review dashboard can be developed, tested, and reasoned
about separately. This is consistent with the interim report's design rationale — it supports maintainability
and future extension, and it keeps the door open for the ASR component to be added later without changing
anything downstream of the point where a transcript would enter as cleaned text.

Two design decisions extend beyond what the interim report anticipated, both following directly from
building on a shared FastAPI service backed by PostgreSQL rather than a single-process prototype script.
The first is access control: department officers are scoped to their own department plus the shared Triage
queue, and administrators see every department, enforced as a SQL predicate applied before rows leave the
database rather than a filter applied after. The second is auditability: every state change to an inquiry —
creation, routing, assignment, status change, escalation, resolution — is written to an append-only timeline,
and every accept/correct decision a reviewer makes is written to a separate review log. Neither mechanism
appears in the interim proposal's architecture diagram, because neither is necessary until the system is
handling department-scoped, accountable decisions rather than producing a single prediction for a research
notebook.

Human-in-the-loop operation remains the central design decision, unchanged in principle from the interim
report but concretized considerably: the system does not answer, reject, or close an inquiry on its own in any
case. When confidence and out-of-distribution checks both pass, the inquiry is auto-routed into a department
queue where staff still see it, act on it, and can correct its category or urgency; when either check fails, it is
held in a shared Triage queue for a human to place. This is a stricter reading of human-in-the-loop than the
interim report's "system displays a suggested label" description, because it makes the model's own confidence
the gate on whether staff see a routing decision already made or a decision still to be made.

Explainability was an intended but unspecified design consideration in the interim report ("highlighted
keywords or simple rationale indicators"). The implemented system instead uses Integrated Gradients, an
axiomatic feature-attribution method introduced by Sundararajan, Taly, and Yan that computes each input
token's contribution to a model's output by integrating the gradient of the output with respect to the input
along a straight-line path from a neutral baseline to the actual input [17]. Integrated Gradients was selected
over ad-hoc keyword highlighting because it satisfies two axioms — sensitivity and implementation invariance
— that a heuristic keyword list cannot guarantee, and because it operates directly on the fine-tuned
transformer without requiring a separate explanation model to be trained and kept in sync. Reviewing staff
see the resulting per-token attribution in the ticket detail view's Explanation tab, framed as a rationale for
review rather than as a proof of correct reasoning, consistent with the interim report's caution that such
explanations "should not be treated as perfect model reasoning."

### 2.5.3 Workflow Analysis

The implemented workflow begins when an inquiry is received through the web submission form or the
automated IMAP mailbox poller; the interim report's phone/help-desk-ticket channels and its ASR
transcription step are not part of the implemented workflow, for the reasons given in Section 2.4. The poller
adds two stages the interim workflow did not need: de-duplication (so a message already ingested is not
processed twice) and automated-sender filtering (so a mailbox-generated notification is never treated as a
student inquiry) — both a direct consequence of moving from a manually curated dataset to a live mailbox,
where nothing about the sender or the message's authenticity is otherwise known in advance.

From there, the workflow follows the algorithmic pipeline reviewed in Section 2.5.1: cleaned text into the
classifier, classifier logits through temperature scaling, the calibrated category confidence compared against
τ, and the energy score compared against the out-of-distribution threshold. The literature behind each of
these steps was written and evaluated independently — Guo et al. calibrate a single classifier [15], Liu et al.
detect out-of-distribution inputs for a single classifier [16] — and neither addresses what should happen
when both checks are combined into one routing decision with a service-level deadline attached. That
combination, and the priority ordering of the resulting queue by urgency rank and then soonest deadline
rather than by arrival time, is a workflow-design contribution of this research rather than a technique taken
directly from a single source.

The workflow still supports the continuous-improvement loop the interim report envisioned: when staff
accept or correct a predicted category or urgency label, the decision is written to the review log. At the time
of writing, this log is populated but not yet consumed by a retraining pipeline, so it is reported here as an
evaluated design output rather than a demonstrated feedback loop; the reasons for that scope decision, and
the recommendation to build the retraining pipeline it enables, are discussed in Chapter 07.

## 2.6 Reflection and Research Gap

The literature confirms that AI chatbots and NLP systems can support educational and service-management
tasks, but also that reliability, ethics, privacy, and contextual suitability remain live concerns [1]. Support-
ticket classification research demonstrates that both classical ML and transformer-based models can classify
and prioritize tickets [2], [3]. None of this literature, however, questions whether the classifier's own
confidence score can be trusted enough to act on without a human check — an assumption every one of these
systems makes implicitly by producing a single predicted label per input.

The calibration and out-of-distribution detection literature closes part of that gap, but only in isolation from
the university-inquiry problem: Guo et al. demonstrate that modern neural networks are miscalibrated and
show how to fix it for a single classifier's output [15], and Liu et al. show how to flag inputs a classifier was
never trained to recognize [16], but neither paper is concerned with routing decisions, service-level
deadlines, department-scoped access control, or an audit trail — the operational requirements a real
administrative deployment imposes. Equally, the explainability literature that Integrated Gradients belongs to
is concerned with faithfully attributing a model's output to its input, not with what a non-technical reviewer
should be shown or how much weight they should place on it [17].

The research gap can now be stated more precisely than in the interim report: there is a need for a validated
AI-based system design that combines university inquiry category classification, calibrated urgency
prediction, selective abstention grounded in both calibrated confidence and out-of-distribution evidence, per-
prediction explainability, and human-in-the-loop administrative review inside an access-controlled, auditable
service — not a classification model evaluated in isolation, and not a calibration or OOD-detection technique
demonstrated on a benchmark dataset disconnected from an operational workflow. The voice/ASR literature
reviewed in Section 2.4 remains relevant to this gap only as a future extension: Radford et al.'s and
Southwell's work on ASR robustness and error modes [4], [5] would need to be revisited before a transcript
channel is added, but this research does not close that part of the gap and reports it honestly as future work
(Chapter 07) rather than as an implemented contribution. This gap is directly addressed by the implemented
artifact described in Chapters 04 and 05 and by the evaluation plan carried out in Chapter 06.
