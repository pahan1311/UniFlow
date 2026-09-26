# Chapter 04 — System Requirement Specification

## 4.1 Chapter Overview

This chapter specifies the functional and non-functional requirements of the implemented system as they were
actually built, rather than as originally proposed. It identifies the system's stakeholders, describes how
requirements were derived and validated, and presents the system through a use-case diagram with detailed
specifications for its three most consequential use cases, a class diagram of the persisted domain model, an
activity diagram of the inquiry lifecycle, three sequence diagrams covering the auto-route, human-review, and
automated-intake paths, a state chart of a ticket's queue status, and the system architecture diagram. It closes
with the functional and non-functional requirements the implemented system satisfies, each traceable to a
concrete module or database constraint rather than stated in the abstract.

## 4.2 Stakeholder Analysis

| Stakeholder | Role in the System | Primary Concerns |
|---|---|---|
| Administrator | Full visibility across all departments; the only role permitted to submit a new inquiry through the web form, promote a model version, or activate a routing policy. | Correct routing across the whole institution, model governance, and system-wide oversight. |
| Department Officer | Scoped to their own department plus the shared Triage queue; reviews, accepts, corrects, and resolves inquiries. | A trustworthy, priority-ordered queue; confidence that a correction is recorded and acted on. |
| Viewer | A read-only role supported by the access-control model (`Role.VIEWER`) for stakeholders who need visibility without the ability to act (e.g. an auditor or a supervisor reviewing activity). | Read access to inquiries and analytics without write access. |
| Student / Applicant | Submits an inquiry indirectly, either by a channel an administrator relays into the web form or, more directly, by sending an email to the monitored mailbox. Never authenticates against the system. | A timely, correctly prioritized response, without needing to know the system exists. |
| Scheduler (system actor) | Runs the SLA-sweep, outbox-drain, email-intake, and retention jobs in-process on a fixed interval. Acts as `SYSTEM`, a synthetic Principal with admin rights and no human `actor_id`, so automated actions are distinguishable from human ones in the audit trail. | Timely escalation, reliable email ingestion, and safe unattended operation without duplicating work across API workers. |
| Supervisor / Evaluator | External to the running system; consumes the dashboard, the audit trail, and the evaluation report (Chapter 06) to assess the artifact. | Evidence that classification, calibration, and human oversight work as claimed. |

## 4.3 Operationalization Process

Because this is a Design Science Research artifact rather than a survey-based study, requirements were not
gathered through a questionnaire; they were operationalized from three concrete sources, each mapped to a
research objective from Chapter 01:

1. **Literature-derived requirements** (Chapter 02): the requirement that the system calibrate its confidence,
   support selective abstention, and detect out-of-distribution inquiries traces directly to the research gap
   identified in Section 2.6, and satisfies Objective 3 ("to design and develop a system that ... calibrates the
   resulting confidence scores, and abstains ... when confidence or out-of-distribution evidence indicates the
   prediction should not be trusted").
2. **Stakeholder role definitions carried forward from supervisor review**: the department-scoping and
   admin/officer distinction were established during the interim-report stage and validated again once
   implemented, by confirming that an officer account genuinely cannot read or act on another department's
   inquiry (Section 4.6, NFR-2).
3. **Requirements discovered during implementation**: several requirements were not anticipated at the
   interim stage and were added once a concrete risk became visible while building the system rather than
   being planned upfront. The automated-sender filter (Section 4.6, FR-11) was added only after a live
   verification round ingested automated account-security notification emails as if they were student
   inquiries (Chapter 03, Section 3.6.2). The optimistic-concurrency requirement (Section 4.6, NFR-5) was
   added once it became clear that two officers acting on the same Triage ticket at the same time was a
   realistic race condition a single-process prototype would not have surfaced. Both are reported here as
   requirements the system satisfies, with the incident or reasoning that produced them made explicit rather
   than presented as if they had been anticipated from the outset.

Requirements were validated by exercising the running system directly — issuing real HTTP requests against
the API, inspecting the resulting database rows, and confirming role-scoping and concurrency behaviour under
deliberately constructed conflicting scenarios — rather than by inspection of the code alone. The formal test
suite that encodes these checks is presented in Chapter 06.

## 4.4 System/Model Analysis

### 4.4.1 Use Case Diagram with Specifications

```
                          UniFlow Use Case Diagram

  Student/Applicant --(email)--> [Automated Email Intake] <---- Scheduler
                                          |
  Administrator ----> [Submit Inquiry (web form)]
       |
       +--------------> [Promote Model Version]
       +--------------> [Activate Routing Policy]
       |
       +--------------> [View All Department Queues]
       |
  Department Officer -> [View Own Department Queue]
       |          \----> [View Triage Queue]
       |          \----> [Review Prediction (Accept/Correct)]
       |          \----> [Route Triage Inquiry to Department]
       |          \----> [Update Ticket Status / Assignee]
       |          \----> [View Explanation (Integrated Gradients)]
       |          \----> [View Analytics]
       |
  Viewer ------------> [View Queue (read-only)]
       |          \----> [View Analytics (read-only)]

  (all human actors) -> [Login] -> [Refresh Session] -> [Logout]

  Scheduler ----------> [SLA Escalation Sweep]
       |          \----> [Outbox Drain (send acknowledgement)]
       |          \----> [Retention Redaction]
```
*Figure 6. Use case diagram. Administrator and Department Officer both inherit every Viewer use case;
only the additive and restricted use cases are drawn against each actor to keep the diagram legible.*

**Use Case Specification: Submit Inquiry and Auto-Route or Abstain**

| Field | Description |
|---|---|
| Primary Actor | Administrator (web form) or Scheduler (email intake) |
| Preconditions | Caller is authenticated as Administrator, or the message arrived via the monitored mailbox; an active `ModelVersion` and `RoutingPolicy` exist. |
| Main Flow | 1. Actor submits inquiry text (and, for email, a `source_ref` Message-ID). 2. System computes an exact-hash and vector-similarity duplicate check. 3. System classifies category and urgency, applies temperature scaling, and compares the calibrated category confidence against τ and the energy score against the out-of-distribution threshold. 4. If both checks pass, the inquiry is routed to the predicted department (`RoutingDecision.AUTO`); the deadline is computed from the department's `SlaPolicy` for the predicted urgency. 5. An `InquiryEvent(CREATED)` is written and the inquiry appears in its destination queue, ordered by priority. |
| Alternate Flow | If the confidence check or the out-of-distribution check fails, the inquiry is instead assigned `RoutingDecision.TRIAGE` or `OUT_OF_SCOPE` and placed in the shared Triage department. |
| Postconditions | Exactly one `Inquiry` row, one append-only `Prediction` row, and one `InquiryEvent(CREATED)` row exist; the inquiry is visible in the correct queue to every principal authorized to see that department. |

**Use Case Specification: Review Prediction (Accept/Correct)**

| Field | Description |
|---|---|
| Primary Actor | Department Officer or Administrator |
| Preconditions | Actor can read the inquiry (`assert_can_read`) and has write access (`can_write`); the inquiry has a `current_prediction`. |
| Main Flow | 1. Officer opens the ticket's Review tab and inspects the predicted category/urgency and confidence. 2. Officer accepts the prediction as correct, or supplies a corrected department and/or urgency. 3. System writes a `ReviewEvent` (action `ACCEPT` or `CORRECT`) tagged against the specific `Prediction` id being reviewed. |
| Alternate Flow | If the officer's chosen department differs from the model's predicted department when routing a Triage ticket, the system records the action as `CORRECT` rather than `ACCEPT` automatically, without the officer having to state which case applies. |
| Postconditions | A `ReviewEvent` row exists that is available, but not yet consumed, as a labelled example for a future feedback-retraining run (`consumed_by_run_id` remains NULL). |

**Use Case Specification: Route Triage Inquiry to Department**

| Field | Description |
|---|---|
| Primary Actor | Department Officer or Administrator |
| Preconditions | The inquiry is currently in the Triage department (`RoutingDecision` is `TRIAGE` or `OUT_OF_SCOPE`); the caller supplies the inquiry's `expected_version`. |
| Main Flow | 1. Officer selects the correct destination department for a ticket the model could not confidently route. 2. System performs an optimistic-concurrency check (`version = expected_version`); on success it sets the department, sets `routing = AUTO`, and recomputes `sla_due_at` from the inquiry's original `created_at` rather than the current time, so the ticket's deadline reflects when it arrived, not when a human got to it. 3. System writes a `ReviewEvent` and an `InquiryEvent(ROUTED)`. |
| Alternate Flow | If another officer routed or otherwise modified the same ticket first, `version` no longer matches `expected_version`; the system rejects the request with HTTP 409 and both the expected and actual version, and the client must refetch and retry rather than silently overwrite the other officer's action. |
| Postconditions | The ticket is visible in its real department's priority-ordered queue with a deadline computed against its true arrival time. |

### 4.4.2 Class Diagram

```
+------------------+        +-------------------+        +---------------------+
|     User         |        |    Department      |        |     SlaPolicy       |
+------------------+        +-------------------+        +---------------------+
| id               |1      *| id                 |1      *| id                  |
| username          |<------| code               |<-------| urgency              |
| role (enum)       |       | name                |        | department_id (FK,  |
| department_id (FK)|------>| is_triage           |        |   nullable=default) |
| password_hash      |      | is_active           |        | response_hours       |
| failed_login_count |      +-------------------+        | warning_fraction     |
| locked_until       |                                    +---------------------+
+------------------+
        |1
        |*
+------------------+        +----------------------+        +----------------------+
|     Session       |       |       Inquiry         |1      *|     Prediction        |
+------------------+        +----------------------+<-------+----------------------+
| id (UUID)          |       | id                    |        | id                    |
| user_id (FK)        |      | reference (unique)     |        | inquiry_id (FK)        |
| token_hash          |      | raw_text / normalized  |        | model_version_id (FK) |
| issued_at / expires |      | department_id (FK)     |        | policy_id (FK)         |
| revoked_at          |      | urgency, status, routing|       | category_label/conf.  |
+------------------+        | assignee_id (FK->User) |        | urgency_label/conf.   |
                              | current_prediction_id  |------>| category/urgency_energy|
                              |   (FK->Prediction)      |       | abstained, latency_ms |
                              | duplicate_of_id (self)  |       +----------------------+
                              | sla_due_at, resolved_at |               |1
                              | version (optimistic)    |               |*
                              +----------------------+        +----------------------+
                               |1        |1         |1        |     Explanation       |
                               |*        |1         |*        +----------------------+
                    +----------------+ +--------+ +----------------+ | prediction_id (FK)|
                    |  ReviewEvent    | |Inquiry | | InquiryEvent    | | task (cat/urgency) |
                    +----------------+ |Embedding| +----------------+ | tokens (JSON)       |
                    | reviewer_id (FK)| +--------+ | event_type (enum)| +----------------------+
                    | action (enum)   |            | actor_id (FK,    |
                    | corrected_dept  |            |   nullable=system)|
                    | corrected_urg.  |            | from_value/to_value|
                    | consumed_by_run |            +----------------+
                    |   _id (FK)      |
                    +----------------+

+----------------------+        +----------------------+        +----------------------+
|     ModelVersion       |1     *|   RoutingPolicy        |       |    TrainingRun         |
+----------------------+<-------+----------------------+       +----------------------+
| id                     |       | id                     |       | id                     |
| name (category/urgency/|       | version (unique)        |       | kind (initial/feedback |
|   multitask)           |       | category_tau/temperature|       |   _retrain/import)     |
| version, backend        |       | urgency_tau/temperature |       | base/produced_model    |
| artifact_uri, checksum  |       | category_energy_threshold|      |   _version_id (FK x2)  |
| label_classes (JSON)    |       | duplicate_threshold      |      | metrics_before/after   |
| is_active               |       | is_active                |      | promoted (bool)        |
+----------------------+       +----------------------+       +----------------------+

+----------------------+
|   OutboundMessage      |
+----------------------+
| id                     |
| inquiry_id (FK)         |
| channel, to_address     |
| status (pending/sent/   |
|   failed)               |
| attempts, scheduled_for |
+----------------------+
```
*Figure 7. Class diagram of the persisted domain model, thirteen tables in total. `Prediction` is
append-only — correcting a prediction never overwrites it, it is recorded as a separate `ReviewEvent`
against the same `Prediction` id — which is what makes the review log usable as a future training signal.*

### 4.4.3 Activity Diagram

```
                     Inquiry Lifecycle Activity Diagram

  [Web form (admin)] or [Email poller]
              |
              v
     Deduplicate (hash + vector similarity) --dup found--> [Link as duplicate, stop]
              |
              v
   Classify (category + urgency, shared/dual transformer)
              |
              v
   Temperature-scale both confidences
              |
              v
      <category confidence >= tau
       AND not out-of-distribution?>
        /                        \
      yes                         no
       |                           |
       v                           v
  Route to predicted           Route to Triage
  department (AUTO)            (TRIAGE / OUT_OF_SCOPE)
       |                           |
       |                    [Officer selects correct
       |                     department] --> ReviewEvent
       |                           |
       +-------------+-------------+
                     |
                     v
        Insert into priority-ordered
        queue (urgency rank, then
        soonest sla_due_at)
                     |
                     v
        [Officer opens ticket] --> Review prediction
                     |             (Accept / Correct) --> ReviewEvent
                     v
        Update status: NEW -> IN_PROGRESS -> ... -> RESOLVED/CLOSED
                     |
              <past sla_due_at
               and unresolved?>--yes--> [SLA sweep job] --> Escalated
                     |no
                     v
                  Resolved
                     |
              <retention_days
               configured and
               past cutoff?>--yes--> [Retention job] --> PII redacted,
                     |no                                  audit trail kept
                     v
                    End
```
*Figure 8. Activity diagram of the inquiry lifecycle from intake to resolution, including the two background
jobs (SLA sweep, retention) that can act on a ticket without a human present.*

### 4.4.4 Sequence Diagrams

**Sequence 1 — Submit Inquiry via Web Form (auto-route path)**

```
Admin        FastAPI(/inquiries)     Classifier      RoutingPolicy       DB
  |  POST inquiry text     |                |                |            |
  |------------------------>|                |                |            |
  |                         | classify(text) |                |            |
  |                         |---------------->|                |            |
  |                         |<--- category/urgency logits -----|            |
  |                         | temperature-scale (policy.temperature)        |
  |                         |----------------------------------->|          |
  |                         | compare conf>=tau, energy<threshold           |
  |                         |----------------------------------->|          |
  |                         | INSERT Inquiry, Prediction, InquiryEvent      |
  |                         |------------------------------------------------>
  |                         |<---------------------------------- committed --|
  |  201 {inquiry, routing} |                |                |            |
  |<------------------------|                |                |            |
  | navigate() to destination queue                                        |
```

**Sequence 2 — Officer Routes a Triage Inquiry**

```
Officer      FastAPI(/inquiries/{id}/route)      DB
  |  POST {department_id, expected_version}       |
  |------------------------------------------------>
  |          UPDATE inquiries SET department_id=..,|
  |          version=expected_version+1            |
  |          WHERE id=.. AND version=expected_version
  |------------------------------------------------>
  |<---------------------- rowcount == 0? ----------|
  |                 yes -> raise ConflictError (409)|
  |                 no  -> INSERT ReviewEvent,       |
  |                        InquiryEvent(ROUTED)      |
  |<-------------------- 200 {inquiry, version} -----|
```

**Sequence 3 — Automated Email Intake**

```
Scheduler      job_lock()      IMAP server      triage.submit()      DB
  |  every email_poll_seconds  |                    |                 |
  |---------------------------->|                    |                 |
  |         acquire lock (skip if another worker holds it)             |
  |----------------------------------------------------->              |
  |                             | FETCH UNSEEN messages                |
  |                             |------------------->                  |
  |                    strip quoted reply/signature                    |
  |                    is_automated_sender(from)? --yes--> skip, mark seen
  |                             |            no          |             |
  |                             |------------------------> submit(text,|
  |                             |                          source=EMAIL,|
  |                             |                          source_ref= |
  |                             |                          Message-ID) |
  |                             |                          (same classify/
  |                             |                           route path as
  |                             |                           Sequence 1)|
  |                             |                          |----------->
  |                             |                          |<-- committed
  |         release lock                                                |
```
*Figures 9-11. The three sequence diagrams that most affect correctness: the routing decision, the
concurrency-safe manual correction, and the unattended intake path — chosen over less consequential flows
(e.g. login) per the template's guidance to keep the highest-value diagrams in the main text.*

### 4.4.5 State Chart — Ticket Queue Status

```
        [NEW] --(officer opens/acts)--> [IN_PROGRESS]
          |                                  |
          |                                  v
          |                       [WAITING_ON_STUDENT] <--\
          |                                  |             |
          |                                  |  (student replies via
          |                                  |   a future channel; not
          |                                  |   automated in this build)
          |                                  |             |
          |                                  \-------------/
          |                                  |
          v                                  v
    (any state) -----------------> [RESOLVED] / [CLOSED]  (terminal)
                                              |
                                  (officer reopens: REOPENED event,
                                   resolved_at cleared)
                                              |
                                              v
                                      [IN_PROGRESS]
```
*Figure 12. `QueueStatus` state chart. `RESOLVED` and `CLOSED` are the only terminal states
(`is_terminal`); every transition, including reopening, is logged as an `InquiryEvent`.*

## 4.5 Proposed System Architecture

```
                    UniFlow System Architecture

  +------------------+        HTTPS (same-origin in         +----------------------+
  |  React SPA        |<-------production, dev-proxy in------>|  FastAPI (Uvicorn)   |
  |  (Vite build,      |        development)                  |  uniflow/api/         |
  |  TanStack Query,    |                                       +----------------------+
  |  React Router)      |                                       | Domain layer:         |
  +------------------+                                        |  triage, workflow,    |
                                                                 |  review, auth, sla    |
                                                                 +----------------------+
                                                                 | ML inference layer:   |
                                                                 |  shared/dual          |
                                                                 |  transformer,         |
  +------------------+                                          |  calibration, energy  |
  |  IMAP mailbox      |<---poll (email_poll_seconds)---+       |  OOD, Integrated      |
  +------------------+                                   \      |  Gradients            |
                                                            \    +----------------------+
  +------------------+                                      \            |
  |  SMTP relay        |<---drain (outbox_drain_seconds)------\-----------+
  +------------------+                                        \          |
                                                                 \        v
                                    +----------------------+     \  +----------------------+
                                    |  APScheduler           |----+->|  SQLAlchemy ORM       |
                                    |  (in-process,           |      |  (13 tables)          |
                                    |  DB-backed job locks,   |      +----------------------+
                                    |  max_instances=1)        |               |
                                    +----------------------+               v
                                                                    +----------------------+
                                                                    |  PostgreSQL            |
                                                                    +----------------------+
```
*Figure 13. System architecture. There is one deployable service (FastAPI serving both the API and the
built React bundle via `StaticFiles`), one database, and no message broker — the scheduler runs in-process
and coordinates across API workers using a database-backed advisory lock rather than a separate queue,
consistent with the minimal-infrastructure decision explained in Chapter 05.*

## 4.6 Functional and Non-Functional Requirements

### Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | The system shall accept a new inquiry through a web form (administrator only) or an automated email poller, producing exactly one `Inquiry` row per genuine message. |
| FR-2 | The system shall classify each inquiry's category and urgency using a shared-or-dual transformer, and shall calibrate both confidence scores using temperature scaling before they are used in any routing decision. |
| FR-3 | The system shall auto-route an inquiry to its predicted department only when the calibrated category confidence is at or above the active policy's τ threshold and the energy-based out-of-distribution score does not flag the inquiry as unfamiliar; otherwise it shall place the inquiry in the shared Triage department. |
| FR-4 | The system shall compute a service-level deadline (`sla_due_at`) for every inquiry from its predicted urgency and the destination department's `SlaPolicy`. |
| FR-5 | The system shall order every department queue and the Triage queue by urgency rank first and soonest deadline second, not by arrival time. |
| FR-6 | The system shall detect duplicate inquiries by exact text hash and by embedding similarity, and shall link a detected duplicate to its original rather than creating an independent ticket. |
| FR-7 | The system shall let an authorized officer or administrator accept or correct a prediction's category and/or urgency, recording the decision as an append-only `ReviewEvent` tied to the specific prediction reviewed. |
| FR-8 | The system shall let an authorized officer or administrator route a Triage inquiry into its correct department, recomputing its deadline from the inquiry's original arrival time rather than the time of routing. |
| FR-9 | The system shall provide a per-token Integrated Gradients explanation for a prediction on request, cached so it is computed once per prediction/task pair. |
| FR-10 | The system shall record every status change, assignment, routing decision, escalation, and resolution of an inquiry as an append-only, timestamped `InquiryEvent`. |
| FR-11 | The email-intake job shall discard messages from recognized automated/no-reply senders before they are classified, and shall not process a message whose `Message-ID` has already been ingested. |
| FR-12 | A background job shall escalate any unresolved, unescalated inquiry once its deadline has passed. |
| FR-13 | The system shall restrict what an authenticated user can read and act on to their own department plus the shared Triage department, unless their role is administrator. |
| FR-14 | The system shall let an administrator promote a `ModelVersion` or activate a `RoutingPolicy`, and shall enforce that at most one version per model slot and at most one policy are active at a time. |
| FR-15 | The system shall issue a short-lived access token and a longer-lived, single-use, rotating refresh token on login, and shall revoke a session immediately on logout or on a refresh-token mismatch. |

### Non-Functional Requirements

| ID | Requirement | How it is satisfied |
|---|---|---|
| NFR-1 | Security — credentials at rest | Passwords are hashed with Argon2id; a legacy PBKDF2-HMAC-SHA256 hash is verified transparently and rehashed to Argon2id on next successful login. |
| NFR-2 | Security — access control | Every read and write is scoped by a `Principal`'s visible department set at the query level (a SQL predicate, not a post-hoc filter), enforced identically whether the caller is an officer, viewer, or administrator. |
| NFR-3 | Security — session integrity | Access tokens expire after 15 minutes; refresh tokens are single-use and rotate on every refresh; a refresh-token secret mismatch revokes the session immediately as a theft signal; five consecutive failed logins lock an account for 15 minutes. |
| NFR-4 | Reliability — unattended jobs | The SLA sweep, outbox drain, email intake, and retention jobs each acquire a database-backed lock before running, so multiple API worker processes never execute the same job concurrently. |
| NFR-5 | Concurrency — safe concurrent edits | Every mutating request on an inquiry is conditioned on an `expected_version`; a stale write is rejected with HTTP 409 rather than silently overwritten, so two officers acting on the same Triage ticket cannot lose each other's changes. |
| NFR-6 | Auditability | Every state-changing action, human or automated, is recorded as an append-only `InquiryEvent`, and every accept/correct decision as an append-only `ReviewEvent`; neither table supports in-place update of a past entry. |
| NFR-7 | Trustworthy automation | An inquiry is only ever auto-routed when both a calibrated-confidence check and an out-of-distribution check pass; the calibration and detection methods and their evaluated performance are reported in Chapters 02 and 06 respectively, not asserted without evidence. |
| NFR-8 | Data minimization | An optional retention job redacts personally identifying fields on resolved inquiries past a configurable age, while preserving the row, its predictions, and its audit trail for continued evaluation. |
| NFR-9 | Maintainability | The system is organized into independently testable layers (API routers, domain logic, repositories, ML inference); the frontend consumes a TypeScript client generated from the API's own OpenAPI schema, so a backend contract change is a type error at build time rather than a runtime surprise. |
| NFR-10 | Operability | The entire system runs as one deployable FastAPI process with an in-process scheduler and no message broker, so it can be operated and reasoned about by a single administrator without separate infrastructure to install or monitor. |

## 4.7 Chapter Summary

This chapter specified the implemented system's requirements by describing the system that was actually
built rather than restating the interim proposal's plan. Its stakeholders were identified, including a
system-actor scheduler and a read-only Viewer role that the interim report did not anticipate; its
requirements were shown to be operationalized from literature, carried-forward stakeholder definitions, and
lessons discovered during implementation rather than a single upfront gathering exercise; and its structure was
presented through a use-case diagram with three detailed specifications, a thirteen-table class diagram, an
activity diagram, three sequence diagrams, a state chart, and the system architecture diagram. The resulting
functional and non-functional requirements are each traceable to a specific database constraint, API
behaviour, or background job rather than stated as an aspiration. The next chapter describes how these
requirements were implemented.
