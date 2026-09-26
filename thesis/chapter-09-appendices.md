# Appendices

## Appendix A — Dataset Construction and Labelling Guide

This appendix documents how secondary dataset labels were mapped onto the university inquiry categories
and urgency labels used throughout this research, extending the mapping guide proposed in the interim
report with the labels actually applied during dataset construction (Chapter 03, Section 3.4).

| University Category | Indicative Keywords / Phrases | Urgency Mapping Guidance |
|---|---|---|
| Admissions | admission, enrollment, application, offer letter, intake, eligibility | High if a deadline or missing document affects the current intake; otherwise Medium/Low. |
| Examinations | exam, assessment, results, resit, timetable, attendance, paper | High if the exam is today/near a deadline or there is an access issue; Medium for result clarification; Low for general exam information. |
| Fees and Payments | fee, payment, invoice, receipt, installment, refund | High if payment blocks registration/exam; Medium if a receipt confirmation is needed; Low for a general fee inquiry. |
| Timetable and Scheduling | schedule, lecture, class, calendar, room, postponement | Medium if a current-week schedule conflict; Low if a general future schedule request. |
| Academic Documents | transcript, certificate, letter, recommendation, document | High if the document is required before an immediate deadline; Medium for normal processing; Low for general information. |
| Learning Platform / IT Access | login, LMS, password, portal, account, email | High if access blocks an exam/submission; Medium if access affects ongoing coursework; Low for general account guidance. |
| General Administration | contact, office hours, general policy, campus facility | Mostly Low unless deadline or impact words are present. |

Public secondary sources reviewed for suitability against this schema, as discussed in Chapter 02:

| Dataset / Source | Repository | Description | Usefulness | Limitations / Handling |
|---|---|---|---|---|
| Smart University Query Classification Dataset | Kaggle | University-related student queries with labelled categories/intents. | High domain relevance for inquiry category classification. | Required additional urgency labelling and quality review before use. |
| Dataset of Computer Science Course Queries from Students | Mendeley Data / Data in Brief | Student course questions categorized and scored according to Bloom's taxonomy. | Useful for academic query language and student question style. | Not an administrative urgency dataset; required mapping/filtering onto the project's category and urgency schema. |
| Customer IT Support - Ticket Dataset | Kaggle | Multilingual customer support tickets with fields including issue/category and priority. | Useful for urgency/priority modelling and ticket-style text. | Non-educational domain; required relabelling to university categories. |
| Classification of IT Support Tickets | Zenodo | 2,229 manually classified support tickets in multiple languages with PII removed. | Useful for testing text-classification baselines and anonymization practice. | IT-specific categories; no direct university labels. |

Rows and phrasing patterns that these public sources did not adequately cover for a given category, urgency,
and scenario combination were supplemented with author-constructed inquiries written against the same
schema, so that every category/urgency pair in the final corpus has a meaningful number of examples rather
than being left sparse because no public source happened to cover it.

## Appendix B — Data Dictionary

This is the field structure actually used in the constructed corpus (`Query_ID`, `Student_Query`,
`Urgency_Label`, `University_Category`, `Scenario_Family`, `Primary_Intent`), extending the interim report's
proposed data dictionary with the two fields — `Scenario_Family` and `Primary_Intent` — that were added
during construction to support scenario-level error analysis (Chapter 03, Section 3.4).

| Field Name | Data Type | Description |
|---|---|---|
| Query_ID | String | Unique identifier for each inquiry row; a suffix (e.g. `_A`) marks a phrasing variant generated from the same underlying scenario. |
| Student_Query | Text | The inquiry text, as a student or applicant might plausibly phrase it. |
| Urgency_Label | Categorical | Ground-truth urgency: High, Medium, or Low. |
| University_Category | Categorical | Ground-truth category: Admissions, Examinations, Fees and Payments, Timetable and Scheduling, Academic Documents, Learning Platform / IT Access, or General Administration. |
| Scenario_Family | String | Short code (e.g. `DOC05A`, `TIM01B`) grouping phrasing variants generated from the same underlying situation. |
| Primary_Intent | String | Short human-readable label for the specific situation represented (e.g. "digital certificate", "class timetable clash"), finer-grained than category. |

At runtime, the deployed system's own schema (Chapter 04) additionally records `model_category_prediction`,
`model_urgency_prediction`, per-prediction confidence scores (pre- and post-calibration), the energy
out-of-distribution score, and `human_review_status` (accepted, corrected, or pending), so that every routing
decision is traceable back to the model output that produced it and the reviewer decision that followed it.

## Appendix C — Evaluation Plan

This is the evaluation plan applied in Chapter 06, extending the interim report's classification-metrics plan
with the calibration and selective-prediction metrics that Chapter 02 identifies as necessary once a classifier's
confidence score is used to gate an unsupervised routing decision.

| Evaluation Item | Meaning | Reason for Use |
|---|---|---|
| Accuracy | Overall proportion of correct predictions. | Useful overview, but insufficient for imbalanced urgency labels. |
| Precision | Correct positive predictions divided by all positive predictions. | Important to avoid too many false urgent predictions. |
| Recall | Correct positive predictions divided by all actual positives. | Critical for High urgency so urgent inquiries are not missed. |
| Macro F1-score | Harmonic mean of precision and recall, averaged per class. | Balanced performance measure; robust to class imbalance. |
| Confusion Matrix | Table of predicted versus actual labels. | Shows which categories or urgency levels are confused. |
| McNemar's exact test / bootstrap CI | Paired significance test and confidence interval on baseline-versus-transformer disagreement. | Confirms an accuracy improvement is statistically supported rather than a point-estimate artifact. |
| Expected Calibration Error | Weighted gap between confidence and empirical accuracy across confidence bins. | Quantifies whether temperature scaling corrected overconfidence (Chapter 02). |
| Risk-coverage curve / AURC | Selective risk as a function of the fraction of inquiries auto-routed. | Quantifies the accuracy-versus-automation trade-off the τ threshold makes explicit. |
| Energy-score AUROC | Separability of in-distribution and out-of-distribution inquiries by energy score. | Validates the out-of-distribution gate that supplements the τ threshold. |
| Inference latency / cost | Forward-pass latency and parameter count on the deployment hardware profile. | Confirms the model is practical to serve inside the request path of the FastAPI service. |
| Error Analysis | Manual review of wrong predictions, grouped by Scenario_Family. | Helps identify whether errors cluster around specific real-world situations rather than being uniformly distributed. |

## Appendix D — Full Automated Test Suite (43 tests)

Ten of these are presented as full test cases in Chapter 06, Section 6.2; the remainder are listed here by
area, grouped as they appear in `tests/test_uniflow.py`.

**SLA computation (6)**: `test_clean_text_normalises_quotes_and_whitespace`,
`test_sla_due_dates_follow_urgency`, `test_sla_state_transitions`,
`test_resolved_ticket_never_becomes_breached_by_the_passage_of_time`,
`test_breached_filter_matches_only_unresolved_overdue_tickets`,
`test_sla_compliance_counts_only_closed_tickets`.

**Authentication and access control (6)**: `test_password_hashing_roundtrip`,
`test_legacy_pbkdf2_hash_still_verifies_and_flags_for_rehash`,
`test_login_succeeds_and_rejects_wrong_credentials`,
`test_login_locks_the_account_after_repeated_failures`,
`test_visible_departments_scoped_to_own_plus_triage`, `test_can_read_denies_other_departments`.

**Database, workflow, and the review log (8)**: `test_new_inquiry_gets_sane_defaults`,
`test_update_status_stamps_response_and_resolution`,
`test_stale_write_is_rejected_by_optimistic_locking`,
`test_review_events_recorded_and_consumable`, `test_correction_rate`,
`test_route_from_triage_logs_a_correction`, `test_email_source_ref_uniqueness_is_enforced`,
`test_create_all_is_idempotent`.

**Calibration and selective prediction (9)**: `test_temperature_scaling_preserves_predictions`,
`test_temperature_above_one_reduces_confidence`, `test_fit_temperature_recovers_a_known_distortion`,
`test_fit_temperature_clamps_the_degenerate_perfect_case`, `test_expected_calibration_error_bounds`,
`test_risk_coverage_curve_is_monotonic_in_coverage`, `test_choose_tau_respects_the_target_risk`,
`test_choose_tau_reports_failure_when_the_target_is_unreachable`,
`test_evaluate_at_tau_handles_total_abstention`.

**Out-of-distribution detection and significance testing (5)**: `test_energy_is_lower_for_confident_logits`,
`test_energy_uses_magnitude_not_just_shape`, `test_mcnemar_detects_a_clear_difference`,
`test_mcnemar_reports_no_difference_for_identical_systems`,
`test_bootstrap_interval_contains_the_point_estimate`.

**Dataset loading and email intake (6)**: `test_column_aliases_are_resolved`,
`test_supplied_split_is_honoured_without_a_leakage_warning`,
`test_generated_split_raises_the_leakage_flag`, `test_duplicate_and_empty_rows_are_dropped`,
`test_quoted_history_and_signature_are_stripped`, `test_automated_senders_are_not_treated_as_students`.

**Routing policy configuration (3)**: `test_thresholds_fall_back_when_the_policy_file_is_absent`,
`test_corrupt_policy_file_does_not_break_startup`, `test_partial_policy_file_is_merged_over_defaults`.
