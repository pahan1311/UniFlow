# UniFlow evaluation report

Generated 2026-08-29 18:29.

## 1. Classification performance

| Task | System | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| category | majority | 0.1429 | 0.0357 | 0.0357 |
| category | tfidf_lr | 1.0000 | 1.0000 | 1.0000 |
| category | tfidf_svm | 1.0000 | 1.0000 | 1.0000 |
| category | **dual transformer** | 0.9000 | **0.9002** | 0.8974 |
| category | _(transformer macro-F1 95% CI)_ | | [0.8074, 0.9638] | |
| urgency | majority | 0.4429 | 0.2046 | 0.2719 |
| urgency | tfidf_lr | 1.0000 | 1.0000 | 1.0000 |
| urgency | tfidf_svm | 1.0000 | 1.0000 | 1.0000 |
| urgency | **dual transformer** | 0.6857 | **0.6574** | 0.6745 |
| urgency | _(transformer macro-F1 95% CI)_ | | [0.5214, 0.7740] | |

## 2. Significance (McNemar exact test)

| Task | Baseline | Transformer only correct (b) | Baseline only correct (c) | p | Verdict |
|---|---|---|---|---|---|
| category | majority | 53 | 0 | 2.22e-16 | significant |
| category | tfidf_lr | 0 | 7 | 0.01562 | significant |
| category | tfidf_svm | 0 | 7 | 0.01562 | significant |
| urgency | majority | 20 | 3 | 0.0004883 | significant |
| urgency | tfidf_lr | 0 | 22 | 4.768e-07 | significant |
| urgency | tfidf_svm | 0 | 22 | 4.768e-07 | significant |

## 3. Confidence calibration (temperature scaling)

| Task | T | ECE before | ECE after | NLL before | NLL after | Mean conf. before | after |
|---|---|---|---|---|---|---|---|
| category | 1.114 | 0.0714 | 0.0443 | 0.3301 | 0.3187 | 0.946 | 0.934 |
| urgency | 3.335 | 0.2956 | 0.0752 | 1.4896 | 0.7334 | 0.958 | 0.747 |

Accuracy is unchanged by temperature scaling by construction; only the reported confidences move. See `reliability_category.png` and `reliability_urgency.png`.

## 4. Selective routing (risk–coverage)

| Task | τ | Coverage | Selective risk | Risk if fully automated | Misroutes avoided | AURC |
|---|---|---|---|---|---|---|
| category | 0.564 | 95.7% | 5.97% | 10.00% | 3 | 0.0081 |
| urgency | 0.817 | 21.4% | 26.67% | 31.43% | 18 | 0.2633 |

τ is selected on the validation split against a target selective risk and then applied unchanged to the test split.

## 5. Out-of-scope detection (energy score)

- AUROC: **0.9029**
- FPR at 95% TPR: 0.6500
- Energy threshold: -3.8952
- Out-of-scope messages caught: 75.0%
- Genuine inquiries diverted to Triage: 12.9%
- Probe set: 20 items (**synthetic** — replace with real out-of-scope mail)

## 6. Inference cost

| Backend | Params (M) | Disk (MB) | Forward passes | Median latency (ms) | p95 (ms) |
|---|---|---|---|---|---|
| dual | 176.44 | 674.5 | 2 | 105.8 | 195.7 |

Measured on cpu, 4 threads, Windows-11-10.0.26200-SP0.

## Figures

All figures are in `evaluation/results/`:

- `confusion_category.png`, `confusion_urgency.png`
- `reliability_category.png`, `reliability_urgency.png`
- `risk_coverage_category.png`, `risk_coverage_urgency.png`
- `ood_energy.png`
