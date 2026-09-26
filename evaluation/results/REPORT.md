# UniFlow evaluation report

Generated 2026-09-02 09:27.

## 1. Classification performance

| Task | System | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| category | majority | 0.0764 | 0.0203 | 0.0108 |
| category | tfidf_lr | 0.9852 | 0.9865 | 0.9851 |
| category | tfidf_svm | 0.9926 | 0.9924 | 0.9926 |
| category | **dual transformer** | 0.9936 | **0.9906** | 0.9935 |
| category | _(transformer macro-F1 95% CI)_ | | [0.9824, 0.9969] | |
| urgency | majority | 0.3319 | 0.1661 | 0.1654 |
| urgency | tfidf_lr | 0.9459 | 0.9451 | 0.9451 |
| urgency | tfidf_svm | 0.9173 | 0.9152 | 0.9153 |
| urgency | **dual transformer** | 0.9809 | **0.9808** | 0.9808 |
| urgency | _(transformer macro-F1 95% CI)_ | | [0.9714, 0.9894] | |

## 2. Significance (McNemar exact test)

| Task | Baseline | Transformer only correct (b) | Baseline only correct (c) | p | Verdict |
|---|---|---|---|---|---|
| category | majority | 871 | 6 | 1.241e-249 | significant |
| category | tfidf_lr | 14 | 6 | 0.1153 | not significant |
| category | tfidf_svm | 7 | 6 | 1 | not significant |
| urgency | majority | 627 | 15 | 9.445e-164 | significant |
| urgency | tfidf_lr | 39 | 6 | 5.418e-07 | significant |
| urgency | tfidf_svm | 65 | 5 | 2.215e-14 | significant |

## 3. Confidence calibration (temperature scaling)

| Task | T | ECE before | ECE after | NLL before | NLL after | Mean conf. before | after |
|---|---|---|---|---|---|---|---|
| category | 0.630 | 0.0119 | 0.0052 | 0.0284 | 0.0284 | 0.989 | 0.998 |
| urgency | 1.569 | 0.0174 | 0.0140 | 0.0806 | 0.0720 | 0.991 | 0.975 |

Accuracy is unchanged by temperature scaling by construction; only the reported confidences move. See `reliability_category.png` and `reliability_urgency.png`.

## 4. Selective routing (risk–coverage)

| Task | τ | Coverage | Selective risk | Risk if fully automated | Misroutes avoided | AURC |
|---|---|---|---|---|---|---|
| category | 0.554 | 100.0% | 0.64% | 0.64% | 0 | 0.0000 |
| urgency | 0.521 | 99.8% | 1.91% | 1.91% | 0 | 0.0014 |

τ is selected on the validation split against a target selective risk and then applied unchanged to the test split.

## 5. Out-of-scope detection (energy score)

- AUROC: **0.9898**
- FPR at 95% TPR: 0.0000
- Energy threshold: -5.1496
- Out-of-scope messages caught: 100.0%
- Genuine inquiries diverted to Triage: 4.5%
- Probe set: 20 items (**synthetic** — replace with real out-of-scope mail)

## 6. Inference cost

| Backend | Params (M) | Disk (MB) | Forward passes | Median latency (ms) | p95 (ms) |
|---|---|---|---|---|---|
| dual | 176.44 | 674.5 | 2 | 364.4 | 466.1 |

Measured on cpu, 4 threads, Windows-11-10.0.26200-SP0.

## Figures

All figures are in `evaluation/results/`:

- `confusion_category.png`, `confusion_urgency.png`
- `reliability_category.png`, `reliability_urgency.png`
- `risk_coverage_category.png`, `risk_coverage_urgency.png`
- `ood_energy.png`
