# FYP Poster Content (Final Draft)

## Poster Title
Closing the Label Gap: A Practical Benchmark of Semi-Supervised Learning on CIFAR-10

## Subtitle (optional)
Comparing Supervised, Pseudo-Label, FixMatch, MixMatch, and FlexMatch under low-label budgets

## Section 1: Problem and Aim
Modern image classifiers require large labeled datasets, but labeling is costly.
This project asks: can semi-supervised learning (SSL) recover strong performance when labels are scarce by using unlabeled data effectively?

We benchmarked four SSL methods against a supervised baseline using a unified training and evaluation pipeline.

## Section 2: Experimental Setup
- Dataset: CIFAR-10 (50,000 train, 10,000 test)
- Methods: Supervised, Pseudo-Label, FixMatch, MixMatch, FlexMatch
- Label budgets: 250 (0.5%), 1000 (2%), 4000 (8%)
- Protocol: consistent model family, data pipeline, and evaluation framework across methods
- Metrics:
	- Primary: Top-1 test accuracy (%)
	- Secondary: calibration behavior (ECE) and convergence stability

## Section 3: Main Results
Best test accuracy (%) by method and label budget:

| Method | 250 labels | 1000 labels | 4000 labels |
|---|---:|---:|---:|
| Supervised | 41.50 | 62.65 | 81.66 |
| Pseudo-Label | 84.72 | 89.12 | 91.22 |
| FixMatch | 84.60 | 79.39 | 90.50 |
| FlexMatch | 81.62 | 81.86 | 89.92 |
| MixMatch | 70.77 | 78.74 | 91.53 |

Key trends (single-seed runs; not directly comparable to heavily tuned published benchmarks):
- 250 labels: SSL gives the largest gains over supervised (+29 to +43 points). Best single-run results are Pseudo-Label 84.72% and FixMatch 84.60%.
- 1000 labels: Pseudo-Label is strongest (89.12%). FixMatch is lower than its 250-label run (79.39%), likely impacted by early-stop/stop-loss settings (rerun recommended).
- 4000 labels: methods converge into a tighter band (89.92–91.53%), with a smaller but consistent SSL advantage over supervised.
- Published papers often report higher headline accuracy under stricter protocol control and broader tuning; these results are intended as a reproducible local baseline rather than a SOTA claim.

## Section 4: Interpretation
- SSL provides the biggest gains when labels are most scarce.
- Method ranking changes with budget, so there is no universal winner.
- As labels increase, the SSL-supervised gap narrows.
- Accuracy gains should be interpreted with calibration and stability, not as standalone numbers.

## Section 5: Conclusion and Impact
This benchmark shows that SSL can dramatically reduce labeled-data requirements on CIFAR-10 while retaining strong performance.

Practical takeaway:
- Under extreme scarcity (250 labels), SSL is essential.
- At moderate budgets (1000 labels), method choice materially affects outcomes.
- At higher budgets (4000 labels), methods converge to a tighter performance band.

## Section 6: Limitations and Future Work
- Single-seed runs limit statistical confidence (no confidence intervals)
- Additional threshold and augmentation ablations are needed
- Future validation should include datasets beyond CIFAR-10

## Figure Placement (for template mapping)
- Main figure: poster/figures/accuracy_by_budget.png
- Supporting figure: poster/figures/improvement_over_supervised.png
- Recommended layout:
	- Left column: Problem + Setup
	- Center column: Main results figure + compact table
	- Right column: Interpretation + Conclusion + Future work

## Acknowledgements
- Supervisor and project committee
- Institutional compute support
