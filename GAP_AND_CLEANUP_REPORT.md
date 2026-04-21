# Gap and Cleanup Report

## Snapshot

- Scope source: project.txt
- Current implementation: Supervised, Pseudo-Label, FixMatch, MixMatch, FlexMatch
- Active rerun queue: run_recovery_and_sweep_queue.bat

## Required vs Current (project.txt alignment)

1. Algorithm integration
- Required core: supervised + pseudo-label + mixmatch + fixmatch
- Current: complete
- Additional implemented: flexmatch
- Deferred from plan: temporal ensembling, mean teacher, UDA, FreeMatch, SoftMatch, SimMatch

2. Label budget coverage
- Required practical budgets in current benchmark protocol: 250, 1000, 4000
- Current pre-rerun: complete but quality issues in FixMatch-250 and MixMatch-250
- In progress: FixMatch-250 rerun + MixMatch-250 rerun + MixMatch-1000 rerun

3. Metrics and reports
- Implemented: Top-1 accuracy, per-class accuracy, confusion matrices, ECE/reliability, training time
- Implemented recently: GPU memory used + GPU utilization logging and summary export
- Partial/missing: precision/recall/F1 in consolidated summary, multi-seed CI/t-tests

4. Comparative analysis outputs
- Implemented: supervised-vs-ssl comparison artifacts, benchmark summary matrix
- In progress: rerun/sweep comparison queue and summary refresh

## Pseudo-Label Threshold Tweaking

To address confidence-threshold sensitivity discussed in project.txt and SSL literature:
- Baseline already present: confidence_threshold=0.95
- Added sweep configs:
  - ssl_pseudolabel_cifar10_250labels_conf90.yaml
  - ssl_pseudolabel_cifar10_250labels_conf80.yaml
- These runs are in the recovery+sweep queue and will be compared against supervised-250.

## Published Benchmark Comparison Status

- Added report artifact: experiments/final_report_assets/published_vs_local_cifar10_ssl.md
- Current references used:
  - TorchSSL published CIFAR-10 benchmark table
  - MixMatch OpenReview statement for 250 labels
- Remaining action: refresh this artifact after reruns finish to include rerun deltas.

## Cleanup Actions

No destructive cleanup was executed automatically for safety.

Candidates to remove after final thesis freeze:
- One-off helper queue scripts no longer needed for reruns.
- Outdated benchmark_summary folders that are superseded by the latest timestamp.

Recommended safe cleanup policy:
1. Keep latest two benchmark_summary folders.
2. Keep scripts used in final reproducibility appendix.
3. Archive obsolete helpers under experiments/archive/scripts_old/ instead of deleting.
