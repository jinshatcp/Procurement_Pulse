# Model card — Procurement Pulse V2

**Version:** 2.0. **Purpose:** a synthetic, reproducible academic prototype for procurement monitoring and human exception review. **Not intended for:** supplier selection, procurement approvals, fraud allegations, operational risk probabilities or claims of organizational effectiveness.

## Data and targets

1,200 simulated completed cases, Jan–Sep 2026, 30 fictitious suppliers, four generic categories, fixed seed 8920. Injected anomalies alter cycle time, delivery lateness, spend variance and/or amendment count. Approximately 3.5% of lateness values are intentionally missing. Labels are generator-defined rather than independently verified issues.

Input features: actual/planned cycle ratio, spend variance %, amendment count and late days. Labels and identifying/grouping fields never enter training. No sensitive organizational data are used.

## Training and evaluation

Training: Jan–May (670). Calibration: June (134). Test: Jul–Sep (396). The imputer and SVM RobustScaler fit only training data. Calibration distributions determine preset 90th-percentile cutoffs. No labels choose cutoffs or hyperparameters.

Isolation Forest uses 250 trees and random state 8920. One-Class SVM uses an RBF kernel, gamma=scale and nu=.08, with scaling. These detectors capture different notions of unusualness. Context rules use fixed OR limits rather than a matched alert budget. Report both accuracy-independent classification metrics and alert workload; do not infer superiority from F1 alone.

Larger negative score_samples means more unusual. Calibration percentile is a rank, not a probability. No automatic winner selection is implemented. Global filters can produce small, unrepresentative evaluation subsets. Sensitivity controls expose trade-offs, but test-set exploration compromises its role as an untouched final benchmark.

## Explanations

Displayed context signals use cycle ratio >1.35, spend variance >10%, amendments >=3 or imputed lateness >10 days. They are not learned feature importance, SHAP, causal explanations or proof of wrongdoing. An unflagged case can still exceed a context limit; a flagged case can reflect a combination with no individual limit exceeded.

## Limitations

- Deliberately simple injected patterns make strong scores easier to obtain than in real procurement data.
- Calibration has only 134 cases and may contain anomalies; percentile estimates can be unstable.
- No actual source joins, access controls, production audit trail or live data latency were evaluated.
- Rare issues, category effects, distribution shifts and incorrect source fields may produce false alerts or missed cases.
- Supplier/category/source are excluded from features, but no fairness or subgroup adequacy claim follows.
- Imputed late days may obscure genuine data-quality problems. Operational KPIs retain missingness.
- The impact scenario expresses capacity value under assumptions, not observed or guaranteed cash savings.

## Before operational use

Obtain approved fields and access, reconcile sources, define reliable denominators, gather expert-reviewed labels, compare models and rules on the same workload constraints, set thresholds using validation data, test once on an untouched later cohort, assess subgroup errors, train reviewers and monitor drift. Implement appropriate security and audit controls with organizational owners.

## Persistence

The local trusted model bundle is serialized using joblib. Do not load untrusted bundles. Record dependency versions, dataset lineage and split dates. Regenerate the models under pinned dependencies if the supplied bundle is incompatible. The Case lab CSV path scores records only; it never retrains models.
