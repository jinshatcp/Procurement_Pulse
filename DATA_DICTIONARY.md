# Data dictionary

Every input record is a completed, synthetic procurement case. `month` is the completion cohort, not a snapshot of open orders.

| Field | Meaning / unit |
|---|---|
| id | Unique synthetic case ID |
| month | YYYY-MM completion month; Jan–Sep 2026 in simulation |
| supplier | Fictitious Supplier 01–30; excluded from model |
| category | One of four generic procurement categories; excluded from model |
| system | Illustrative TMT / TMS / Oracle source attribution; excluded from model |
| planned_days | Approved cycle duration in calendar days; positive |
| cycle_days | Actual request-to-completion duration in calendar days; positive |
| budget_aed | Approved spend, AED; positive |
| actual_aed | Actual spend, AED |
| variance_pct | 100 × (actual_aed / budget_aed − 1); percentage points |
| change_orders | Count of amendments; nonnegative |
| late_days | Days late; 0 means on time; blank means missing |
| quality_issue | Whether the simulator intentionally removed delivery lateness |
| injected_anomaly | Generator-defined true/false label; evaluation ONLY; never a model input |

Derived outputs:

| Field | Meaning |
|---|---|
| cycle_ratio | cycle_days / planned_days; derived model input, not stored in raw CSV |
| anomaly_score | Negative Isolation Forest score_samples; larger = more unusual |
| risk_percentile | Percentile relative to June calibration anomaly scores, not a probability |
| flagged | Whether score is strictly above the chosen calibration-percentile cutoff |
| imputed_late_days | Model input after filling missing lateness with the training median |
| signals | Human-readable context-rule indicators, not causal/model feature attributions |

KPIs: actual spend = sum(actual_aed); average cycle = unweighted mean(cycle_days); on-time delivery = count(late_days==0) / count(nonmissing late_days); flagged spend = total spend on flagged cases, not financial loss.

V2 adds `model` (method name) and `cohort` (Training / Calibration / Held-out test) to scored outputs. Rules have binary anomaly_score values and no percentile rank. V2 training ends in May; June calibrates cutoffs; July onward is held out.
