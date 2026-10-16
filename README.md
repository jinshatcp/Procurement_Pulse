# Procurement Pulse V2 — Python ML Workbench

Open **START_HERE.html** for the visual setup guide. This package contains runnable Python, a local dashboard, a self-contained Colab notebook, synthetic data, trained models and reproducible outputs. Every record is simulated. This is independent academic demonstration software, not an official organizational product.

## What is better in V2?

| Previous package | V2 |
|---|---|
| Basic monitoring interface | Six views with interactive Plotly charts, review notes and export controls |
| One ML model | Isolation Forest and One-Class SVM, plus transparent context rules |
| Cutoff derived from training data | Separate chronological training, calibration and held-out test cohorts |
| Scoring through the terminal | Individual case form, CSV scoring and terminal prediction |
| Command-only setup | Visual guide, Windows setup/launch scripts and Colab notebook |
| Basic ROI example | Reporting-effort scenario including estimated alert-review costs |
| Core workflow tested | Core workflow and Streamlit interface interactions tested |

## Windows: simplest route

1. Install **Python 3.12** (or 3.11) from https://www.python.org/downloads/. Select **Add Python to PATH**.
2. Extract the ZIP. Enter the `Procurement_Pulse_V2` folder containing this README.
3. Double-click **SETUP_WINDOWS.bat**. It creates a local virtual environment, installs pinned dependencies and trains the models. Keep internet available for installation. If Windows blocks the script, open Command Prompt in this folder and use the manual commands below.
4. Double-click **START_DASHBOARD.bat**. Open the Local URL printed in the terminal, normally http://localhost:8501. Keep the terminal open; press Ctrl+C to stop the app.
5. Next time, use **START_DASHBOARD.bat** only. Setup and retraining are not required each time.

No GPU, API key, paid cloud account or database is required. VS Code is optional.

### Manual setup (Command Prompt)

Open File Explorer in this project folder, type `cmd` into the address bar and press Enter. Run one line at a time:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python run.py all
python -m streamlit run dashboard.py
```

Use `py -3.11` if that is the installed version. If `py` is unavailable but `python --version` shows 3.11 or 3.12, use `python -m venv .venv`. PowerShell users can use Command Prompt to avoid activation-policy issues.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py all
python -m streamlit run dashboard.py
```

## Google Colab: no local installation

Go to https://colab.research.google.com/. Choose **File → Upload notebook** and upload `Procurement_Pulse_V2.ipynb`. Choose **Runtime → Run all**. The notebook embeds the core Python code, simulates the dataset, trains both models, compares results, shows figures and downloads a results ZIP. No other uploads are needed. The local Streamlit dashboard uses the Windows/macOS/Linux route, not the notebook.

## The six dashboard sections

1. **Executive overview:** spend, average cycle, on-time delivery, anomaly counts, monthly trends, category spend and supplier comparison.
2. **Review queue:** searchable exceptions, context signals, case inspection, review notes, reviewed/reopened status and downloadable review records. Notes exist in the current Streamlit session only.
3. **Model lab:** two ML models and the rules baseline, precision/recall/F1, confusion matrices and score distributions. Global filters narrow the held-out test cohort. The sensitivity slider is explicitly exploratory.
4. **Case lab:** create a hypothetical case or upload a matching CSV; score it with the fitted model without retraining. Download an unlabeled input template or resulting scores.
5. **Data quality:** raw missing values, input validation rules, completeness, synthetic source attribution and KPI definitions.
6. **Impact simulator:** editable reporting effort, staff rate, review workload, setup/recurring costs, ROI and payback. Independent of portfolio filters; uses the chosen detector's fixed default synthetic test alert rate for workload.

## How to run the machine learning separately

```bash
python run.py simulate
python run.py train
python run.py predict --input data/new_cases_example.csv
```

Or run `python run.py all` to generate the synthetic data and train/evaluate both detectors in one step. **all** overwrites the selected input path with simulated data; use **train** when working with an existing approved input CSV.

Select a different model for prediction:

```bash
python run.py predict --input data/new_cases_example.csv --model one_class_svm --output predictions_svm
python run.py predict --input data/new_cases_example.csv --model rules --output predictions_rules
```

Prediction defaults to the saved cutoff. An optional `--percentile 95` changes the calibration cutoff for ML detectors. Rules always use fixed limits and ignore the percentile. Predictions are scored, not used to update the models.

Train an approved dataset with suitable cohort dates:

```bash
python run.py train --input path/to/approved_cases.csv --train-end 2026-05 --calibration-end 2026-06 --output approved_outputs
```

Then score future cases using that trained bundle:

```bash
python run.py predict --input path/to/new_cases.csv --bundle approved_outputs/model_bundle.joblib --output approved_predictions
```

This requires the documented schema, units and validation. The bundled dashboard intentionally reads the default synthetic training data. Uploaded Case lab CSVs are scored using that synthetic model, which has not been validated for real organizational use. Do not infer real predictive quality from those scores.

## The ML workflow explained

- **Synthetic input:** 1,200 completed cases across Jan–Sep 2026, seed 8920. About 8% receive injected unusual behavior; 38 late-day values are missing.
- **Feature engineering:** actual cycle / planned cycle, spend variance %, change-order count and late days. Identity, supplier, category, source and injected truth are excluded.
- **Training:** Jan–May, **670 cases**. Fit the median imputer, RobustScaler (SVM only), Isolation Forest (250 trees, seed 8920) and One-Class SVM (RBF, gamma=scale, nu=0.08). Both learn without labels.
- **Calibration:** June, **134 cases**. Score the calibration data with the fitted detectors. Use the **90th percentile** of each detector's calibration scores as a preset cutoff. No labels choose the cutoff or model settings.
- **Held-out test:** Jul–Sep, **396 cases**. Test labels are used only for evaluation. Do not select a winner, tune hyperparameters or repeatedly change cutoffs using final-test results and then describe them as unbiased final results.
- **Rules baseline:** alert if cycle ratio > 1.35 OR variance > 10% OR amendments >= 3 OR imputed lateness > 10 days. Fixed limits mean the baseline may generate a different alert volume.
- **Scoring:** negate `score_samples`; larger means more unusual. A score strictly above the calibration cutoff creates a flag. Calibration rank is an empirical percentile, not a probability.
- **Human review:** context signals are visible rules, not SHAP values, learned feature importance or causal explanations. A flag does not prove a procurement issue.

## Computed default benchmark (synthetic only)

| Held-out result | Isolation Forest | One-Class SVM | Context rules |
|---|---:|---:|---:|
| Cases | 396 | 396 | 396 |
| True positives | 37 | 21 | 38 |
| False positives | 0 | 10 | 5 |
| False negatives | 1 | 17 | 0 |
| True negatives | 358 | 348 | 353 |
| Alerts | 37 | 31 | 43 |
| Precision | 100.0% | 67.7% | 88.4% |
| Recall | 97.4% | 55.3% | 100.0% |
| F1 | 98.7% | 60.9% | 93.8% |

**These numbers cannot establish real-world performance.** The injected anomaly patterns are intentionally simple. The default changed from V1: training ends in May, June is reserved for calibration, the default percentile is 90 and the forest has 250 trees. Therefore V1 and V2 results are not a controlled comparison of algorithm quality. V2 was not tuned by optimizing the test set. Operational model selection requires approved data, expert-reviewed labels, consistent workload constraints and an untouched final test.

## Project files

| File | Purpose |
|---|---|
| START_HERE.html | Visual setup and workflow guide; open directly in a browser |
| SETUP_WINDOWS.bat | Create environment, install dependencies and train |
| START_DASHBOARD.bat | Launch the local app using that environment |
| core.py | Simulation, input checks, feature engineering, training, scoring, evaluation, figures |
| run.py | Command-line workflow |
| dashboard.py | Streamlit dashboard and case-scoring UI |
| charts.py | Interactive Plotly chart functions |
| Procurement_Pulse_V2.ipynb | Self-contained Colab notebook |
| requirements.txt | Pinned Python dependencies; Python 3.11/3.12 recommended |
| DATA_DICTIONARY.md | Field definitions and model/KPI units |
| MODEL_CARD.md | Model design, limitations and intended use |
| data/synthetic_procurement.csv | Generated raw records and synthetic evaluation labels |
| data/new_cases_example.csv | Five unlabeled example cases matching the input schema |
| outputs/model_bundle.joblib | Both fitted models, imputer/scaler, calibration distributions and settings |
| outputs/evaluation.json | Fixed default held-out metrics and split details |
| outputs/scored_procurement.csv | Default Isolation Forest scores for all cases |
| outputs/all_model_scores.csv | All cases scored by all three methods |
| outputs/model_comparison.csv | Compact held-out comparison table |
| outputs/cycle_time.png | Generated cycle-time figure |
| outputs/model_comparison.png | Generated confusion-matrix figure |
| outputs/environment.json | Versions that generated the shipped results |
| tests/test_pipeline.py | Reproducibility, leakage, persistence and input-validation checks |
| tests/test_app.py | Streamlit interaction checks; no browser required |

The notebook and CLI can regenerate model files; only load trusted joblib bundles, because deserialization can execute code. Model persistence is version-sensitive. If the shipped bundle is incompatible with your environment, run `python run.py all` using the pinned dependencies to regenerate it.

## Verification

```bash
python -m unittest discover -s tests -p "test_pipeline.py"
python tests/test_app.py
```

Core calculations and notebook computational cells were executed. Streamlit AppTest verified six views, model switching, threshold adjustment, source/cohort filtering, the individual-case form, and reviewed/reopen status. Plotly figures are generated and their data are tested; automated browser screenshot review was not performed. The Windows .bat commands are provided for Windows and were not executed on this Linux environment.

## Troubleshooting

- **Python not found:** install Python 3.11/3.12, enable PATH and reopen the terminal.
- **Package missing:** activate `.venv`, then rerun `python -m pip install -r requirements.txt`.
- **Model missing or incompatible:** run `python run.py all` in the project environment.
- **Port busy:** `python -m streamlit run dashboard.py --server.port 8502`.
- **CSV rejected:** match the example header and dictionary; do not use currency symbols or thousands separators in numeric fields.
- **No test records in Model lab:** select All periods or a Jul–Sep cohort.
- **Review note lost:** notes are session-only; download JSON before closing the app.
- **No payback / negative ROI:** your assumed costs exceed benefits; change inputs only if justified.
- **Browser does not open:** copy the Local URL printed in the terminal.

## Academic and operational boundaries

The supplied project proposal concerns procurement performance monitoring, readiness, trends, anomalies and human judgment. This prototype supports those mechanics. It does not integrate systems, recommend awards, change procurement actions or establish organizational savings. Synthetic data do not replace any requirement for real secondary-data evidence. Disclose AI assistance and follow the supplied assessment restrictions on submitted writing. No live benchmarks or claims about actual Tawazun performance are embedded.

## Official technical documentation

- Isolation Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- One-Class SVM: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
- Model persistence: https://scikit-learn.org/stable/model_persistence.html
- Streamlit installation: https://docs.streamlit.io/get-started/installation/command-line
- Streamlit app testing: https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest
