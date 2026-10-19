# Formula 1 Podium Prediction — ACL Milestone 1

**Course:** CSEN 903 — Advanced Computer Lab, German University in Cairo  
**Task:** Predict whether a driver will finish **on the podium (positions 1–3)** using only information available **after qualifying and before the race begins**.

This repository contains the completed Jupyter notebook and the matching analytical report. The work covers data auditing and cleaning, relational joins, historical analysis, leakage-safe features, machine-learning experiments, model explanations, and inference.

## Repository contents

| File | Purpose |
| --- | --- |
| [`acl-project-milestone-1.ipynb`](67-acl-67.ipynb) | Main executable notebook, including code, charts, and saved results. |
| [`ACL_Milestone1_F1_Analytical_Report.pdf`](ACL_Milestone1_F1_Analytical_Report.pdf) | Accompanying 19-page report with methods, figures, answers to all three data-engineering questions, and limitations. |
| [`requirements.txt`](requirements.txt) | Python packages needed to run the notebook. |
| `README.md` | Project overview, data instructions, execution steps, and result summary. |

## Dataset

**Formula 1 World Championship (1950–2024), Ergast/Kaggle relational dataset**  
Source: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

The notebook expects the **original 14 CSV tables**, either directly in a single folder or as `archive.zip` containing these filenames at the root of the archive:

```text
circuits.csv                 constructors.csv
constructor_results.csv      constructor_standings.csv
drivers.csv                  driver_standings.csv
races.csv                    results.csv
qualifying.csv               lap_times.csv
pit_stops.csv                sprint_results.csv
seasons.csv                  status.csv
```

The source files are loaded into a `raw` dictionary; cleaning occurs on copies. The notebook does not modify the original CSVs. The dataset is not redistributed in this repository; download it from the linked source.

## Run the notebook

### Kaggle

1. Import/open `67-acl-67.ipynb` as a Kaggle notebook.
2. Attach the original Formula 1 dataset as an **Input**. The loader searches `/kaggle/input` for `archive.zip` or a directory with all 14 CSV files.
3. Enable Internet in notebook settings **if** the installation cell needs to fetch an unavailable package.
4. Choose **Run All / Save & Run All**. Do not manually skip earlier cells: later cells use their variables.
5. Find the exported CSV tables and generated PDF under `acl_labstyle_results/` in the notebook's working directory.

### Google Colab

1. Open https://colab.research.google.com/ and import `67-acl-67.ipynb`.
2. Upload the original `archive.zip` into Colab's `/content/` directory using the Files sidebar. The loader searches the notebook's current working directory recursively (normally `/content/`). Alternatively, extract the 14 CSV files together into a single folder under `/content/`.
3. If the ZIP contains extra nested folders instead of the CSV files at the archive root, **extract the ZIP into a folder**: the loader recognizes a directory containing all 14 CSV files.
4. Run all cells in order. Colab runtimes are temporary, so download the final outputs before disconnecting.

### Local Jupyter

Use a compatible Python environment with Jupyter installed (for example Python 3.11 or a supported Colab/Kaggle runtime):

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install jupyterlab
jupyter lab
```

Put `archive.zip` alongside the notebook, or put the 14 extracted CSV files together in a subfolder. The loader searches the current working directory recursively. TensorFlow/SHAP/LIME can take time to train or compute; the notebook's package-installation cell may also install missing libraries automatically.

> **Environment note:** `requirements.txt` lists required packages but does not lock versions. Different TensorFlow, NumPy, Pandas, or SHAP versions can lead to minor numerical or plotting differences. For exact replication, record and reuse versions from a successfully executed environment.

## Modeling approach

- **Prediction unit:** one `(raceId, driverId)` combination. Historical shared-drive duplicates are resolved by keeping the lowest `resultId` as a documented, outcome-independent source-record rule.
- **Target:** podium (`positionOrder` 1, 2, or 3) versus no podium.
- **Data preparation:** replace `\N` missing-value sentinels, parse dates and durations, audit primary and foreign keys, standardize country text, and perform EDA before and after cleaning.
- **Features:** 13 numerical pre-race features in five groups — starting grid, qualifying, driver form, constructor form, and prior championship standings. Past performance is calculated in chronological order and shifted so the current race result is never used as a feature.
- **Training / validation / test:** **1950–2019 (24,529 rows)** / **2020–2021 (780 rows)** / **2022–2024 (1,359 rows)**. Test years are reserved for final evaluation.
- **Experiments:** historical-window backtests, two preprocessing-order comparisons, feature-group ablation, and comparison with a simple `grid <= 3` rule.
- **Models:** logistic regression, random forests (depth-limited and unrestricted), and shallow feedforward neural networks (Adam and SGD optimizers).
- **Explainability:** permutation feature importance, global and local SHAP, and LIME; none of these explanations establish causation.

## Results from the saved run

The report and notebook select **Neural Network (Adam)** using **validation PR-AUC** (approximately **0.715**), with a classification threshold of **0.34** selected using validation F1.

| Metric | Held-out test: 2022–2024 |
| --- | ---: |
| ROC-AUC | 0.9324 |
| PR-AUC (average precision) | 0.7000 |
| F1 | 0.6453 |
| Precision | 0.6052 |
| Recall | 0.6912 |
| Accuracy | 0.8859 |

**Context:** Differences between model candidates are small. The selected neural network does **not** outperform every simpler model on the unseen test period. We keep the model selected on validation rather than choosing retrospectively from test scores.

### Data-engineering questions

1. **Front-row-to-podium conversion:** circuits are ranked separately for 1950–2013 and 2014–2024, including a qualifying-versus-grid comparison. Circuit rank correlation across the 16 eligible circuits represented in both eras is approximately **−0.03**.
2. **Home-race advantage:** after adjustment for grid band and decade, the estimated home-race odds ratio is **0.917** with a 95% bootstrap interval of **[0.780, 1.087]**. This does not provide convincing evidence of a home-race advantage or disadvantage.
3. **Mechanical retirements:** the recorded overall rate falls from **9.1%** (2014–2021) to **5.6%** (2022–2024). The report documents team rankings and classification limitations.

See the PDF report for the plots, denominators, methodological choices, and caveats supporting these findings.

## Generated files

After running the notebook, `acl_labstyle_results/` contains output artifacts including:

- `modeling_table.csv`, `model_comparison.csv`
- `primary_key_audit.csv`, `foreign_key_audit.csv`
- `training_window_backtests.csv`, `training_window_summary.csv`
- `historical_era_summary.csv`, `historical_grid_podium_rates.csv`
- `preprocessing_experiments.csv`, `feature_group_ablation.csv`
- `front_row_circuits.csv`, `mechanical_retirements.csv`
- `ACL_LabStyle_Analytical_Report.pdf` (a report generated by the notebook itself)

The standalone PDF included at the repository root is the submitted analytical report and should match the committed notebook version.

## Limitations

- The early dataset has sparse qualifying coverage and changing points systems and regulations.
- `grid = 0` combines several historical situations and is not a guaranteed pit-lane-start indicator.
- Deduplicating 1950s shared-drive records has historical ambiguity.
- Team identity can change when constructors are renamed.
- The small validation/test periods and model stochasticity limit confidence in tiny metric differences.
- Weather, safety cars, crashes, and other in-race events cannot be known at the prediction cutoff.
- Feature importance, SHAP, LIME, and observational associations **must not be interpreted causally**.

## Submission

The milestone brief requests an **accessible Kaggle notebook link**, a **GitHub repository link**, and a **PDF analytical report**. Committing these files to GitHub satisfies the repository component, not the separate notebook-hosting or report-submission steps. Verify that the committed notebook and uploaded report come from the same final run.
