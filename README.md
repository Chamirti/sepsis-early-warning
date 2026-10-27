# Sepsis Early-Warning System

Predicting sepsis hours before onset from hourly ICU data, with external validation on a second hospital.

*Work in progress: data pipeline and cohort complete; modeling, evaluation and deployment in progress.*

## Data
PhysioNet/Computing in Cardiology Challenge 2019 (v1.0.0): hourly ICU vitals, labs and demographics for 40,336 patients from two hospital systems (A: 20,336; B: 20,000). SepsisLabel follows Sepsis-3 and turns on 6 hours before onset.

## Decisions

**Cohort**
- Excluded septic patients labeled from their first recorded hour (n = 426; A: 203, B: 223). Their onset falls within ~6 h of the start of the record or before ICU admission, so it cannot be determined, and there is nothing to predict in advance. This follows the practice of excluding early-onset cases in prior work (e.g. DeepAISE; Moor et al., 2021), with the threshold set by this dataset's labeling.
- The model targets sepsis that develops during the ICU stay. Results are not directly comparable to the 2019 challenge leaderboard, which evaluated all patients.
- Cohort: 39,910 patients included (A: 20,133; B: 19,777).

**Splits**
| Split | Patients | Septic (rate) | Hours | Positive hours |
|---|---|---|---|---|
| Train (A) | 14,092 | 1,110 (7.88%) | 550,591 | 1.95% |
| Validation (A) | 3,020 | 238 (7.88%) | 119,450 | 1.93% |
| Test (A) | 3,021 | 239 (7.91%) | 118,406 | 1.96% |
| External (B) | 19,777 | 919 (4.65%) | 760,117 | 1.17% |

- Hospital A: patient-level 70/15/15 split, stratified by sepsis status, using a reproducible md5-based shuffle.
- Hospital B: held out entirely as an external test set; not used for any modeling decisions.

**Labels**
- SepsisLabel used as provided (already shifted 6 h before onset); no additional shift.

**Known data artifacts**
- **Truncation:** septic records end 1–3 h after onset (~99% end 8–10 rows after the first label), so total record length is leakage.
- **Length cap:** non-septic records rarely exceed 60 h (A: 6 of 18,546), while ~35% of septic records do. Rows after hour 60 are ~9× more likely to be positive at A (14.2% vs 1.5%). ICULOS is therefore excluded from the main model, and only fixed-window features are used.
- **Hospital differences:** sepsis prevalence differs (7.9% vs 4.6% after exclusion), so lower PPV and a calibration shift are expected at B. Measurement patterns also differ: blood gases are nearly absent at B, EtCO2 is never measured at A, and DBP is missing 48% of hours at A vs 14% at B.
- **Lab ordering** surges at onset, reflecting clinician suspicion rather than early warning; performance is evaluated on pre-onset hours separately.

## Reproduce
1. Download the data:
   `aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/ data/raw/`
2. `pip install -r requirements.txt`
3. `python -m src.load_data`
4. `python -m src.build_cohort`

## Project structure
- `src/` — pipeline code (`columns.py`, `load_data.py`, `build_cohort.py`)
- `sql/` — cohort and feature SQL
- `notebooks/` — exploration and checks (01–04)