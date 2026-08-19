# Pipeline 2 Results Report

Generated: 2026-07-30T00:50:46.711610

## Task

Predict whether CKD first appears at the next observation using current and strict past only predictors.

## Modeling dataset

Raw records: 4,000

Eligible preprocessed examples: 3,399

Patients: 400

Positive first onset examples: 185

Positive prevalence: 0.0544

Main predictors: 80

## Validation

Outer patient folds: 5

Inner patient folds: 3

Supplementary stability seeds: [42, 7, 2026]

No patient appears in both training and testing data in the main evaluation.

## Selected model

Model: lightgbm

Mean fold PR AUC: 0.5447

Fold PR AUC standard deviation: 0.0352

Pooled OOF PR AUC: 0.5209

Patient bootstrap 95 percent confidence interval: 0.4429 to 0.6005

Pooled OOF ROC AUC: 0.8712

Pooled OOF MCC: 0.5478

Selected calibration method: sigmoid

## Interpretation boundary

This is internal patient separated future first onset prediction. It is not causal inference, external validation, or deployment evidence.