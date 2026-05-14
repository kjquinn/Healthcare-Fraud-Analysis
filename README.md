Healthcare Claims Fraud Detection
A full end-to-end machine learning pipeline for detecting fraudulent healthcare insurance claims, built in Python using a dataset of 43,335 real-world claims processed between 2017 and 2019.
Overview
Insurance fraud costs the U.S. healthcare system tens of billions of dollars annually. This project builds a deployable fraud detection model that flags suspicious claims for investigator review — replacing manual screening with a data-driven approach. The project covers the full analytics lifecycle: data profiling, feature engineering, cluster analysis, classification modeling, and a live interactive interface built with Gradio.
Tools & Libraries

Python — pandas, numpy, matplotlib, seaborn
Machine Learning — PyCaret, XGBoost, Random Forest, scikit-learn
Class Imbalance — SMOTE (via PyCaret's fix_imbalance)
EDA — ydata-profiling
Deployment — Gradio (interactive fraud detection interface)

Project Structure
PhaseDescriptionPhase 1Data profiling — shape, types, missing values, fraud rate (4.47%)Phase 2Univariate analysis — fraud rates by specialty, age, billing amount, drive distancePhase 3Feature engineering — billed_to_allowed_ratio, overbilling flags, age bins, temporal featuresPhase 4Bivariate & interaction analysis — specialty × age heatmaps, combination risk flagsPhase 5AK-Means cluster analysis — unsupervised fraud pattern detectionPhase 5BClassification modeling — XGBoost vs. Random Forest with SMOTEPhase 6ABusiness recommendations — flagging rules and investigator prioritization logicPhase 6BGradio interface — auditor-facing tool with fraud probability and risk level output
Key Findings

Fraud rate: 4.47% (1,935 of 43,335 claims) — a severely imbalanced dataset requiring SMOTE
Most important feature: billed_to_allowed_ratio — fraudulent claims systematically bill at extreme multiples of the allowed amount
Data leakage identified: claim_disputed was excluded from modeling — it is generated post-investigation and unavailable at prediction time
Age imputation strategy: Missing member_age values were imputed using median grouped by specialty, not the global median, to preserve contextual accuracy
Model evaluation: AUC-ROC, Recall, Precision, and F1 used — not accuracy, which is misleading on imbalanced data

How to Run

Open the notebook in Google Colab
Mount your Google Drive and update the CSV path in Cell 2
Run all cells in order (Phases 1–6)
Launch the Gradio interface in Phase 6B to test the live fraud scoring tool
