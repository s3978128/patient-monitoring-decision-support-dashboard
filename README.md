# Patient Monitoring and Decision Support Dashboard

Interactive Streamlit dashboard for monitoring synthetic clinical data quality, anomaly signals, and simplified guideline-based risk inference.

This project is designed for monitoring and educational use. It is not intended for direct clinical diagnosis or treatment decisions.

## What This Project Does

- Generates simulated patient records with configurable anomalies, missing values, severe errors, and duplicate rows.
- Runs data quality checks (missing values, schema completeness, physiological limits, duplicate context).
- Detects anomalies using:
	- Rule-based vital-sign thresholds
	- Isolation Forest for multivariate outliers
- Provides explainability views:
	- Isolation Forest feature-importance proxy
	- Rule trigger tables
	- Guideline-based diagnosis summaries and recommendations
- Produces downloadable report outputs (JSON/CSV) from the dashboard.

## Dashboard Views

- Overview
	- Cohort-level summary metrics and distributions.
- Monitoring Panel
	- System Health metrics:
		- Data Quality Score
		- Anomaly Rate
		- Alerted Patient Rate
	- Embedded Quality Metrics section.
- Anomaly Detection
	- Rule-based anomaly counts and details.
	- Isolation Forest anomalies plus feature-importance explainability.
- Clinical Alerts
	- Severity and framework breakdowns.
	- Guideline-based diagnosis context:
		- diagnosis
		- triggered guideline rule
		- recommendation
		- explanation
- Reports
	- Comprehensive Report
	- Quality Report
	- Summary Statistics

## Project Structure

```
clinical-cdss-monitoring/
	dashboard/
		app.py
	modules/
		anomaly_detection.py
		clinical_rules.py
		clinical_thresholds.py
		data_simulator.py
		quality_checks.py
		reporting.py
	data/
		simulated_patients.csv
	notebooks/
		exploration.ipynb
	requirements.txt
	README.md
```

## Requirements

Dependencies listed in requirements.txt:

- pandas
- numpy
- scikit-learn
- streamlit
- plotly

## How to Use the App

1. In the sidebar, configure data generation parameters:
	 - Number of Patients
	 - Clinical Anomaly Rate
	 - Severe Error Rate
	 - Duplicate Rate
	 - Missing Value Rate
2. Choose whether to clean duplicate rows before analysis.
3. Navigate through views to inspect quality, anomalies, alerts, and reports.
4. Use Report Generation to export results.

## Explainability Notes

- Isolation Forest feature importance is an approximate global signal.
	It is based on distribution differences between anomaly and non-anomaly groups, not a causal per-patient explanation.
- Guideline-based diagnoses are simplified inference rules meant for transparent monitoring support.
- Always combine dashboard outputs with proper clinical workflows and professional judgment.

## Safety Disclaimer

This project demonstrates monitoring, alerting, and explainability patterns using simulated data and simplified thresholds. It is not a medical device and must not be used as the sole basis for clinical decisions.
