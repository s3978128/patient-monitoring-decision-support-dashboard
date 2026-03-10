"""
Clinical Thresholds
Central definition of all vital-sign ranges and clinical alert thresholds
used across the CDSS pipeline.

Changing a value here propagates automatically to quality checks,
anomaly detection, and the clinical rules engine.
"""

# ---------------------------------------------------------------------------
# Data-quality limits
# Values outside these ranges indicate sensor errors or data-entry mistakes.
# Used by: modules/quality_checks.py
# ---------------------------------------------------------------------------
PHYSIOLOGICAL_LIMITS = {
    "heart_rate":               (30, 200),    # bpm
    "blood_pressure_systolic":  (50, 250),    # mmHg
    "blood_pressure_diastolic": (30, 150),    # mmHg
    "temperature":              (30.0, 45.0), # °C
    "respiratory_rate":         (5,  60),     # breaths/min
    "oxygen_saturation":        (50, 100),    # %
}

# ---------------------------------------------------------------------------
# Clinical anomaly ranges
# Narrower than physiological limits; flags values warranting clinical review.
# Used by: modules/anomaly_detection.py (rule-based detector)
# ---------------------------------------------------------------------------
CLINICAL_ANOMALY_RANGES = {
    "heart_rate":               (40, 120),
    "blood_pressure_systolic":  (80, 180),
    "blood_pressure_diastolic": (50, 100),
    "temperature":              (35.0, 39.0),
    "respiratory_rate":         (8,  30),
    "oxygen_saturation":        (85, 100),
}

# ---------------------------------------------------------------------------
# Isolation Forest feature columns
# Only clinical measurements; demographics are intentionally excluded so
# they cannot distort the multivariate anomaly model.
# Used by: modules/anomaly_detection.py (Isolation Forest)
# ---------------------------------------------------------------------------
VITAL_SIGN_FEATURES = [
    "heart_rate",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "temperature",
    "respiratory_rate",
    "oxygen_saturation",
]

# ---------------------------------------------------------------------------
# Clinical-rules alert thresholds
# Used by: modules/clinical_rules.py
# ---------------------------------------------------------------------------

# Tachycardia / bradycardia  (AHA ACLS / common inpatient triage)
TACHYCARDIA_WARNING  = 100   # heart_rate > this  → WARNING
TACHYCARDIA_CRITICAL = 130   # heart_rate >= this → CRITICAL

# Hypertension  (ACC/AHA 2017, simplified)
HYPERTENSION_STAGE1_SYSTOLIC  = 130
HYPERTENSION_STAGE1_DIASTOLIC = 80
HYPERTENSION_STAGE2_SYSTOLIC  = 140
HYPERTENSION_STAGE2_DIASTOLIC = 90
HYPERTENSION_CRISIS_SYSTOLIC  = 180
HYPERTENSION_CRISIS_DIASTOLIC = 120

# Hypotension  (common acute-care thresholds)
HYPOTENSION_WARNING_SYSTOLIC   = 90
HYPOTENSION_WARNING_DIASTOLIC  = 60
HYPOTENSION_CRITICAL_SYSTOLIC  = 80
HYPOTENSION_CRITICAL_DIASTOLIC = 50

# Hypoxemia / oxygen saturation  (NEWS2-inspired)
HYPOXEMIA_WARNING  = 95   # oxygen_sat <= this → WARNING
HYPOXEMIA_CRITICAL = 91   # oxygen_sat <= this → CRITICAL

# Respiratory rate  (NEWS2-inspired)
RESPIRATORY_HIGH_WARNING  = 22
RESPIRATORY_HIGH_CRITICAL = 30
RESPIRATORY_LOW_WARNING   = 8
RESPIRATORY_LOW_CRITICAL  = 6

# Temperature / fever  (common inpatient triage)
FEVER_WARNING_HIGH  = 38.0
FEVER_CRITICAL_HIGH = 39.5
FEVER_WARNING_LOW   = 35.0
FEVER_CRITICAL_LOW  = 34.0
