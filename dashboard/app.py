"""
Patient Monitoring and Decision Support Dashboard
Interactive web dashboard for monitoring clinical decision support system.
"""

# imports
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent)) # append parent directory to Python path 
from modules.data_simulator import PatientDataSimulator
from modules.quality_checks import DataQualityChecker
from modules.anomaly_detection import AnomalyDetector
from modules.clinical_rules import ClinicalRulesEngine, AlertSeverity
from modules.reporting import ClinicalReportGenerator
from modules.clinical_thresholds import CLINICAL_ANOMALY_RANGES


# Page configuration
st.set_page_config(
    page_title="Patient Monitoring and Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


def apply_frontend_theme():
    """Apply lightweight visual theme refinements for readability and hierarchy."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Source+Sans+3:wght@400;600&display=swap');

        :root {
            --brand-ink: #10273f;
            --brand-teal: #0b5f59;
            --brand-sand: #f3f6f4;
            --brand-warm: #fbf2e6;
            --brand-border: #c6d2ce;
            --text-main: #17212b;
            --text-secondary: #334155;
            --text-muted: #475569;
        }

        html, body, [class*="css"] {
            font-family: 'Source Sans 3', sans-serif;
           
        }

        h1, h2, h3 {
            font-family: 'Manrope', sans-serif !important;
           
            letter-spacing: -0.01em;
        }

        p, li, label, .stMarkdown, .stText, .stCaption {
           
        }

        .stCaption {
            
        }

        .app-banner {
            border: 1px solid var(--brand-border);
           
            border-radius: 14px;
            padding: 0.9rem 1.1rem;
            margin: 0.25rem 0 1rem 0;
            
        }

        .app-banner strong {
           
            font-family: 'Manrope', sans-serif;
            font-size: 1.05rem;
        }

        .clinician-note {
            border-left: 5px solid var(--brand-teal);
           
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
            margin: 0.35rem 0 0.8rem 0;
        }

        .stMetric {
            border: 1px solid #d8e2de;
            border-radius: 10px;
            padding: 0.2rem 0.35rem;
           
            box-shadow: 0 1px 0 rgba(16, 39, 63, 0.03);
        }

        .stMetric label,
        .stMetric div {
            
        }

        .stAlert {
           
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #d8e2de;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_data(n_patients=100, anomaly_rate=0.15, severe_error_rate=0.05, 
             duplicate_rate=0.1, missing_value_rate=0.1):
    """Load or generate patient data with configurable error injection."""
    data_path = Path(__file__).parent.parent / 'data' / 'simulated_patients.csv'
    
    # Check if we should use existing data or regenerate
    use_existing = st.sidebar.checkbox("Use existing CSV (if available)", value=False)
    
    df = None
    if use_existing and data_path.exists():
        try:
            df = pd.read_csv(data_path, encoding='utf-8', on_bad_lines='warn')
            st.info(f"Loaded existing data from {data_path}")
        except Exception as e:
            st.warning(f"Could not load existing CSV file: {e}")
            df = None
    
    if df is None or not use_existing: # Generate new data if loading failed or if user opted to generate
        # Generate sample data with error injection
        simulator = PatientDataSimulator(
            seed=42, 
            anomaly_rate=anomaly_rate,
            severe_error_rate=severe_error_rate
        )
        df = simulator.generate_dataset(
            n_patients=n_patients,
            duplicate_rate=duplicate_rate,
            missing_value_rate=missing_value_rate
        )
        
        # Save the generated data
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(data_path, index=False, encoding='utf-8')
            st.info(f"Generated and saved new data with {len(df)} records")
        except Exception as e:
            st.warning(f"Could not save data: {e}")
    
    return df


def get_duplicate_rows_preview(df: pd.DataFrame) -> pd.DataFrame:
    """Return duplicate rows by patient_id for pre-cleaning review."""
    if df.empty or 'patient_id' not in df.columns:
        return pd.DataFrame()

    duplicate_rows = df[df.duplicated(subset=['patient_id'], keep=False)].copy()
    if duplicate_rows.empty:
        return duplicate_rows

    if 'timestamp' in duplicate_rows.columns:
        duplicate_rows['timestamp'] = pd.to_datetime(duplicate_rows['timestamp'], errors='coerce') # ensure timestamp is datetime for sorting, prioritize most recent records in preview
        duplicate_rows = duplicate_rows.sort_values(by='timestamp', ascending=False, na_position='last')

    priority_cols = [col for col in ['patient_id', 'name', 'timestamp'] if col in duplicate_rows.columns]
    remaining_cols = [col for col in duplicate_rows.columns if col not in priority_cols]
    return duplicate_rows[priority_cols + remaining_cols]


def deduplicate_patient_rows(df: pd.DataFrame, clean_dataset: bool = True):
    """Optionally remove duplicate patient rows and keep the most recent record per patient_id."""
    if df.empty or 'patient_id' not in df.columns:
        return df, {
            'input_rows': len(df),
            'output_rows': len(df),
            'duplicate_rows_detected': 0,
            'duplicate_patient_ids_affected': 0,
            'potential_removed_rows': 0,
            'removed_rows': 0,
            'cleaning_applied': clean_dataset,
        }

    working_df = df.copy()
    duplicate_mask = working_df.duplicated(subset=['patient_id'], keep=False)
    duplicate_rows_detected = int(duplicate_mask.sum())
    duplicate_patient_ids_affected = int(working_df.loc[duplicate_mask, 'patient_id'].nunique())
    potential_removed_rows = max(0, len(working_df) - int(working_df['patient_id'].nunique()))

    if clean_dataset:
        dedup_df = working_df.copy()
        if 'timestamp' in dedup_df.columns:
            dedup_df['timestamp'] = pd.to_datetime(dedup_df['timestamp'], errors='coerce')
            dedup_df = dedup_df.sort_values(by='timestamp', ascending=False, na_position='last')
        deduped_df = dedup_df.drop_duplicates(subset=['patient_id'], keep='first').reset_index(drop=True)
    else:
        deduped_df = working_df.reset_index(drop=True)

    removed_rows = len(working_df) - len(deduped_df)

    return deduped_df, {
        'input_rows': len(df),
        'output_rows': len(deduped_df),
        'duplicate_rows_detected': duplicate_rows_detected,
        'duplicate_patient_ids_affected': duplicate_patient_ids_affected, 
        'potential_removed_rows': potential_removed_rows,
        'removed_rows': removed_rows,
        'cleaning_applied': clean_dataset,
    }


def evaluate_alerts_for_dataset(df: pd.DataFrame): 
    """Evaluate clinical rules for all rows and return a flat alert list."""
    engine = ClinicalRulesEngine() 
    all_alerts = []
    for _, row in df.iterrows():
        all_alerts.extend(engine.evaluate_all_rules(row.to_dict()))
    return all_alerts


def evaluate_inferences_for_dataset(df: pd.DataFrame):
    """Evaluate guideline-based inferences for all rows and return a flat list."""
    engine = ClinicalRulesEngine()
    all_inferences = []
    for _, row in df.iterrows():
        all_inferences.extend(engine.evaluate_guideline_inferences(row.to_dict()))
    return all_inferences


def calculate_system_health_metrics(
    df: pd.DataFrame,
    dedup_summary: dict = None,
    contamination: float = 0.1,
):
    """Compute dashboard-level health metrics and their explainability breakdown."""
    checker = DataQualityChecker()
    quality_report = checker.generate_quality_report(df)

    missing_cells = int(df.isnull().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1]) if not df.empty else 0
    completeness_pct = (1 - (missing_cells / total_cells)) * 100 if total_cells > 0 else 100.0

    dedup_summary = dedup_summary or {}  # dedup_summary may be None if deduplication hasn't been performed yet
    duplicate_rows_detected = int(dedup_summary.get('duplicate_rows_detected', 0))
    duplicate_rows_removed = int(dedup_summary.get('removed_rows', 0))
    duplicate_patient_ids_affected = int(dedup_summary.get('duplicate_patient_ids_affected', 0))
    input_rows = int(dedup_summary.get('input_rows', len(df)))

    if input_rows > 0:
        duplicate_rate_pct = (duplicate_rows_detected / input_rows) * 100
    else:
        duplicate_rate_pct = 0.0

    phys_violations = checker.check_physiological_limits(df)
    total_phys_violations = sum(len(indices) for indices in phys_violations.values())
    phys_rate_pct = (total_phys_violations / len(df) * 100) if len(df) > 0 else 0.0

    quality_score = max(
        0.0,
        min(
            100.0,
            completeness_pct - (0.6 * duplicate_rate_pct) - (0.4 * phys_rate_pct),
        ),
    ) # quality score is a weighted combination of completeness and error rates, with duplicate rate weighted more heavily since duplicates can distort analysis more than missing values in this context
    # TODO: add this explanation to dashboard

    anomaly_report = AnomalyDetector(contamination=contamination).generate_anomaly_report(df)
    anomaly_rate_pct = anomaly_report.get('isolation_forest_anomaly_rate', 0.0)

    all_alerts = evaluate_alerts_for_dataset(df)
    patient_ids_with_alerts = {
        alert.patient_id for alert in all_alerts if getattr(alert, 'patient_id', None) is not None
    }
    total_patients = int(df['patient_id'].nunique()) if 'patient_id' in df.columns else len(df)
    alert_rate_pct = (len(patient_ids_with_alerts) / total_patients * 100) if total_patients > 0 else 0.0

    return {
        'data_quality_score': round(quality_score, 2),
        'anomaly_rate': round(float(anomaly_rate_pct), 2),
        'alert_rate': round(float(alert_rate_pct), 2),
        'quality_breakdown': {
            'completeness_pct': round(float(completeness_pct), 2),
            'duplicate_rate_pct': round(float(duplicate_rate_pct), 2),
            'duplicate_rows_detected': duplicate_rows_detected,
            'duplicate_rows_removed': duplicate_rows_removed,
            'duplicate_patient_ids_affected': duplicate_patient_ids_affected,
            'phys_violation_rate_pct': round(float(phys_rate_pct), 2),
            'missing_cells': missing_cells,
            'missing_columns': quality_report.get('missing_columns', []),
        },
        'anomaly_report': anomaly_report,
        'alerts': all_alerts,
    }


def main():
    """Main dashboard application."""
    apply_frontend_theme()
    
    # Header
    st.title("Patient Monitoring and Decision Support Dashboard")
    st.markdown(
        """
        <div class="app-banner">
            <strong>Clinical Monitoring Workspace</strong><br/>
            Use this dashboard to review data quality, anomaly signals, and guideline-based alerts with transparent reasoning.
        </div>
        """,
        unsafe_allow_html=True, # allow HTML for custom styling of the banner
    )

    
    # Sidebar options
    view_mode = st.sidebar.selectbox(
        "Select View",
        ["Overview", "Monitoring Panel", "Anomaly Detection", "Clinical Alerts", "Reports"]
    )
    
    # Data generation options
    st.sidebar.subheader("Data Generation Settings")
    n_patients = st.sidebar.slider("Number of Patients", 10, 500, 100)
    anomaly_rate = st.sidebar.slider("Clinical Anomaly Rate", 0.0, 0.5, 0.15)
    severe_error_rate = st.sidebar.slider("Severe Error Rate", 0.0, 0.2, 0.05)
    duplicate_rate = st.sidebar.slider("Duplicate Rate", 0.0, 0.3, 0.1)
    missing_value_rate = st.sidebar.slider("Missing Value Rate", 0.0, 0.3, 0.0)
    clean_duplicates = st.sidebar.checkbox("Clean duplicate patient rows before analysis", value=True)
    
    # Load data with settings
    raw_df = load_data(
        n_patients=n_patients,
        anomaly_rate=anomaly_rate,
        severe_error_rate=severe_error_rate,
        duplicate_rate=duplicate_rate,
        missing_value_rate=missing_value_rate
    )
    duplicate_preview = get_duplicate_rows_preview(raw_df)
    if not duplicate_preview.empty:
        st.warning(
            "Duplicate rows detected in raw dataset before cleaning. "
            f"Detected {len(duplicate_preview)} rows across duplicated patient IDs."
        )
        with st.expander(f"View {len(duplicate_preview)} duplicate rows before cleaning"):
            st.dataframe(duplicate_preview, use_container_width=True, hide_index=True)

    df, dedup_summary = deduplicate_patient_rows(raw_df, clean_dataset=clean_duplicates)

    if dedup_summary['cleaning_applied'] and dedup_summary['removed_rows'] > 0:
        st.info(
            "Deduplicated records before analysis: detected "
            f"{dedup_summary['duplicate_rows_detected']} duplicate rows across "
            f"{dedup_summary['duplicate_patient_ids_affected']} patient IDs and removed "
            f"{dedup_summary['removed_rows']} rows by keeping the most recent record per patient ID."
        )
    elif (not dedup_summary['cleaning_applied']) and dedup_summary['duplicate_rows_detected'] > 0:
        st.info(
            "Duplicate cleaning is currently disabled. "
            f"{dedup_summary['duplicate_rows_detected']} duplicate rows will remain in analysis unless cleaning is enabled."
        )
    
    # Main content based on view mode
    if view_mode == "Overview":
        show_overview(df)
    elif view_mode == "Monitoring Panel":
        show_monitoring_panel(df, dedup_summary=dedup_summary)
    elif view_mode == "Anomaly Detection":
        show_anomaly_detection(df)
    elif view_mode == "Clinical Alerts":
        show_clinical_alerts(df)
    elif view_mode == "Reports":
        show_reports(df)
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Records", len(df))
    st.sidebar.metric("Total Columns", len(df.columns))
    if dedup_summary['duplicate_rows_detected'] > 0:
        st.sidebar.metric("Detected Duplicate Rows", dedup_summary['duplicate_rows_detected'])
        st.sidebar.metric("Removed Duplicates", dedup_summary['removed_rows'])


def show_overview(df):
    """Display overview dashboard."""
    st.header("System Overview")
    st.caption("Snapshot of the current patient cohort after optional data cleaning.")
    view_df = df.copy()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        avg_age = view_df['age'].mean() if 'age' in view_df.columns else 0
        st.metric("Average Age", f"{avg_age:.1f}")
    with col3:
        if 'heart_rate' in view_df.columns:
            heart_rate_series = pd.to_numeric(view_df['heart_rate'], errors='coerce')
            avg_hr = heart_rate_series.mean()
            st.metric("Avg Heart Rate", f"{avg_hr:.0f} bpm")
    with col4:
        if 'temperature' in view_df.columns:
            temperature_series = pd.to_numeric(view_df['temperature'], errors='coerce')
            avg_temp = temperature_series.mean()
            st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
    
    # Visualizations
    st.subheader("Data Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'age' in view_df.columns:
            fig = px.histogram(view_df, x='age', nbins=20, title="Age Distribution", color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        if 'gender' in view_df.columns:
            gender_counts = view_df['gender'].value_counts()
            fig = px.pie(values=gender_counts.values, names=gender_counts.index, 
                        title="Gender Distribution", color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # All patient data
    st.subheader("All Patient Records")
    st.dataframe(view_df, use_container_width=True)


def show_quality_metrics(df, dedup_summary=None, embedded=False):
    """Display data quality metrics."""
    if embedded:
        st.subheader("Quality Metrics")
    else:
        st.header("Data Quality Metrics")
    st.caption("Quality findings are shown with duplicate handling context so results remain interpretable.")
    
    checker = DataQualityChecker()
    quality_report = checker.generate_quality_report(df)
    
    # Quality score
    missing_pct = quality_report.get('missing_values', {})
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())
    missing_columns = quality_report.get('missing_columns', [])
    
    # Check for duplicates and physiological violations
    duplicate_count = 0
    duplicate_ids = []
    if 'patient_id' in df.columns:
        duplicate_count, duplicate_ids = checker.check_duplicates(df, 'patient_id')

    detected_duplicate_rows = (dedup_summary or {}).get('duplicate_rows_detected', 0)
    removed_duplicate_rows = (dedup_summary or {}).get('removed_rows', 0)
    affected_duplicate_ids = (dedup_summary or {}).get('duplicate_patient_ids_affected', 0)
    potential_removed_rows = (dedup_summary or {}).get('potential_removed_rows', 0)
    cleaning_applied = (dedup_summary or {}).get('cleaning_applied', False)
    duplicate_metric_label = (
        "Duplicate IDs Remaining After Cleaning"
        if cleaning_applied
        else "Duplicate IDs In Current Dataset"
    )
    
    phys_violations = checker.check_physiological_limits(df)
    total_phys_violations = sum(len(v) for v in phys_violations.values())
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", quality_report['total_records'])
    with col2:
        st.metric("Total Columns", quality_report['total_columns'])
    with col3:
        st.metric(
            duplicate_metric_label,
            duplicate_count,
        )
    with col4:
        st.metric("Physiological Violations", total_phys_violations)
    
    # Missing values count
    if total_missing > 0:
        st.subheader("Missing Values")
        st.warning(f"Found {total_missing} missing values across the dataset")

        missing_rows = df[df.isnull().any(axis=1)].copy() # Get rows with any missing values
        missing_rows['missing_fields'] = missing_rows.apply(
            # Create a comma-separated list of missing fields for each row
            lambda row: ', '.join(row.index[row.isnull()].tolist()),
            axis=1, # Add a new column that lists which fields are missing for each row
        )

        priority_cols = [
            col for col in ['patient_id', 'name', 'timestamp', 'missing_fields']
            if col in missing_rows.columns
        ]
        remaining_cols = [col for col in missing_rows.columns if col not in priority_cols]

        with st.expander(f"View {len(missing_rows)} Rows With Missing Values"):
            st.dataframe(
                missing_rows[priority_cols + remaining_cols],
                use_container_width=True,
            )
    elif not missing_columns:
        st.success("✅ No missing values detected!")

    if missing_columns:
        st.subheader("Missing Columns")
        st.warning(
            f"Found {len(missing_columns)} missing columns in the dataset schema"
        )
        missing_columns_df = pd.DataFrame({'Missing Column': missing_columns})
        st.dataframe(missing_columns_df, use_container_width=True)
    
    # Duplicate records - show both duplicates sorted by similalrity and by timestamp

    st.subheader("Duplicate Records")
    if detected_duplicate_rows > 0:
        st.warning(
            f"Detected {detected_duplicate_rows} duplicate rows across {affected_duplicate_ids} patient IDs before analysis."
        )
        if not cleaning_applied:
            st.info(
                "Duplicate cleaning is currently disabled, so these duplicate rows are included in current analysis."
            )
        summary_df = pd.DataFrame(
            [
                {
                    'Metric': 'Duplicate rows detected at ingestion',
                    'Value': detected_duplicate_rows,
                },
                {
                    'Metric': 'Patient IDs affected',
                    'Value': affected_duplicate_ids,
                },
                {
                    'Metric': 'Rows removed before analysis',
                    'Value': removed_duplicate_rows,
                },
                {
                    'Metric': 'Rows removable if cleaning is enabled',
                    'Value': potential_removed_rows,
                },
                {
                    'Metric': 'Duplicate IDs remaining in cleaned dataset',
                    'Value': duplicate_count,
                },
                {
                    'Metric': 'Cleaning applied',
                    'Value': 'Yes' if cleaning_applied else 'No',
                },
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    elif duplicate_count > 0:
        st.subheader("Duplicate Records")
        st.error(f"Found {duplicate_count} duplicate patient IDs")
        
        if duplicate_ids:
            with st.expander(f"View {len(duplicate_ids)} Duplicate Records"):
                duplicate_records = df[df['patient_id'].isin(duplicate_ids)].sort_values(by='timestamp', ascending=False)
                # Select relevant columns for display, including name if available
                display_cols = [col for col in ['patient_id', 'name', 'timestamp'] if col in duplicate_records.columns]
                st.dataframe(duplicate_records[display_cols], use_container_width=True)
    else:
        st.success("✅ No duplicate patient IDs detected!")
    
    
    # Physiological limit violations
    # explain what this means with an info box
    st.subheader("Physiological Limit Violations")
    st.info("Physiological limit violations indicate values that fall outside of normal human ranges for vital signs. These may indicate data entry errors or critical patient conditions.")
    if total_phys_violations > 0:
        st.error(f"Found {total_phys_violations} values outside physiological limits")
        fig = px.bar(x=[f"{k} - {len(v)}" for k, v in phys_violations.items()], 
                    y=[len(v) for v in phys_violations.values()],
                    title="Physiological Limit Violations by Vital Sign", 
                    color_discrete_sequence=['#e74c3c'])
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Show details of violations
        for vital_sign, indices in phys_violations.items():
            with st.expander(f"{vital_sign} - {len(indices)} violations"):
                if vital_sign in df.columns:
                    # Show patient_id, name, and the vital sign value for the violating records
                    display_cols = [col for col in ['patient_id', 'name', vital_sign] if col in df.columns]
                    violation_records = df.loc[indices, display_cols]
                    st.dataframe(violation_records, use_container_width=True)
    else:
        st.success("✅ No physiological limit violations!")

def _flag_vitals(row: pd.Series) -> str:
    """Return a human-readable string listing which vital signs are outside CLINICAL_ANOMALY_RANGES."""
    flags = []
    for col, (lo, hi) in CLINICAL_ANOMALY_RANGES.items():
        val = row.get(col)
        if pd.isna(val):
            continue
        if val < lo:
            flags.append(f"{col} {val} (low < {lo})")
        elif val > hi:
            flags.append(f"{col} {val} (high > {hi})")
    return ", ".join(flags) if flags else "multivariate pattern"


def show_anomaly_detection(df):
    """Display anomaly detection results."""
    st.header("Anomaly Detection")
    st.info(
        "Two complementary methods are used. **Rule-based** detection flags individual "
        "vital signs outside predefined clinical ranges. **Isolation Forest** is a "
        "machine learning model that catches unusual *combinations* of values even when "
        "no single vital sign is outside its range. Use the contamination slider to "
        "control how aggressively the model flags outliers."
    )

    contamination = st.slider("Isolation Forest Contamination Rate", 0.01, 0.3, 0.1)
    detector = AnomalyDetector(contamination=contamination)
    anomaly_report = detector.generate_anomaly_report(df)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", anomaly_report['total_records'])
    with col2:
        st.metric("Rule-Based Anomalies", anomaly_report['total_anomalies'])
    with col3:
        st.metric("Isolation Forest Anomalies", anomaly_report['isolation_forest_anomaly_count'])
    with col4:
        st.metric("IF Anomaly Rate", f"{anomaly_report['isolation_forest_anomaly_rate']:.2f}%")

    st.divider()

    # Rule-Based ───────────────────────────────────────────────────────────
    st.subheader("Rule-Based Anomalies")
    vital_anomalies = anomaly_report['vital_sign_anomalies']

    if vital_anomalies:
        anomaly_counts = {k: len(v) for k, v in vital_anomalies.items()}
        fig = px.bar(
            x=list(anomaly_counts.keys()),
            y=list(anomaly_counts.values()),
            labels={'x': 'Vital Sign', 'y': 'Count'},
            title="Anomalies by Vital Sign",
            color_discrete_sequence=['#e74c3c'],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Build a single unified table: one row per (patient, flagged vital)
        rows = []
        vital_cols = [col for col in CLINICAL_ANOMALY_RANGES if col in df.columns]
        for vital_sign, indices in vital_anomalies.items():
            if vital_sign not in df.columns:
                continue
            lo, hi = CLINICAL_ANOMALY_RANGES[vital_sign]
            for idx in indices:
                val = df.at[idx, vital_sign]
                direction = "low" if val < lo else "high"
                normal_range = f"{lo} – {hi}"
                rows.append({
                    'patient_id':   df.at[idx, 'patient_id'] if 'patient_id' in df.columns else idx,
                    'name':         df.at[idx, 'name'] if 'name' in df.columns else '',
                    'flagged_vital': vital_sign,
                    'value':        val,
                    'direction':    direction,
                    'normal_range': normal_range,
                })

        if rows:
            flagged_df = pd.DataFrame(rows).sort_values(['patient_id', 'flagged_vital'])
            with st.expander(f"View {len(rows)} flagged vital sign readings", expanded=True):
                st.dataframe(flagged_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No rule-based anomalies detected in vital signs!")

    st.divider()

    st.subheader("Isolation Forest Explainability")
    feature_importance = anomaly_report.get('isolation_forest_feature_importance', {})
    if feature_importance:
        fi_df = pd.DataFrame(
            [
                {'Feature': feature, 'Importance': value}
                for feature, value in feature_importance.items()
            ]
        ).sort_values(by='Importance', ascending=True)

        fig = px.bar(
            fi_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Approximate Feature Importance for Isolation Forest Anomalies",
            color_discrete_sequence=['#1abc9c'],
        )
        fig.update_layout(xaxis_title='Relative contribution (%)', yaxis_title='Feature')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.caption(
            "Importance is estimated from how strongly anomaly populations differ from "
            "non-anomaly populations for each feature; it is a global signal, not a per-patient causal explanation."
        )
    else:
        st.info("Feature importance is not available for the current sample (for example, no anomalies detected).")

    st.divider()

    # Isolation Forest 
    st.subheader("Isolation Forest Anomalies")
    if_anomalies = anomaly_report['isolation_forest_anomalies']

    if if_anomalies:
        st.warning(f"{len(if_anomalies)} records flagged as multivariate outliers.")

        if_records = df.loc[if_anomalies].copy()

        # Explain *why* each row was flagged by comparing against clinical ranges
        if_records['flagged_vitals'] = if_records.apply(_flag_vitals, axis=1)

        # Column order: identifiers → explanation → vitals
        vital_cols = [c for c in CLINICAL_ANOMALY_RANGES if c in if_records.columns]
        id_cols    = [c for c in ['patient_id', 'name'] if c in if_records.columns]
        display_cols = id_cols + ['flagged_vitals'] + vital_cols

        with st.expander(f"View {len(if_anomalies)} Isolation Forest anomalies", expanded=True):
            st.dataframe(if_records[display_cols], use_container_width=True, hide_index=True)
    else:
        st.success("✅ No Isolation Forest anomalies detected!")


def show_monitoring_panel(df, dedup_summary=None):
    """Display centralized monitoring and health-status panel."""
    st.header("Monitoring Panel")
    st.markdown(
        """
        <div class="clinician-note">
            <strong>How to read this panel:</strong> Start with System Health, then review score drivers,
            and finally inspect alert explainability and embedded quality checks for context.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = calculate_system_health_metrics(df, dedup_summary=dedup_summary)

    st.subheader("System Health")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Data Quality Score", f"{metrics['data_quality_score']:.1f}/100")
    with c2:
        st.metric("Anomaly Rate", f"{metrics['anomaly_rate']:.2f}%")
    with c3:
        st.metric("Alerted Patient Rate", f"{metrics['alert_rate']:.2f}%")

    quality_color = "#27ae60" if metrics['data_quality_score'] >= 85 else "#f39c12" if metrics['data_quality_score'] >= 70 else "#e74c3c"
    anomaly_color = "#27ae60" if metrics['anomaly_rate'] <= 5 else "#f39c12" if metrics['anomaly_rate'] <= 15 else "#e74c3c"
    alert_color = "#27ae60" if metrics['alert_rate'] <= 10 else "#f39c12" if metrics['alert_rate'] <= 25 else "#e74c3c"

    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
    with gauge_col1:
        fig_quality = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=metrics['data_quality_score'],
                title={'text': "Quality Score"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': quality_color}},
            )
        )
        fig_quality.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_quality, use_container_width=True, config={'displayModeBar': False})

    with gauge_col2:
        fig_anomaly = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=metrics['anomaly_rate'],
                title={'text': "Anomaly Rate %"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': anomaly_color}},
            )
        )
        fig_anomaly.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_anomaly, use_container_width=True, config={'displayModeBar': False})

    with gauge_col3:
        fig_alert = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=metrics['alert_rate'],
                title={'text': "Alert Rate %"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': alert_color}},
            )
        )
        fig_alert.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_alert, use_container_width=True, config={'displayModeBar': False})

    st.subheader("Why These Scores Look This Way")
    st. markdown("""Duplicate Penalty Basis: Duplicates can distort analysis and lead to misleading insights if not handled properly. Physiological Violation Basis: Data entry errors or critical patient conditions that can impact the reliability of analysis.""")
    q = metrics['quality_breakdown']
    explain_col1, explain_col2 = st.columns(2)

    with explain_col1:
        quality_components = pd.DataFrame(
            [
                {'Component': 'Completeness', 'Value (%)': q['completeness_pct']},
                {'Component': 'Duplicate Penalty Basis', 'Value (%)': q['duplicate_rate_pct']}, # explain that this is the percentage of duplicate rows relative to total input rows, which is used as a penalty factor in the quality score calculation since duplicates can distort analysis and lead to misleading insights if not handled properly
                {'Component': 'Physio Violation Basis', 'Value (%)': q['phys_violation_rate_pct']}, # explain that this is the percentage of records with physiological limit violations relative to total records, which is used as a penalty factor in the quality score calculation since such violations may indicate data entry errors or critical patient conditions that can impact the reliability of analysis
            ]
        )
        st.dataframe(quality_components, use_container_width=True, hide_index=True)
        st.caption(
            f"Duplicate cleanup at ingestion: detected {q.get('duplicate_rows_detected', 0)} duplicate rows, "
            f"removed {q.get('duplicate_rows_removed', 0)} rows across "
            f"{q.get('duplicate_patient_ids_affected', 0)} patient IDs."
        )
        if q['missing_columns']:
            st.warning(f"Missing required columns: {', '.join(q['missing_columns'])}")
        else:
            st.caption("All required columns are present.")

    with explain_col2:
        anomaly_feature_importance = metrics['anomaly_report'].get('isolation_forest_feature_importance', {})
        if anomaly_feature_importance:
            top_features = pd.DataFrame(
                [
                    {'Feature': feature, 'Importance (%)': importance}
                    for feature, importance in list(anomaly_feature_importance.items())[:5]
                ]
            )
            st.dataframe(top_features, use_container_width=True, hide_index=True)
        else:
            st.caption("No anomaly feature-importance data available in current run.")

    st.divider()
    show_quality_metrics(df, dedup_summary=dedup_summary, embedded=True)


def show_clinical_alerts(df):
    """Display clinical alerts."""
    st.header("Clinical Alerts")
    st.markdown(
        """
        <div class="clinician-note">
            <strong>Suggested triage order:</strong> Critical alerts first, then warnings,
            with diagnosis/rule/recommendation used as transparent decision support context.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    engine = ClinicalRulesEngine()
    
    # Evaluate rules for all patients
    all_alerts = []
    for _, row in df.iterrows():
        patient_data = row.to_dict()
        alerts = engine.evaluate_all_rules(patient_data)
        all_alerts.extend(alerts)

    all_inferences = evaluate_inferences_for_dataset(df)
    inference_summary = engine.summarize_inferences(all_inferences)
    unique_inference_patients = len({str(inference.patient_id) for inference in all_inferences})
    
    # Alert summary derived from the collected alerts only
    summary = ClinicalRulesEngine.get_alerts_summary(all_alerts)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", summary['total_alerts'])
    with col2:
        st.metric("Critical", summary['severity_breakdown']['critical'])
    with col3:
        st.metric("Warning", summary['severity_breakdown']['warning'])
    with col4:
        st.metric("Info", summary['severity_breakdown']['info'])

    inference_col1, inference_col2 = st.columns(2)
    with inference_col1:
        st.metric("Diagnosis Flags", inference_summary.get('total_inferences', 0))
    with inference_col2:
        st.metric("Patients With Risks", unique_inference_patients)

    eval_col, refresh_col = st.columns([4, 1])
    with eval_col:
        st.caption(
            f"Last alert evaluation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            "(updates on each app rerun)."
        )
    with refresh_col:
        if st.button("Refresh Alerts"):
            st.rerun()

    st.subheader("Clinical Framework Context")
    st.markdown(
        "- ACC/AHA blood pressure staging reference: "
        "https://www.ahajournals.org/doi/10.1161/HYP.0000000000000065"
    )
    st.markdown(
        "- NEWS2 early warning framework reference: "
        "https://www.rcp.ac.uk/improving-care/resources/national-early-warning-score-news-2/"
    )
    st.markdown(
        "- Other rules and thresholds are based on common clinical knowledge and are intended for educational demonstration purposes only."
    )
    st.info(
        "Simplification note: this dashboard uses simplified threshold logic for monitoring and educational "
        "purposes. It is not a complete implementation of any guideline and is not intended for direct "
        "clinical decision-making."
    )

    st.subheader("Guideline-Based Diagnoses")
    top_risks = inference_summary.get('top_risks', [])
    if top_risks:
        for risk in top_risks:
            st.write(f"- {risk.get('summary_text')}")

        triggered_rules = inference_summary.get('triggered_rule_breakdown', {})
        if triggered_rules:
            triggered_rules_df = pd.DataFrame(
                [
                    {'Triggered Guideline Rule': rule, 'Count': count}
                    for rule, count in triggered_rules.items()
                ]
            ).sort_values(by='Count', ascending=False)
            st.dataframe(triggered_rules_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No guideline-based diagnosis flags were generated for the current dataset.")
    
    if all_alerts:
        # Framework breakdown
        framework_counts = {}
        for alert in all_alerts:
            framework = getattr(alert, 'framework', 'unspecified')
            framework_counts[framework] = framework_counts.get(framework, 0) + 1
        
        framework_df = pd.DataFrame(list(framework_counts.items()), 
                                   columns=['Framework', 'Count'])
        fig = px.pie(framework_df, values='Count', names='Framework',
                    title="Alerts by Clinical Framework")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Alert Breakdown by Type")
        
        alert_types = {}
        for alert in all_alerts:
            alert_types[alert.rule_name] = alert_types.get(alert.rule_name, 0) + 1
        
        alert_df = pd.DataFrame(list(alert_types.items()), 
                               columns=['Rule Name', 'Count'])
        
        fig = px.bar(alert_df, x='Rule Name', y='Count',
                    title="Alerts by Clinical Rule",
                    color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Alerts")
        st.caption("Structured view with filters and human-readable details.")

        patient_name_lookup = {}
        if 'patient_id' in df.columns and 'name' in df.columns:
            patient_name_lookup = (
                df[['patient_id', 'name']]
                .drop_duplicates(subset=['patient_id'])
                .set_index('patient_id')['name']
                .to_dict()
            )

        patient_inference_map = {}
        for inference in all_inferences:
            patient_inference_map.setdefault(str(inference.patient_id), []).append(inference)

        alert_rows = []
        for alert in all_alerts:
            values = getattr(alert, 'values', {}) or {}
            value_summary = ", ".join([f"{k}: {v}" for k, v in values.items()]) if values else "N/A"
            severity_text = alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity)
            patient_id = alert.patient_id
            patient_inferences = patient_inference_map.get(str(patient_id), [])
            diagnosis_summary = " | ".join(
                sorted({inference.diagnosis for inference in patient_inferences})
            ) if patient_inferences else "No additional diagnosis"
            recommendation_summary = " | ".join(
                [inference.recommendation for inference in patient_inferences]
            ) if patient_inferences else "Continue routine monitoring and reassess if symptoms evolve."
            rule_summary = " | ".join(
                [inference.triggered_guideline_rule for inference in patient_inferences]
            ) if patient_inferences else "N/A"
            explanation_summary = " | ".join(
                [inference.explanation for inference in patient_inferences]
            ) if patient_inferences else "No separate diagnosis inference for this case."
            alert_rows.append({
                'triggered_at': alert.triggered_at,
                'severity': severity_text,
                'patient_id': patient_id,
                'patient_name': patient_name_lookup.get(patient_id, 'N/A'),
                'rule': alert.rule_name,
                'framework': getattr(alert, 'framework', 'N/A'),
                'message': alert.message,
                'trigger_values': value_summary,
                'diagnosis': diagnosis_summary,
                'recommendation': recommendation_summary,
                'triggered_guideline_rule': rule_summary,
                'explanation': explanation_summary,
            })

        alerts_df = pd.DataFrame(alert_rows)
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        alerts_df['severity_rank'] = alerts_df['severity'].map(severity_order).fillna(99)
        alerts_df = alerts_df.sort_values(by=['severity_rank', 'triggered_at'], ascending=[True, False])

        severity_options = sorted(alerts_df['severity'].dropna().unique().tolist())
        framework_options = sorted(alerts_df['framework'].dropna().unique().tolist())

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_severity = st.multiselect(
                "Filter by Severity",
                options=severity_options,
                default=severity_options,
            )
        with filter_col2:
            selected_framework = st.multiselect(
                "Filter by Framework",
                options=framework_options,
                default=framework_options,
            )

        filtered_alerts = alerts_df[
            alerts_df['severity'].isin(selected_severity)
            & alerts_df['framework'].isin(selected_framework)
        ]

        st.dataframe(
            filtered_alerts[
                [
                    'triggered_at',
                    'severity',
                    'patient_id',
                    'patient_name',
                    'diagnosis',
                    'rule',
                    'framework',
                    'message',
                    'trigger_values',
                    'triggered_guideline_rule',
                    'recommendation',
                ]
            ],
            use_container_width=True,
            hide_index=True,
            # AUTOSIZE
            column_config={
                "recommendation": st.column_config.TextColumn(
                    "Recommendation",
                    width="large"
                )
            }
        )

        st.subheader("Clinician Guidance")
        st.info(
            "Triage flow: prioritize critical severity, verify trigger values, review the triggered guideline rule, "
            "then use the recommendation column as the suggested next step."
        )
        with st.expander("Quick safety notes"):
            st.markdown(
                "- Use this dashboard as decision support, not a standalone diagnosis tool.\n"
                "- Recheck vitals/labs when the reading does not match the clinical picture.\n"
                "- Escalate critical findings per your local protocol.\n"
                "- Document final clinical judgment in your standard clinical workflow."
            )
    else:
        st.success("✅ No clinical alerts triggered!")


def show_reports(df):
    """Display report generation interface."""
    st.header("Report Generation")
    st.caption("Generate exportable summaries for audits, handover, and multidisciplinary review.")
    
    generator = ClinicalReportGenerator()

    st.subheader("Generate Report")
    st.info(
        "Choose a report based on your goal: comprehensive for full clinical oversight, "
        "quality for data integrity audits, or summary statistics for fast descriptive analytics."
    )

    report_type = st.selectbox(
        "Report Type",
        ["Comprehensive Report", "Quality Report", "Summary Statistics"]
    )
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            if report_type == "Comprehensive Report":
                # Generate alerts for inclusion in report
                engine = ClinicalRulesEngine()
                all_alerts = []
                for _, row in df.iterrows():
                    alerts = engine.evaluate_all_rules(row.to_dict())
                    all_alerts.extend(alerts)
                
                report = generator.generate_comprehensive_report(
                    df, 
                    alerts=all_alerts,
                    include_statistics=True,
                    include_anomalies=True
                )
            elif report_type == "Quality Report":
                report = generator.generate_quality_report(df)
            else:  # Summary Statistics
                report = generator.generate_summary_statistics(df)
            
            st.success("✅ Report generated successfully!")

            st.subheader("Report Output")

            if report_type == "Comprehensive Report":
                executive = report.get('executive_summary', {})
                quality_metrics = report.get('quality_metrics', {})
                alerts = report.get('alerts', {})
                anomalies = report.get('anomalies', {})
                clinical_inferences = report.get('clinical_inferences', {})
                recommendations = report.get('recommendations', [])

                st.markdown("### Executive Summary")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Quality Grade", executive.get('quality_grade', 'N/A'))
                with c2:
                    st.metric("Missing Cells", executive.get('missing_cells', 0))
                with c3:
                    st.metric("IF Anomalies", executive.get('isolation_forest_anomalies', 0))
                with c4:
                    st.metric("Critical Alerts", executive.get('critical_alerts', 0))

                st.markdown("### Key Findings")
                findings = quality_metrics.get('key_issues', [])
                if findings:
                    for item in findings:
                        st.write(f"- {item}")
                else:
                    st.write("- No major data-quality issues detected.")

                risk_summaries = executive.get('potential_clinical_risks', [])
                st.markdown("### Potential Clinical Risks")
                if risk_summaries:
                    for risk in risk_summaries:
                        st.write(f"- {risk}")
                else:
                    st.write("- No guideline-based clinical risks detected.")

                inference_summary = clinical_inferences.get('summary', {}) if isinstance(clinical_inferences, dict) else {}
                patient_level_inferences = clinical_inferences.get('patient_level_inferences', []) if isinstance(clinical_inferences, dict) else []
                if patient_level_inferences:
                    st.markdown("### Guideline-Based Inference Details")
                    inference_df = pd.DataFrame(patient_level_inferences)
                    preferred_cols = [
                        col for col in [
                            'patient_id',
                            'diagnosis',
                            'severity',
                            'triggered_guideline_rule',
                            'recommendation',
                            'explanation',
                            'evidence',
                        ] if col in inference_df.columns
                    ]
                    st.dataframe(inference_df[preferred_cols], use_container_width=True, hide_index=True)

                triggered_rules = inference_summary.get('triggered_rule_breakdown', {})
                if triggered_rules:
                    st.markdown("### Triggered Guideline Rules")
                    triggered_rules_df = pd.DataFrame(
                        [
                            {'Guideline Rule': rule, 'Count': count}
                            for rule, count in triggered_rules.items()
                        ]
                    ).sort_values(by='Count', ascending=False)
                    st.dataframe(triggered_rules_df, use_container_width=True, hide_index=True)

                alert_breakdown = (alerts.get('severity_breakdown', {}) if isinstance(alerts, dict) else {})
                if alert_breakdown:
                    st.markdown("### Alert Severity Breakdown")
                    alert_df = pd.DataFrame(
                        [
                            {'Severity': k.title(), 'Count': v}
                            for k, v in alert_breakdown.items()
                        ]
                    ).sort_values(by='Count', ascending=False)
                    st.dataframe(alert_df, use_container_width=True, hide_index=True)

                if isinstance(anomalies, dict) and anomalies.get('vital_sign_anomalies'):
                    st.markdown("### Rule-Based Anomaly Hotspots")
                    anomaly_df = pd.DataFrame(
                        [
                            {'Vital Sign': k, 'Anomaly Count': len(v)}
                            for k, v in anomalies.get('vital_sign_anomalies', {}).items()
                        ]
                    ).sort_values(by='Anomaly Count', ascending=False)
                    st.dataframe(anomaly_df, use_container_width=True, hide_index=True)

                st.markdown("### Recommended Actions")
                for rec in recommendations:
                    st.write(f"- {rec}")

            elif report_type == "Quality Report":
                st.markdown("### Quality Summary")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Quality Grade", report.get('quality_grade', 'N/A'))
                with c2:
                    st.metric("Completeness", f"{report.get('completeness_percentage', 0)}%")
                with c3:
                    st.metric("Missing Cells", report.get('missing_cells', 0))
                with c4:
                    st.metric("Duplicate IDs", report.get('duplicate_patient_ids_count', 0))

                missing_pct = report.get('missing_values_pct_by_column', {})
                if missing_pct:
                    st.markdown("### Missing Values by Column")
                    missing_df = pd.DataFrame(
                        [
                            {'Column': col, 'Missing %': pct}
                            for col, pct in missing_pct.items()
                        ]
                    ).sort_values(by='Missing %', ascending=False)
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)

                phys = report.get('physiological_limit_violations', {})
                if phys:
                    st.markdown("### Physiological Limit Violations")
                    phys_df = pd.DataFrame(
                        [
                            {'Field': col, 'Violation Count': len(indices)}
                            for col, indices in phys.items()
                        ]
                    ).sort_values(by='Violation Count', ascending=False)
                    st.dataframe(phys_df, use_container_width=True, hide_index=True)

                issues = report.get('key_issues', [])
                if issues:
                    st.markdown("### Key Issues")
                    for issue in issues:
                        st.write(f"- {issue}")

            else:  # Summary Statistics
                st.markdown("### Summary Statistics")
                if report:
                    stats_rows = []
                    for metric, values in report.items():
                        stats_rows.append({
                            'Metric': metric,
                            'Mean': values.get('mean'),
                            'Median': values.get('median'),
                            'Std': values.get('std'),
                            'Min': values.get('min'),
                            'Max': values.get('max'),
                            'Count': values.get('count'),
                        })
                    stats_df = pd.DataFrame(stats_rows)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No numeric fields available for summary statistics.")

            import json
            col1, col2 = st.columns(2)
            with col1:
                json_str = json.dumps(report, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            with col2:
                if report_type == "Summary Statistics":
                    csv_data = pd.DataFrame(
                        [
                            {
                                'metric': metric,
                                **values,
                            }
                            for metric, values in report.items()
                        ]
                    ).to_csv(index=False)
                else:
                    csv_data = pd.json_normalize(report, sep='__').to_csv(index=False)

                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()
