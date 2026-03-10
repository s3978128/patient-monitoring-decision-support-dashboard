"""
Clinical CDSS Monitoring Dashboard
Interactive web dashboard for monitoring clinical decision support system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from modules.data_simulator import PatientDataSimulator
from modules.quality_checks import DataQualityChecker
from modules.anomaly_detection import AnomalyDetector
from modules.clinical_rules import ClinicalRulesEngine, AlertSeverity
from modules.reporting import ClinicalReportGenerator


# Page configuration
st.set_page_config(
    page_title="Clinical CDSS Monitoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_data(n_patients=100, anomaly_rate=0.15, severe_error_rate=0.05, 
             duplicate_rate=0.1, missing_column_rate=0.0):
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
    
    if df is None or not use_existing:
        # Generate sample data with error injection
        simulator = PatientDataSimulator(
            seed=42, 
            anomaly_rate=anomaly_rate,
            severe_error_rate=severe_error_rate
        )
        df = simulator.generate_dataset(
            n_patients=n_patients,
            duplicate_rate=duplicate_rate,
            missing_column_rate=missing_column_rate
        )
        
        # Save the generated data
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(data_path, index=False, encoding='utf-8')
            st.info(f"Generated and saved new data with {len(df)} records")
        except Exception as e:
            st.warning(f"Could not save data: {e}")
    
    return df


def main():
    """Main dashboard application."""
    
    # Header
    st.title("🏥 Clinical CDSS Monitoring Dashboard")
    st.markdown("Real-time monitoring and quality assurance for Clinical Decision Support Systems")
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Data generation options
    st.sidebar.subheader("Data Generation Settings")
    n_patients = st.sidebar.slider("Number of Patients", 10, 500, 100)
    anomaly_rate = st.sidebar.slider("Clinical Anomaly Rate", 0.0, 0.5, 0.15)
    severe_error_rate = st.sidebar.slider("Severe Error Rate", 0.0, 0.2, 0.05)
    duplicate_rate = st.sidebar.slider("Duplicate Rate", 0.0, 0.3, 0.1)
    missing_column_rate = st.sidebar.slider("Missing Column Rate", 0.0, 0.3, 0.0)
    
    # Load data with settings
    df = load_data(
        n_patients=n_patients,
        anomaly_rate=anomaly_rate,
        severe_error_rate=severe_error_rate,
        duplicate_rate=duplicate_rate,
        missing_column_rate=missing_column_rate
    )
    
    # Sidebar options
    st.sidebar.markdown("---")
    view_mode = st.sidebar.selectbox(
        "Select View",
        ["Overview", "Quality Metrics", "Anomaly Detection", "Clinical Alerts", "Reports"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Records", len(df))
    st.sidebar.metric("Total Columns", len(df.columns))
    
    # Main content based on view mode
    if view_mode == "Overview":
        show_overview(df)
    elif view_mode == "Quality Metrics":
        show_quality_metrics(df)
    elif view_mode == "Anomaly Detection":
        show_anomaly_detection(df)
    elif view_mode == "Clinical Alerts":
        show_clinical_alerts(df)
    elif view_mode == "Reports":
        show_reports(df)


def show_overview(df):
    """Display overview dashboard."""
    st.header("System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        avg_age = df['age'].mean() if 'age' in df.columns else 0
        st.metric("Average Age", f"{avg_age:.1f}")
    with col3:
        if 'heart_rate' in df.columns:
            avg_hr = df['heart_rate'].mean()
            st.metric("Avg Heart Rate", f"{avg_hr:.0f} bpm")
    with col4:
        if 'temperature' in df.columns:
            avg_temp = df['temperature'].mean()
            st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
    
    # Visualizations
    st.subheader("Data Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'age' in df.columns:
            fig = px.histogram(df, x='age', nbins=20, title="Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'gender' in df.columns:
            gender_counts = df['gender'].value_counts()
            fig = px.pie(values=gender_counts.values, names=gender_counts.index, 
                        title="Gender Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent data
    st.subheader("Recent Patient Records")
    st.dataframe(df.head(10), use_container_width=True)


def show_quality_metrics(df):
    """Display data quality metrics."""
    st.header("Data Quality Metrics")
    
    checker = DataQualityChecker()
    quality_report = checker.generate_quality_report(df)
    
    # Quality score
    missing_pct = quality_report.get('missing_values', {})
    total_missing = sum(missing_pct.values()) if isinstance(missing_pct, dict) else 0
    
    # Check for duplicates and physiological violations
    duplicate_count = 0
    duplicate_ids = []
    if 'patient_id' in df.columns:
        duplicate_count, duplicate_ids = checker.check_duplicates(df, 'patient_id')
    
    phys_violations = checker.check_physiological_limits(df)
    total_phys_violations = sum(len(v) for v in phys_violations.values())
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", quality_report['total_records'])
    with col2:
        st.metric("Total Columns", quality_report['total_columns'])
    with col3:
        st.metric("Duplicate IDs", duplicate_count)
    with col4:
        st.metric("Physiological Violations", total_phys_violations)
    
    # Missing values details
    st.subheader("Missing Values by Column")
    missing_values = checker.check_missing_values(df)
    
    if missing_values:
        missing_df = pd.DataFrame(list(missing_values.items()), 
                                 columns=['Column', 'Missing %'])
        fig = px.bar(missing_df, x='Column', y='Missing %', 
                    title="Missing Values Percentage by Column")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values detected!")
    
    # Duplicate records
    if duplicate_count > 0:
        st.subheader(f"Duplicate Patient IDs ({duplicate_count})")
        st.warning(f"Found {duplicate_count} duplicate patient IDs")
        with st.expander("View Duplicate IDs"):
            st.write(duplicate_ids[:20])  # Show first 20
    
    # Physiological limit violations
    if total_phys_violations > 0:
        st.subheader("Physiological Limit Violations")
        st.error(f"Found {total_phys_violations} values outside physiological limits")
        
        for vital_sign, indices in phys_violations.items():
            with st.expander(f"{vital_sign} - {len(indices)} violations"):
                if vital_sign in df.columns:
                    violation_records = df.loc[indices, ['patient_id', vital_sign]]
                    st.dataframe(violation_records, use_container_width=True)
    else:
        st.success("✅ No physiological limit violations!")
    
    # Data types
    st.subheader("Data Types")
    dtypes_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str)
    })
    st.dataframe(dtypes_df, use_container_width=True)


def show_anomaly_detection(df):
    """Display anomaly detection results."""
    st.header("Anomaly Detection")
    
    contamination = st.slider("Isolation Forest Contamination Rate", 0.01, 0.3, 0.1)
    detector = AnomalyDetector(contamination=contamination)
    anomaly_report = detector.generate_anomaly_report(df)
    
    # Anomaly summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records Analyzed", anomaly_report['total_records'])
    with col2:
        st.metric("Rule-Based Anomalies", anomaly_report['total_anomalies'])
    with col3:
        st.metric("Isolation Forest Anomalies", anomaly_report['isolation_forest_anomaly_count'])
    with col4:
        st.metric("IF Anomaly Rate", f"{anomaly_report['isolation_forest_anomaly_rate']:.2f}%")
    
    # Anomalies by type
    st.subheader("Rule-Based Anomalies by Vital Sign")
    
    vital_anomalies = anomaly_report['vital_sign_anomalies']
    
    if vital_anomalies:
        anomaly_counts = {k: len(v) for k, v in vital_anomalies.items()}
        anomaly_df = pd.DataFrame(list(anomaly_counts.items()), 
                                 columns=['Vital Sign', 'Anomaly Count'])
        
        fig = px.bar(anomaly_df, x='Vital Sign', y='Anomaly Count',
                    title="Rule-Based Anomaly Count by Vital Sign",
                    color_discrete_sequence=['#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Details
        st.subheader("Anomaly Details")
        for vital_sign, indices in vital_anomalies.items():
            with st.expander(f"{vital_sign} - {len(indices)} anomalies"):
                if vital_sign in df.columns:
                    anomalous_records = df.loc[indices, ['patient_id', vital_sign]]
                    st.dataframe(anomalous_records, use_container_width=True)
    else:
        st.success("✅ No rule-based anomalies detected in vital signs!")
    
    # Isolation Forest Anomalies
    st.subheader("Isolation Forest Anomalies")
    if_anomalies = anomaly_report['isolation_forest_anomalies']
    
    if if_anomalies:
        st.warning(f"Found {len(if_anomalies)} multivariate anomalies using Isolation Forest")
        with st.expander(f"View {len(if_anomalies)} Isolation Forest Anomalies"):
            if_records = df.loc[if_anomalies]
            st.dataframe(if_records, use_container_width=True)
    else:
        st.success("✅ No Isolation Forest anomalies detected!")


def show_clinical_alerts(df):
    """Display clinical alerts."""
    st.header("Clinical Alerts")
    
    engine = ClinicalRulesEngine()
    
    # Evaluate rules for all patients
    all_alerts = []
    for _, row in df.iterrows():
        patient_data = row.to_dict()
        alerts = engine.evaluate_all_rules(patient_data)
        all_alerts.extend(alerts)
    
    # Alert summary
    summary = engine.get_alerts_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", summary['total_alerts'])
    with col2:
        st.metric("Critical", summary['severity_breakdown']['critical'])
    with col3:
        st.metric("Warning", summary['severity_breakdown']['warning'])
    with col4:
        st.metric("Info", summary['severity_breakdown']['info'])
    
    # Alert breakdown
    if all_alerts:
        # Framework breakdown
        st.subheader("Alert Breakdown by Framework")
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
        
        # Recent alerts with framework info
        st.subheader("Recent Alerts")
        
        for alert in all_alerts[:20]:  # Show first 20
            severity_color = {
                AlertSeverity.CRITICAL: "🔴",
                AlertSeverity.WARNING: "🟡",
                AlertSeverity.INFO: "🔵"
            }
            
            icon = severity_color.get(alert.severity, "⚪")
            framework = getattr(alert, 'framework', 'N/A')
            st.write(f"{icon} **{alert.rule_name}** ({framework}) - Patient {alert.patient_id}: {alert.message}")
    else:
        st.success("✅ No clinical alerts triggered!")


def show_reports(df):
    """Display report generation interface."""
    st.header("Report Generation")
    
    generator = ClinicalReportGenerator()
    
    # Report options
    st.subheader("Generate Report")
    
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
            
            # Display report
            st.json(report)
            
            # Download options
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
                # Create a flattened version for CSV
                try:
                    csv_data = pd.DataFrame([report]).to_csv(index=False)
                except:
                    csv_data = str(report)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()
