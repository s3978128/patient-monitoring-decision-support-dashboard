"""
Reporting Module
Generates reports and visualizations for clinical monitoring data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class ClinicalReportGenerator:
    """Generates reports for clinical monitoring system."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.report_history = []

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        """Convert common pandas/numpy values to JSON-serializable Python values."""
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        return value

    @classmethod
    def _safe_round(cls, value: Any, digits: int = 2) -> Optional[float]:
        """Round numeric values while returning None for missing/invalid entries."""
        if pd.isna(value):
            return None
        return round(float(value), digits)

    @staticmethod
    def _severity_rank(severity: str) -> int:
        """Sort helper for severity labels."""
        order = {'critical': 0, 'warning': 1, 'info': 2}
        return order.get(str(severity).lower(), 99)

    def _build_recommendations(
        self,
        quality_report: Dict[str, Any],
        anomaly_report: Optional[Dict[str, Any]] = None,
        alert_report: Optional[Dict[str, Any]] = None,
        inference_report: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Build concise action-oriented recommendations."""
        recommendations: List[str] = []

        missing_cells = int(quality_report.get('missing_cells', 0))
        duplicate_count = int(quality_report.get('duplicate_patient_ids_count', 0))
        phys_violations = quality_report.get('physiological_limit_violations', {}) or {}
        phys_violation_count = sum(len(v) for v in phys_violations.values())

        if missing_cells > 0:
            recommendations.append(
                "Address missing fields in source data feeds and add input validation at data entry points."
            )
        if duplicate_count > 0:
            recommendations.append(
                "Resolve duplicate patient identifiers and enforce unique constraints in upstream systems."
            )
        if phys_violation_count > 0:
            recommendations.append(
                "Review out-of-physiological-range measurements for sensor faults or data-entry errors."
            )

        if anomaly_report:
            if_count = int(anomaly_report.get('isolation_forest_anomaly_count', 0))
            if if_count > 0:
                recommendations.append(
                    "Triage Isolation Forest outliers to identify multivariate deterioration patterns."
                )

        if alert_report:
            sev = alert_report.get('severity_breakdown', {}) or {}
            if int(sev.get('critical', 0)) > 0:
                recommendations.append(
                    "Escalate critical alerts immediately according to local clinical governance workflows."
                )

        if inference_report:
            diagnosis_breakdown = inference_report.get('diagnosis_breakdown', {}) or {}
            if int(diagnosis_breakdown.get('Hypertension risk', 0)) > 0:
                recommendations.append(
                    "Recheck elevated blood pressure readings and review patients flagged for hypertension risk."
                )
            if int(diagnosis_breakdown.get('Diabetes risk', 0)) > 0:
                recommendations.append(
                    "Arrange confirmatory glucose follow-up for patients flagged with diabetes risk."
                )
            if int(diagnosis_breakdown.get('Blood-pressure drop flagged', 0)) > 0:
                recommendations.append(
                    "Prioritize reassessment of patients with low blood pressure flags for possible instability."
                )

        if not recommendations:
            recommendations.append(
                "No high-priority data quality or clinical risk signals detected; continue routine monitoring."
            )

        return recommendations
    
    def generate_summary_statistics(self, df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None) -> Dict:
        """
        Generate summary statistics for numerical columns.
        
        Args:
            df: DataFrame containing clinical data
            columns: List of columns to include (None for all numerical columns)
            
        Returns:
            Dictionary containing summary statistics
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        summary = {}
        for col in columns:
            if col in df.columns:
                summary[col] = {
                    'mean': self._safe_round(df[col].mean(), 2),
                    'median': self._safe_round(df[col].median(), 2),
                    'std': self._safe_round(df[col].std(), 2),
                    'min': self._safe_round(df[col].min(), 2),
                    'max': self._safe_round(df[col].max(), 2),
                    'count': int(df[col].count())
                }
        
        return summary
    
    def generate_alert_report(self, alerts: List[Any]) -> Dict:
        """
        Generate report from clinical alerts.
        
        Args:
            alerts: List of clinical alerts
            
        Returns:
            Dictionary containing alert report
        """
        if not alerts:
            return {
                'total_alerts': 0,
                'severity_breakdown': {},
                'alert_types': {},
                'top_rules': [],
                'top_patients': [],
                'generated_at': datetime.now().isoformat()
            }
        
        severity_counts = {}
        alert_types = {}
        framework_breakdown = {}
        patient_ids = set()
        patient_alert_counts: Dict[str, int] = {}
        
        for alert in alerts:
            # Count by severity
            severity = getattr(alert, 'severity', None)
            if severity:
                severity_key = severity.value if hasattr(severity, 'value') else str(severity)
                severity_counts[severity_key] = severity_counts.get(severity_key, 0) + 1
            
            # Count by rule/type
            rule_name = getattr(alert, 'rule_name', 'Unknown')
            alert_types[rule_name] = alert_types.get(rule_name, 0) + 1

            framework = getattr(alert, 'framework', 'unspecified')
            framework_breakdown[framework] = framework_breakdown.get(framework, 0) + 1

            patient_id = getattr(alert, 'patient_id', None)
            if patient_id is not None:
                patient_ids.add(patient_id)
                patient_alert_counts[str(patient_id)] = patient_alert_counts.get(str(patient_id), 0) + 1

        top_rules = sorted(
            [{'rule_name': k, 'count': v} for k, v in alert_types.items()],
            key=lambda x: x['count'],
            reverse=True,
        )[:5]
        top_patients = sorted(
            [{'patient_id': k, 'alert_count': v} for k, v in patient_alert_counts.items()],
            key=lambda x: x['alert_count'],
            reverse=True,
        )[:5]
        
        return {
            'total_alerts': len(alerts),
            'patients_affected': len(patient_ids),
            'severity_breakdown': severity_counts,
            'alert_types': alert_types,
            'top_rules': top_rules,
            'top_patients': top_patients,
            'framework_breakdown': framework_breakdown,
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_time_series_report(self, df: pd.DataFrame, 
                                   time_column: str,
                                   value_columns: List[str],
                                   frequency: str = 'D') -> Dict:
        """
        Generate time series analysis report.
        
        Args:
            df: DataFrame with time series data
            time_column: Column containing timestamps
            value_columns: Columns to analyze
            frequency: Resampling frequency ('D' for daily, 'H' for hourly, etc.)
            
        Returns:
            Dictionary containing time series metrics
        """
        if time_column not in df.columns:
            return {'error': f'Time column {time_column} not found'}
        
        df_copy = df.copy()
        df_copy[time_column] = pd.to_datetime(df_copy[time_column], errors='coerce')
        df_copy = df_copy.dropna(subset=[time_column])

        if df_copy.empty:
            return {
                'frequency': frequency,
                'date_range': {'start': None, 'end': None},
                'metrics': {},
                'error': f'No valid timestamps available in {time_column}'
            }

        df_copy = df_copy.set_index(time_column)
        
        report = {
            'frequency': frequency,
            'date_range': {
                'start': df_copy.index.min().isoformat(),
                'end': df_copy.index.max().isoformat()
            },
            'metrics': {}
        }
        
        for col in value_columns:
            if col in df_copy.columns:
                resampled = df_copy[col].resample(frequency).agg(['mean', 'count'])
                report['metrics'][col] = {
                    'average_value': self._safe_round(resampled['mean'].mean(), 2),
                    'total_readings': int(resampled['count'].sum())
                }
        
        return report
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            from modules.quality_checks import DataQualityChecker
        except ModuleNotFoundError:
            from quality_checks import DataQualityChecker

        checker = DataQualityChecker()
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = int(df.isnull().sum().sum())
        quality_score = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
        is_complete, missing_columns = checker.check_data_completeness(
            df,
            checker.REQUIRED_COLUMNS,
        )

        duplicate_count = 0
        duplicate_ids: List[Any] = []
        if 'patient_id' in df.columns:
            duplicate_count, duplicate_ids = checker.check_duplicates(df, 'patient_id')

        physiological_violations = checker.check_physiological_limits(df)
        total_physiological_violations = sum(len(v) for v in physiological_violations.values())

        key_issues = []
        if missing_cells > 0:
            key_issues.append(f"Missing cells detected: {missing_cells}")
        if duplicate_count > 0:
            key_issues.append(f"Duplicate patient IDs detected: {duplicate_count}")
        if total_physiological_violations > 0:
            key_issues.append(
                f"Physiological limit violations detected: {total_physiological_violations}"
            )

        quality_grade = 'A'
        if missing_cells > 0 or duplicate_count > 0 or total_physiological_violations > 0:
            quality_grade = 'B'
        if missing_cells > 20 or duplicate_count > 5 or total_physiological_violations > 10:
            quality_grade = 'C'
        if missing_cells > 50 or duplicate_count > 15 or total_physiological_violations > 20:
            quality_grade = 'D'

        return {
            'total_records': len(df),
            'total_fields': len(df.columns),
            'missing_cells': missing_cells,
            'missing_values_pct_by_column': checker.check_missing_values(df),
            'completeness_percentage': round(quality_score, 2),
            'quality_grade': quality_grade,
            'key_issues': key_issues,
            'schema_complete': is_complete,
            'missing_columns': missing_columns,
            'column_completeness': {
                col: round((df[col].notna().sum() / len(df) * 100), 2) if len(df) > 0 else 0.0
                for col in df.columns
            },
            'duplicate_patient_ids_count': duplicate_count,
            'duplicate_patient_ids': duplicate_ids,
            'physiological_limit_violations': physiological_violations,
            'total_physiological_violations': total_physiological_violations,
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_comprehensive_report(self, df: pd.DataFrame, 
                                     alerts: Optional[List] = None,
                                     include_statistics: bool = True,
                                     include_anomalies: bool = True) -> Dict:
        """
        Generate comprehensive monitoring report.
        
        Args:
            df: DataFrame containing clinical data
            alerts: Optional list of alerts
            include_statistics: Whether to include summary statistics
            include_anomalies: Whether to include anomaly analysis
            
        Returns:
            Dictionary containing comprehensive report
        """
        report = {
            'report_id': f"RPT{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'data_overview': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'columns': df.columns.tolist()
            }
        }
        
        # Add quality metrics
        quality_report = self.generate_quality_report(df)
        report['quality_metrics'] = quality_report
        
        # Add summary statistics if requested
        if include_statistics:
            report['summary_statistics'] = self.generate_summary_statistics(df)
        
        # Add alert summary if provided
        if alerts:
            report['alerts'] = self.generate_alert_report(alerts)

        try:
            from modules.clinical_rules import ClinicalRulesEngine
        except ModuleNotFoundError:
            from clinical_rules import ClinicalRulesEngine

        engine = ClinicalRulesEngine()
        clinical_inferences = []
        for _, row in df.iterrows():
            clinical_inferences.extend(engine.evaluate_guideline_inferences(row.to_dict()))

        report['clinical_inferences'] = {
            'summary': engine.summarize_inferences(clinical_inferences),
            'patient_level_inferences': [
                {
                    'patient_id': inference.patient_id,
                    'diagnosis': inference.diagnosis,
                    'recommendation': inference.recommendation,
                    'triggered_guideline_rule': inference.triggered_guideline_rule,
                    'evidence': {
                        key: self._to_serializable(value)
                        for key, value in inference.evidence.items()
                    },
                    'explanation': inference.explanation,
                    'severity': inference.severity,
                }
                for inference in clinical_inferences
            ],
        }

        # Add anomaly summary if requested
        if include_anomalies:
            try:
                from modules.anomaly_detection import AnomalyDetector
            except ModuleNotFoundError:
                from anomaly_detection import AnomalyDetector

            detector = AnomalyDetector()
            report['anomalies'] = detector.generate_anomaly_report(df)

        alert_report = report.get('alerts')
        anomaly_report = report.get('anomalies')
        inference_report = (report.get('clinical_inferences', {}) or {}).get('summary', {})

        potential_risks = [
            item.get('summary_text')
            for item in inference_report.get('top_risks', [])
            if item.get('summary_text')
        ]

        report['executive_summary'] = {
            'quality_grade': quality_report.get('quality_grade'),
            'records_analyzed': len(df),
            'missing_cells': quality_report.get('missing_cells', 0),
            'duplicate_patient_ids': quality_report.get('duplicate_patient_ids_count', 0),
            'physiological_violations': quality_report.get('total_physiological_violations', 0),
            'isolation_forest_anomalies': (
                anomaly_report.get('isolation_forest_anomaly_count', 0)
                if isinstance(anomaly_report, dict) else 0
            ),
            'critical_alerts': (
                (alert_report.get('severity_breakdown', {}) or {}).get('critical', 0)
                if isinstance(alert_report, dict) else 0
            ),
            'potential_clinical_risks': potential_risks,
            'patients_with_inference_flags': sum(
                item.get('patients_affected', 0)
                for item in inference_report.get('top_risks', [])
            ),
        }
        report['recommendations'] = self._build_recommendations(
            quality_report=quality_report,
            anomaly_report=anomaly_report if isinstance(anomaly_report, dict) else None,
            alert_report=alert_report if isinstance(alert_report, dict) else None,
            inference_report=inference_report if isinstance(inference_report, dict) else None,
        )
        
        # Store in history
        self.report_history.append({
            'report_id': report['report_id'],
            'timestamp': report['generated_at']
        })
        
        return report
    
    def export_report_to_json(self, report: Dict, filepath: str):
        """
        Export report to JSON file.
        
        Args:
            report: Report dictionary
            filepath: Path to save JSON file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=self._to_serializable)
    
    def export_report_to_html(self, report: Dict, filepath: str):
        """
        Export report to HTML file.
        
        Args:
            report: Report dictionary
            filepath: Path to save HTML file
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clinical Monitoring Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .metric {{ background-color: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Clinical Monitoring Report</h1>
            <div class="metric">
                <strong>Report ID:</strong> {report.get('report_id', 'N/A')}<br>
                <strong>Generated:</strong> {report.get('generated_at', 'N/A')}
            </div>
            <pre>{json.dumps(report, indent=2, default=self._to_serializable)}</pre>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)


if __name__ == "__main__":
    # Example usage
    generator = ClinicalReportGenerator()
    
    sample_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003'],
        'heart_rate': [72, 85, 68],
        'temperature': [36.8, 37.1, 36.9]
    })
    
    report = generator.generate_comprehensive_report(sample_data)
    print(json.dumps(report, indent=2, default=generator._to_serializable))
