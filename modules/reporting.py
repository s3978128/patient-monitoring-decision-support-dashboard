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
                'generated_at': datetime.now().isoformat()
            }
        
        severity_counts = {}
        alert_types = {}
        framework_breakdown = {}
        patient_ids = set()
        
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
        
        return {
            'total_alerts': len(alerts),
            'patients_affected': len(patient_ids),
            'severity_breakdown': severity_counts,
            'alert_types': alert_types,
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

        duplicate_count = 0
        duplicate_ids: List[Any] = []
        if 'patient_id' in df.columns:
            duplicate_count, duplicate_ids = checker.check_duplicates(df, 'patient_id')

        return {
            'total_records': len(df),
            'total_fields': len(df.columns),
            'missing_cells': missing_cells,
            'missing_values_pct_by_column': checker.check_missing_values(df),
            'completeness_percentage': round(quality_score, 2),
            'column_completeness': {
                col: round((df[col].notna().sum() / len(df) * 100), 2) if len(df) > 0 else 0.0
                for col in df.columns
            },
            'duplicate_patient_ids_count': duplicate_count,
            'duplicate_patient_ids': duplicate_ids,
            'physiological_limit_violations': checker.check_physiological_limits(df),
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
        report['quality_metrics'] = self.generate_quality_report(df)
        
        # Add summary statistics if requested
        if include_statistics:
            report['summary_statistics'] = self.generate_summary_statistics(df)
        
        # Add alert summary if provided
        if alerts:
            report['alerts'] = self.generate_alert_report(alerts)

        # Add anomaly summary if requested
        if include_anomalies:
            try:
                from modules.anomaly_detection import AnomalyDetector
            except ModuleNotFoundError:
                from anomaly_detection import AnomalyDetector

            detector = AnomalyDetector()
            report['anomalies'] = detector.generate_anomaly_report(df)
        
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
