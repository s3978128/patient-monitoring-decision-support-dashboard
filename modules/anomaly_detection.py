"""
Anomaly Detection Module
Detects anomalies and outliers in clinical data using various algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Detects anomalies in clinical data."""
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in the dataset
        """
        self.contamination = contamination
    
    def detect_statistical_outliers(self, data: pd.Series, 
                                    method: str = 'zscore',
                                    threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers using statistical methods.
        
        Args:
            data: Series containing numerical data
            method: Method to use ('zscore' or 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean array indicating outliers
        """
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data.dropna()))
            outliers = np.zeros(len(data), dtype=bool)
            outliers[data.notna()] = z_scores > threshold
            return outliers
        
        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def detect_vital_sign_anomalies(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Detect clinically significant anomalies in vital signs.
        
        Uses narrow clinical ranges to identify values that may require
        medical attention. These are more restrictive than data quality
        checks (see quality_checks.check_physiological_limits).
        
        Args:
            df: DataFrame containing vital signs
            
        Returns:
            Dictionary mapping vital sign columns to indices of anomalous records
        """
        anomalies = {}
        
        # Narrow ranges for clinical anomalies: catches concerning values
        # that may require medical intervention
        vital_sign_ranges = {
            'heart_rate': (40, 120),
            'blood_pressure_systolic': (80, 180),
            'blood_pressure_diastolic': (50, 100),
            'temperature': (35.0, 39.0),
            'respiratory_rate': (8, 30),
            'oxygen_saturation': (85, 100)
        }
        
        # Check for out-of-range values
        
        for column, (min_val, max_val) in vital_sign_ranges.items():
            if column in df.columns:
                anomaly_mask = (df[column] < min_val) | (df[column] > max_val)
                if anomaly_mask.any():
                    anomalies[column] = df[anomaly_mask].index.tolist()
        
        return anomalies
    
    def detect_temporal_anomalies(self, df: pd.DataFrame, 
                                  value_column: str,
                                  time_column: str,
                                  window: int = 5) -> List[int]:
        """
        Detect anomalies based on temporal patterns.
        
        Args:
            df: DataFrame with time series data
            value_column: Column containing values to check
            time_column: Column containing timestamps
            window: Rolling window size for comparison
            
        Returns:
            List of indices with temporal anomalies
        """
        df_sorted = df.sort_values(time_column)
        rolling_mean = df_sorted[value_column].rolling(window=window, center=True).mean()
        rolling_std = df_sorted[value_column].rolling(window=window, center=True).std()
        
        deviations = np.abs(df_sorted[value_column] - rolling_mean)
        threshold = 2 * rolling_std
        
        anomaly_mask = deviations > threshold
        return df_sorted[anomaly_mask].index.tolist()

    def detect_isolation_forest_anomalies(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> List[int]:
        """
        Detect multivariate anomalies using Isolation Forest.

        Args:
            df: DataFrame containing vital signs/features
            feature_columns: Optional feature columns to include. If None,
                numeric columns are used automatically.
            random_state: Random state for reproducibility

        Returns:
            List of indices predicted as anomalies
        """
        if df.empty:
            return []

        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if not feature_columns:
            return []

        available_features = [col for col in feature_columns if col in df.columns]
        if not available_features:
            return []

        model_input = df[available_features].copy()

        # Isolation Forest cannot handle NaN directly; median fill keeps distribution stable.
        model_input = model_input.fillna(model_input.median(numeric_only=True))

        if model_input.empty:
            return []

        model = IsolationForest(
            contamination=self.contamination,
            random_state=random_state,
            n_estimators=200,
        )

        predictions = model.fit_predict(model_input)
        return df.index[predictions == -1].tolist()
    
    def generate_anomaly_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive anomaly detection report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing anomaly detection results
        """
        report = {
            'total_records': len(df),
            'vital_sign_anomalies': self.detect_vital_sign_anomalies(df),
            'isolation_forest_anomalies': self.detect_isolation_forest_anomalies(df),
            'timestamp': pd.Timestamp.now()
        }
        
        # Count total anomalies
        total_anomalies = sum(len(indices) for indices in report['vital_sign_anomalies'].values())
        report['total_anomalies'] = total_anomalies
        report['anomaly_rate'] = round(total_anomalies / len(df) * 100, 2) if len(df) > 0 else 0
        report['isolation_forest_anomaly_count'] = len(report['isolation_forest_anomalies'])
        report['isolation_forest_anomaly_rate'] = (
            round(len(report['isolation_forest_anomalies']) / len(df) * 100, 2)
            if len(df) > 0 else 0
        )
        
        return report


if __name__ == "__main__":
    # Example usage
    detector = AnomalyDetector()
    sample_data = pd.DataFrame({
        'heart_rate': [72, 85, 150, 68, 75],  # 150 is anomalous
        'temperature': [36.8, 37.1, 40.5, 36.9, 37.0]  # 40.5 is anomalous
    })
    
    report = detector.generate_anomaly_report(sample_data)
    print(report)
