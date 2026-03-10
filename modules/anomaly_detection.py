"""
Anomaly Detection Module
Detects anomalies and outliers in clinical data using various algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.ensemble import IsolationForest

try:
    from modules.clinical_thresholds import CLINICAL_ANOMALY_RANGES, VITAL_SIGN_FEATURES
except ImportError:
    from clinical_thresholds import CLINICAL_ANOMALY_RANGES, VITAL_SIGN_FEATURES


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
        
        # Check for out-of-range values
        for column, (min_val, max_val) in CLINICAL_ANOMALY_RANGES.items():
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
            feature_columns: Columns to use as features. Defaults to the
                known vital-sign columns so demographics are never included.
            random_state: Random state for reproducibility

        Returns:
            List of indices predicted as anomalies
        """
        if df.empty:
            return []

        if feature_columns is None:
            feature_columns = VITAL_SIGN_FEATURES

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

    def _compute_isolation_forest_feature_importance(
        self,
        model_input: pd.DataFrame,
        predictions: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute an explainability proxy for Isolation Forest anomaly drivers.

        Isolation Forest does not expose native per-feature importances for a
        fitted model instance. This method estimates feature influence by
        comparing anomaly vs non-anomaly center values and scaling by overall
        feature variability.

        Args:
            model_input: Numeric model input used for training/prediction
            predictions: Isolation Forest predictions (1 for normal, -1 anomaly)

        Returns:
            Dictionary mapping feature names to normalized importance percentages
        """
        if model_input.empty or len(predictions) != len(model_input):
            return {}

        anomaly_mask = predictions == -1
        normal_mask = predictions == 1

        if anomaly_mask.sum() == 0 or normal_mask.sum() == 0:
            return {}

        anomaly_frame = model_input.loc[anomaly_mask]
        normal_frame = model_input.loc[normal_mask]

        raw_scores = {}
        for column in model_input.columns:
            std_val = float(model_input[column].std())
            if std_val == 0 or pd.isna(std_val):
                continue

            anomaly_center = float(anomaly_frame[column].median())
            normal_center = float(normal_frame[column].median())
            raw_scores[column] = abs(anomaly_center - normal_center) / std_val

        total_score = sum(raw_scores.values())
        if total_score <= 0:
            return {}

        return {
            feature: round((score / total_score) * 100, 2)
            for feature, score in sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
        }
    
    def generate_anomaly_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive anomaly detection report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing anomaly detection results
        """
        if df.empty:
            return {
                'total_records': 0,
                'vital_sign_anomalies': {},
                'isolation_forest_anomalies': [],
                'isolation_forest_feature_importance': {},
                'total_anomalies': 0,
                'anomaly_rate': 0,
                'isolation_forest_anomaly_count': 0,
                'isolation_forest_anomaly_rate': 0,
                'timestamp': pd.Timestamp.now(),
            }

        available_features = [col for col in VITAL_SIGN_FEATURES if col in df.columns]
        model_input = df[available_features].copy() if available_features else pd.DataFrame(index=df.index)
        if not model_input.empty:
            model_input = model_input.fillna(model_input.median(numeric_only=True))

        if_anomalies = []
        if_feature_importance = {}
        if not model_input.empty:
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=200,
            )
            predictions = model.fit_predict(model_input)
            if_anomalies = df.index[predictions == -1].tolist()
            if_feature_importance = self._compute_isolation_forest_feature_importance(
                model_input=model_input,
                predictions=predictions,
            )

        report = {
            'total_records': len(df),
            'vital_sign_anomalies': self.detect_vital_sign_anomalies(df),
            'isolation_forest_anomalies': if_anomalies,
            'isolation_forest_feature_importance': if_feature_importance,
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
