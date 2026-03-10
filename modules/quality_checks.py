"""
Quality Checks Module
Performs data quality validation and monitoring for clinical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

try:
    from modules.clinical_thresholds import PHYSIOLOGICAL_LIMITS
except ImportError:
    from clinical_thresholds import PHYSIOLOGICAL_LIMITS


class DataQualityChecker:
    """Performs quality checks on clinical data."""

    REQUIRED_COLUMNS = [
        'patient_id',
        'name',
        'age',
        'gender',
        'timestamp',
        'heart_rate',
        'blood_pressure_systolic',
        'blood_pressure_diastolic',
        'temperature',
        'respiratory_rate',
        'oxygen_saturation',
    ]

    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Check for missing values in the dataset.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with column names and percentage of missing values
        """
        missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        return {col: round(pct, 2) for col, pct in missing_percentages.items() if pct > 0}
    
    def check_duplicates(self, df: pd.DataFrame, id_column: str) -> Tuple[int, List]:
        """
        Check for duplicate records.
        
        Args:
            df: DataFrame to check
            id_column: Column name containing unique identifiers
            
        Returns:
            Tuple of (number of duplicates, list of duplicate IDs)
        """
        duplicates = df[df.duplicated(subset=[id_column], keep=False)]
        duplicate_ids = duplicates[id_column].unique().tolist()
        
        return len(duplicate_ids), duplicate_ids
    
    def check_value_ranges(self, df: pd.DataFrame, 
                          ranges: Dict[str, Tuple[float, float]]) -> Dict[str, List]:
        """
        Check if values are within expected ranges.
        
        Args:
            df: DataFrame to check
            ranges: Dictionary mapping column names to (min, max) tuples
            
        Returns:
            Dictionary of columns with out-of-range values
        """
        out_of_range = {}
        
        for column, (min_val, max_val) in ranges.items():
            if column in df.columns:
                invalid = (df[column] < min_val) | (df[column] > max_val)
                if invalid.any():
                    out_of_range[column] = df[invalid].index.tolist()
        
        return out_of_range
    
    def check_physiological_limits(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        Check for physiologically impossible values (data quality issues).
        
        Uses wide ranges to catch sensor errors, data entry mistakes, or
        physically impossible measurements. Values outside these ranges
        indicate data quality problems rather than clinical conditions.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary of columns with physiologically impossible values
        """
        return self.check_value_ranges(df, PHYSIOLOGICAL_LIMITS)
    
    def check_data_completeness(self, df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if all required columns are present.
        
        Args:
            df: DataFrame to check
            required_columns: List of required column names

        Returns:
            Tuple of (is_complete, missing_columns)
        """
        missing_columns = [col for col in required_columns if col not in df.columns]

        is_complete = len(missing_columns) == 0

        return is_complete, missing_columns
    
    def check_data_types(self, df: pd.DataFrame, expected_types: Dict[str, Any]) -> Dict[str, bool]:
        """
        Check if columns have expected data types.
        
        Args:
            df: DataFrame to check
            expected_types: Dictionary mapping column names to expected data types
        Returns:
            Dictionary of columns with correct data types (True/False)
        """
        type_checks = {}
        
        for column, expected_type in expected_types.items():
            if column in df.columns:
                type_checks[column] = df[column].map(type).eq(expected_type).all()
            else:
                type_checks[column] = False  # Column missing, so type check fails
        
        return type_checks
    
    def generate_quality_report(
        self,
        df: pd.DataFrame,
        required_columns: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing quality metrics
        """
        if required_columns is None:
            required_columns = self.REQUIRED_COLUMNS

        is_complete, missing_columns = self.check_data_completeness(df, required_columns)

        report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': self.check_missing_values(df),
            'schema_complete': is_complete,
            'missing_columns': missing_columns,
            'data_types': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        return report


if __name__ == "__main__":
    # Example usage
    checker = DataQualityChecker()
    sample_data = pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003'],
        'age': [45, None, 67],
        'heart_rate': [72, 85, 68]
    })
    
    report = checker.generate_quality_report(sample_data)
    print(report)
