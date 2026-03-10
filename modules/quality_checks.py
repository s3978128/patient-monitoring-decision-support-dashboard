"""
Quality Checks Module
Performs data quality validation and monitoring for clinical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


class DataQualityChecker:
    """Performs quality checks on clinical data."""

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
        # Wide ranges for data quality: catches impossible values
        # (e.g., sensor errors, data entry mistakes)
        physiological_ranges = {
            'heart_rate': (30, 200),  # bpm 
            'blood_pressure_systolic': (50, 250),  # mmHg
            'blood_pressure_diastolic': (30, 150),  # mmHg
            'temperature': (30.0, 45.0),  # °C
            'respiratory_rate': (5, 60),  # breaths per minute
            'oxygen_saturation': (50, 100)  # %
        }
        
        return self.check_value_ranges(df, physiological_ranges)
    
    def check_data_completeness(self, df: pd.DataFrame, 
                               required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if all required columns are present.
        
        Args:
            df: DataFrame to check
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_complete, list of missing columns)
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        is_complete = len(missing_columns) == 0
        
        return is_complete, missing_columns
    
    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing quality metrics
        """
        report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': self.check_missing_values(df),
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
