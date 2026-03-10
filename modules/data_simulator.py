"""
Data Simulator Module
Generates simulated patient data for testing and monitoring the CDSS system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


# Lists of common first and last names for patient simulation
FIRST_NAMES = [
    'James', 'Mary', 'Robert', 'Patricia', 'Michael', 'Jennifer', 'William', 'Linda',
    'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica', 'Thomas', 'Sarah',
    'Charles', 'Karen', 'Christopher', 'Nancy', 'Daniel', 'Betty', 'Matthew', 'Margaret',
    'Anthony', 'Sandra', 'Mark', 'Ashley', 'Donald', 'Kimberly', 'Steven', 'Emily',
    'Paul', 'Donna', 'Andrew', 'Michelle', 'Joshua', 'Dorothy', 'Kenneth', 'Carol',
    'Kevin', 'Amanda', 'Brian', 'Melissa', 'Edward', 'Deborah', 'Ronald', 'Stephanie',
    'Timothy', 'Rebecca', 'Jason', 'Sharon', 'Jeffrey', 'Laura', 'Ryan', 'Cynthia',
    'Jacob', 'Kathleen', 'Gary', 'Amy', 'Nicholas', 'Shirley', 'Eric', 'Angela'
]

LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
    'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
    'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
    'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
    'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Peterson', 'Phillips',
    'Campbell', 'Parker', 'Edwards', 'Collins', 'Reyes', 'Stewart', 'Morris', 'Morales',
    'Gutierrez', 'Ortiz', 'Murphy', 'Ojeda', 'Aguilar', 'Medina', 'Velez', 'Cervantes'
]


class PatientDataSimulator:
    """Simulates patient data for CDSS testing and monitoring."""
    
    def __init__(
        self,
        seed: Optional[int] = None,
        anomaly_rate: float = 0.3,
        severe_error_rate: float = 0.05,
    ):
        """
        Initialize the data simulator.
        
        Args:
            seed: Random seed for reproducibility
            anomaly_rate: Probability of generating anomalous values (0.0 to 1.0)
            severe_error_rate: Probability of injecting extreme erroneous values
        """
        if seed is not None:
            np.random.seed(seed)
        self.anomaly_rate = anomaly_rate
        self.severe_error_rate = severe_error_rate
    
    def _generate_random_name(self) -> str:
        """
        Generate a random patient name.
        
        Returns:
            String containing a random first and last name
        """
        first_name = np.random.choice(FIRST_NAMES)
        last_name = np.random.choice(LAST_NAMES)
        return f"{first_name} {last_name}"
    
    def generate_patients(self, n_patients: int) -> pd.DataFrame:
        """
        Generate simulated patient records.
        
        Args:
            n_patients: Number of patient records to generate
            
        Returns:
            DataFrame containing simulated patient data
        """
        patients = []
        
        for i in range(n_patients):
            patient = {
                'patient_id': f'PAT{i+1:05d}',
                'name': self._generate_random_name(),
                'age': np.random.randint(18, 90),
                'gender': np.random.choice(['M', 'F']),
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
            }
            patients.append(patient)
        
        return pd.DataFrame(patients)
    
    def generate_vital_signs(self, patient_id: str) -> Dict:
        """
        Generate simulated vital signs for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary containing vital signs
        """
        # Generate normal vital signs first
        vital_signs = {
            'patient_id': patient_id,
            'heart_rate': np.random.randint(60, 100),
            'blood_pressure_systolic': np.random.randint(110, 140),
            'blood_pressure_diastolic': np.random.randint(70, 90),
            'temperature': round(np.random.uniform(36.5, 37.5), 1),
            'respiratory_rate': np.random.randint(12, 20),
            'oxygen_saturation': np.random.randint(95, 100)
        }
        
        # Inject clinically significant anomalies.
        if np.random.random() < self.anomaly_rate:
            anomaly_type = np.random.choice([
                'tachycardia', 'bradycardia', 'hypertension', 'hypotension',
                'fever', 'hypothermia', 'hypoxemia', 'tachypnea'
            ])
            
            if anomaly_type == 'tachycardia':
                # Heart rate > 120 (normal max is 120)
                vital_signs['heart_rate'] = np.random.randint(125, 160)
            
            elif anomaly_type == 'bradycardia':
                # Heart rate < 40 (normal min is 40)
                vital_signs['heart_rate'] = np.random.randint(25, 38)
            
            elif anomaly_type == 'hypertension':
                # Blood pressure > 180 systolic or > 100 diastolic
                vital_signs['blood_pressure_systolic'] = np.random.randint(185, 220)
                vital_signs['blood_pressure_diastolic'] = np.random.randint(95, 120)
            
            elif anomaly_type == 'hypotension':
                # Blood pressure < 80 systolic or < 50 diastolic
                vital_signs['blood_pressure_systolic'] = np.random.randint(60, 78)
                vital_signs['blood_pressure_diastolic'] = np.random.randint(35, 48)
            
            elif anomaly_type == 'fever':
                # Temperature > 39.0°C (normal max is 39.0)
                vital_signs['temperature'] = round(np.random.uniform(39.2, 41.5), 1)
            
            elif anomaly_type == 'hypothermia':
                # Temperature < 35.0°C (normal min is 35.0)
                vital_signs['temperature'] = round(np.random.uniform(32.5, 34.8), 1)
            
            elif anomaly_type == 'hypoxemia':
                # Oxygen saturation < 85% (normal min is 85)
                vital_signs['oxygen_saturation'] = np.random.randint(70, 83)
            
            elif anomaly_type == 'tachypnea':
                # Respiratory rate > 30 (normal max is 30)
                vital_signs['respiratory_rate'] = np.random.randint(32, 45)

        # Inject severe data errors that are likely sensor/data-entry problems.
        if np.random.random() < self.severe_error_rate:
            severe_error_type = np.random.choice([
                'heart_rate_extreme',
                'bp_extreme',
                'temperature_extreme',
                'respiratory_extreme',
                'spo2_extreme',
            ])

            if severe_error_type == 'heart_rate_extreme':
                vital_signs['heart_rate'] = np.random.choice([0, 12, 260, 320])
            elif severe_error_type == 'bp_extreme':
                vital_signs['blood_pressure_systolic'] = np.random.choice([30, 320])
                vital_signs['blood_pressure_diastolic'] = np.random.choice([15, 210])
            elif severe_error_type == 'temperature_extreme':
                vital_signs['temperature'] = float(np.random.choice([25.0, 48.5]))
            elif severe_error_type == 'respiratory_extreme':
                vital_signs['respiratory_rate'] = np.random.choice([2, 85])
            elif severe_error_type == 'spo2_extreme':
                vital_signs['oxygen_saturation'] = np.random.choice([20, 45, 130])
        
        return vital_signs

    def generate_dataset(
        self,
        n_patients: int,
        duplicate_rate: float = 0.1,
        missing_value_rate: float = 0.2,
        missing_column_rate: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Generate a full dataset with optional real-world data errors.

        Args:
            n_patients: Number of base patient records to generate
            duplicate_rate: Fraction of rows to duplicate
            missing_value_rate: Probability of replacing each eligible cell with a missing value
            missing_column_rate: Backward-compatible alias for missing_value_rate

        Returns:
            DataFrame with demographics, vitals, and injected errors
        """
        if missing_column_rate is not None:
            missing_value_rate = missing_column_rate

        patients_df = self.generate_patients(n_patients=n_patients)

        vital_signs_list = []
        for patient_id in patients_df['patient_id']:
            vital_signs_list.append(self.generate_vital_signs(patient_id))

        vitals_df = pd.DataFrame(vital_signs_list)
        full_df = patients_df.merge(vitals_df, on='patient_id')

        full_df = self._inject_duplicates(full_df, duplicate_rate)
        full_df = self._inject_missing_values(full_df, missing_value_rate)

        return full_df.reset_index(drop=True)

    def _inject_duplicates(self, df: pd.DataFrame, duplicate_rate: float) -> pd.DataFrame:
        """Duplicate a random subset of rows to simulate duplicate records."""
        if df.empty or duplicate_rate <= 0:
            return df

        duplicate_count = int(len(df) * duplicate_rate)
        if duplicate_count <= 0:
            return df

        duplicate_rows = df.sample(n=duplicate_count, replace=True)
        return pd.concat([df, duplicate_rows], ignore_index=True)

    def _inject_missing_values(self, df: pd.DataFrame, missing_value_rate: float) -> pd.DataFrame:
        """Replace random non-key cells with missing values to simulate incomplete records."""
        if missing_value_rate <= 0 or df.empty:
            return df

        protected_columns = {'patient_id', 'name', 'timestamp'}
        candidate_columns = [c for c in df.columns if c not in protected_columns]

        if not candidate_columns:
            return df

        df_with_missing = df.copy()
        random_mask = np.random.random((len(df_with_missing), len(candidate_columns))) < missing_value_rate
        mask_df = pd.DataFrame(random_mask, columns=candidate_columns, index=df_with_missing.index)

        # Keep at least one populated clinical field per row so downstream analysis still has signal.
        rows_all_missing = mask_df.all(axis=1)
        for row_index in mask_df.index[rows_all_missing]:
            column_to_keep = np.random.choice(candidate_columns)
            mask_df.at[row_index, column_to_keep] = False

        for column in candidate_columns:
            df_with_missing.loc[mask_df[column], column] = np.nan

        return df_with_missing


if __name__ == "__main__":
    # Example usage with realistic data issues.
    simulator = PatientDataSimulator(seed=42, anomaly_rate=0.2, severe_error_rate=0.1)
    full_df = simulator.generate_dataset(
        n_patients=20,
        duplicate_rate=0.15,
        missing_value_rate=0.15,
    )
    
    print("Generated patient data with vitals:")
    print(full_df)
    print("\n" + "="*50)
    print("Vital signs summary:")
    numeric_columns = full_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        print(full_df[numeric_columns].describe())
    else:
        print("No numeric columns available.")
