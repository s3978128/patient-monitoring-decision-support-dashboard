"""
Clinical Rules Module
Implements clinical decision support rules and alert logic.
"""

import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from modules.clinical_thresholds import (
        TACHYCARDIA_WARNING, TACHYCARDIA_CRITICAL,
        HYPERTENSION_STAGE1_SYSTOLIC, HYPERTENSION_STAGE1_DIASTOLIC,
        HYPERTENSION_STAGE2_SYSTOLIC, HYPERTENSION_STAGE2_DIASTOLIC,
        HYPERTENSION_CRISIS_SYSTOLIC, HYPERTENSION_CRISIS_DIASTOLIC,
        HYPOTENSION_WARNING_SYSTOLIC, HYPOTENSION_WARNING_DIASTOLIC,
        HYPOTENSION_CRITICAL_SYSTOLIC, HYPOTENSION_CRITICAL_DIASTOLIC,
        HYPOXEMIA_WARNING, HYPOXEMIA_CRITICAL,
        RESPIRATORY_HIGH_WARNING, RESPIRATORY_HIGH_CRITICAL,
        RESPIRATORY_LOW_WARNING, RESPIRATORY_LOW_CRITICAL,
        FEVER_WARNING_HIGH, FEVER_CRITICAL_HIGH,
        FEVER_WARNING_LOW, FEVER_CRITICAL_LOW,
    )
except ImportError:
    from clinical_thresholds import (
        TACHYCARDIA_WARNING, TACHYCARDIA_CRITICAL,
        HYPERTENSION_STAGE1_SYSTOLIC, HYPERTENSION_STAGE1_DIASTOLIC,
        HYPERTENSION_STAGE2_SYSTOLIC, HYPERTENSION_STAGE2_DIASTOLIC,
        HYPERTENSION_CRISIS_SYSTOLIC, HYPERTENSION_CRISIS_DIASTOLIC,
        HYPOTENSION_WARNING_SYSTOLIC, HYPOTENSION_WARNING_DIASTOLIC,
        HYPOTENSION_CRITICAL_SYSTOLIC, HYPOTENSION_CRITICAL_DIASTOLIC,
        HYPOXEMIA_WARNING, HYPOXEMIA_CRITICAL,
        RESPIRATORY_HIGH_WARNING, RESPIRATORY_HIGH_CRITICAL,
        RESPIRATORY_LOW_WARNING, RESPIRATORY_LOW_CRITICAL,
        FEVER_WARNING_HIGH, FEVER_CRITICAL_HIGH,
        FEVER_WARNING_LOW, FEVER_CRITICAL_LOW,
    )


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ClinicalAlert:
    """Represents a clinical alert."""
    patient_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    triggered_at: pd.Timestamp
    values: Dict
    framework: str = "local"


class ClinicalRulesEngine:
    """
    Evaluates clinical rules and generates alerts.

    Thresholds are intentionally simplified but loosely aligned to common frameworks:
    - Blood pressure staging: ACC/AHA-style cutoffs
    - Oxygen saturation and respiratory urgency: NEWS2-inspired cutoffs
    - Fever thresholds: common inpatient triage cutoffs
    """
    
    def __init__(self):
        """Initialize the clinical rules engine."""

    @staticmethod
    def _is_missing(value) -> bool:
        """Return True when a value should be treated as missing."""
        return value is None or pd.isna(value)
    
    def check_tachycardia(self, patient_data: Dict) -> Optional[ClinicalAlert]:
        """
        Check for tachycardia (elevated heart rate).
        
        Args:
            patient_data: Dictionary containing patient data
            
        Returns:
            ClinicalAlert if rule is triggered, None otherwise
        """
        heart_rate = patient_data.get('heart_rate')
        patient_id = patient_data.get('patient_id')
        
        if self._is_missing(heart_rate):
            return None
        
        if heart_rate > TACHYCARDIA_WARNING:
            severity = AlertSeverity.CRITICAL if heart_rate >= TACHYCARDIA_CRITICAL else AlertSeverity.WARNING
            return ClinicalAlert(
                patient_id=patient_id,
                rule_name="Tachycardia Detection",
                severity=severity,
                message=f"Elevated heart rate detected: {heart_rate} bpm",
                triggered_at=pd.Timestamp.now(),
                values={'heart_rate': heart_rate},
                framework="AHA ACLS / common inpatient triage"
            )
        
        return None
    
    def check_hypertension(self, patient_data: Dict) -> Optional[ClinicalAlert]:
        """
        Check for hypertension (high blood pressure).
        
        Args:
            patient_data: Dictionary containing patient data
            
        Returns:
            ClinicalAlert if rule is triggered, None otherwise
        """
        systolic = patient_data.get('blood_pressure_systolic')
        diastolic = patient_data.get('blood_pressure_diastolic')
        patient_id = patient_data.get('patient_id')
        
        if self._is_missing(systolic) or self._is_missing(diastolic):
            return None
        
        if systolic >= HYPERTENSION_STAGE1_SYSTOLIC or diastolic >= HYPERTENSION_STAGE1_DIASTOLIC:
            if systolic >= HYPERTENSION_CRISIS_SYSTOLIC or diastolic >= HYPERTENSION_CRISIS_DIASTOLIC:
                severity = AlertSeverity.CRITICAL
                stage = "Hypertensive crisis"
            elif systolic >= HYPERTENSION_STAGE2_SYSTOLIC or diastolic >= HYPERTENSION_STAGE2_DIASTOLIC:
                severity = AlertSeverity.WARNING
                stage = "Stage 2 hypertension"
            else:
                severity = AlertSeverity.INFO
                stage = "Stage 1 hypertension"

            return ClinicalAlert(
                patient_id=patient_id,
                rule_name="Hypertension Staging",
                severity=severity,
                message=f"{stage}: {systolic}/{diastolic} mmHg",
                triggered_at=pd.Timestamp.now(),
                values={'systolic': systolic, 'diastolic': diastolic, 'stage': stage},
                framework="ACC/AHA 2017 (simplified)"
            )
        
        return None

    def check_hypotension(self, patient_data: Dict) -> Optional[ClinicalAlert]:
        """Check for hypotension using common acute-care thresholds."""
        systolic = patient_data.get('blood_pressure_systolic')
        diastolic = patient_data.get('blood_pressure_diastolic')
        patient_id = patient_data.get('patient_id')

        if self._is_missing(systolic) or self._is_missing(diastolic):
            return None

        if systolic < HYPOTENSION_WARNING_SYSTOLIC or diastolic < HYPOTENSION_WARNING_DIASTOLIC:
            severity = AlertSeverity.CRITICAL if systolic < HYPOTENSION_CRITICAL_SYSTOLIC or diastolic < HYPOTENSION_CRITICAL_DIASTOLIC else AlertSeverity.WARNING
            return ClinicalAlert(
                patient_id=patient_id,
                rule_name="Hypotension Detection",
                severity=severity,
                message=f"Low blood pressure: {systolic}/{diastolic} mmHg",
                triggered_at=pd.Timestamp.now(),
                values={'systolic': systolic, 'diastolic': diastolic},
                framework="Common acute-care thresholds"
            )

        return None
    
    def check_hypoxemia(self, patient_data: Dict) -> Optional[ClinicalAlert]:
        """
        Check for hypoxemia (low oxygen saturation).
        
        Args:
            patient_data: Dictionary containing patient data
            
        Returns:
            ClinicalAlert if rule is triggered, None otherwise
        """
        oxygen_sat = patient_data.get('oxygen_saturation')
        patient_id = patient_data.get('patient_id')
        
        if self._is_missing(oxygen_sat):
            return None
        
        if oxygen_sat <= HYPOXEMIA_WARNING:
            severity = AlertSeverity.CRITICAL if oxygen_sat <= HYPOXEMIA_CRITICAL else AlertSeverity.WARNING
            return ClinicalAlert(
                patient_id=patient_id,
                rule_name="Hypoxemia Detection",
                severity=severity,
                message=f"Low oxygen saturation: {oxygen_sat}%",
                triggered_at=pd.Timestamp.now(),
                values={'oxygen_saturation': oxygen_sat},
                framework="NEWS2-inspired oxygen thresholds"
            )
        
        return None

    def check_respiratory_distress(self, patient_data: Dict) -> Optional[ClinicalAlert]:
        """Check respiratory rate with NEWS2-inspired urgency cutoffs."""
        respiratory_rate = patient_data.get('respiratory_rate')
        patient_id = patient_data.get('patient_id')

        if self._is_missing(respiratory_rate):
            return None

        if respiratory_rate >= RESPIRATORY_HIGH_WARNING or respiratory_rate <= RESPIRATORY_LOW_WARNING:
            severity = AlertSeverity.CRITICAL if respiratory_rate >= RESPIRATORY_HIGH_CRITICAL or respiratory_rate <= RESPIRATORY_LOW_CRITICAL else AlertSeverity.WARNING
            return ClinicalAlert(
                patient_id=patient_id,
                rule_name="Respiratory Distress Detection",
                severity=severity,
                message=f"Abnormal respiratory rate: {respiratory_rate} breaths/min",
                triggered_at=pd.Timestamp.now(),
                values={'respiratory_rate': respiratory_rate},
                framework="NEWS2-inspired respiratory thresholds"
            )

        return None
    
    def check_fever(self, patient_data: Dict) -> Optional[ClinicalAlert]:
        """
        Check for fever (elevated temperature).
        
        Args:
            patient_data: Dictionary containing patient data
            
        Returns:
            ClinicalAlert if rule is triggered, None otherwise
        """
        temperature = patient_data.get('temperature')
        patient_id = patient_data.get('patient_id')
        
        if self._is_missing(temperature):
            return None
        
        if temperature >= FEVER_WARNING_HIGH or temperature < FEVER_WARNING_LOW:
            severity = AlertSeverity.CRITICAL if temperature >= FEVER_CRITICAL_HIGH or temperature < FEVER_CRITICAL_LOW else AlertSeverity.WARNING
            return ClinicalAlert(
                patient_id=patient_id,
                rule_name="Temperature Abnormality Detection",
                severity=severity,
                message=f"Abnormal temperature: {temperature}°C",
                triggered_at=pd.Timestamp.now(),
                values={'temperature': temperature},
                framework="Common inpatient triage"
            )
        
        return None
    
    def evaluate_all_rules(self, patient_data: Dict) -> List[ClinicalAlert]:
        """
        Evaluate all clinical rules for a patient.
        
        Args:
            patient_data: Dictionary containing patient data
            
        Returns:
            List of triggered alerts
        """
        rules = [
            self.check_tachycardia,
            self.check_hypertension,
            self.check_hypotension,
            self.check_hypoxemia,
            self.check_respiratory_distress,
            self.check_fever
        ]
        
        alerts = []
        for rule in rules:
            alert = rule(patient_data)
            if alert:
                alerts.append(alert)

        return alerts
    
    @staticmethod
    def get_alerts_summary(alerts: List[ClinicalAlert]) -> Dict:
        """
        Compute a summary from an explicit list of alerts.

        Args:
            alerts: List of ClinicalAlert objects to summarise.

        Returns:
            Dictionary containing alert statistics.
        """
        severity_counts = {
            'critical': sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL),
            'warning': sum(1 for a in alerts if a.severity == AlertSeverity.WARNING),
            'info': sum(1 for a in alerts if a.severity == AlertSeverity.INFO)
        }

        return {
            'total_alerts': len(alerts),
            'severity_breakdown': severity_counts,
        }


if __name__ == "__main__":
    # Example usage
    engine = ClinicalRulesEngine()
    
    patient = {
        'patient_id': 'P001',
        'heart_rate': 115,
        'blood_pressure_systolic': 145,
        'blood_pressure_diastolic': 92,
        'temperature': 38.5,
        'oxygen_saturation': 96
    }
    
    alerts = engine.evaluate_all_rules(patient)
    for alert in alerts:
        print(f"{alert.severity.value.upper()}: {alert.message}")
