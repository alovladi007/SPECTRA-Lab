"""
Fault Classification

Classifies detected faults to root causes based on fault type, metric, and context.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .detector import Fault, FaultType, RootCause
from ..spc.series import SPCMetric


@dataclass
class RootCauseHypothesis:
    """A potential root cause with confidence"""
    root_cause: RootCause
    confidence: float  # 0-1
    reasoning: str
    recommended_action: str


class FaultClassifier:
    """Base class for fault classifiers"""

    def classify(self, fault: Fault) -> List[RootCauseHypothesis]:
        """
        Classify fault to potential root causes

        Returns:
            List of root cause hypotheses sorted by confidence
        """
        raise NotImplementedError


class ThicknessFaultClassifier(FaultClassifier):
    """
    Classifier for thickness-related faults

    Patterns:
    - Gradual drift upward → MFC drift (more precursor)
    - Gradual drift downward → Precursor depletion, MFC drift (less carrier)
    - Sudden shift → Recipe change, chamber clean, VM mis-tune
    - Increased variation → Temperature instability, flow instability
    """

    def classify(self, fault: Fault) -> List[RootCauseHypothesis]:
        hypotheses = []

        if fault.fault_type == FaultType.GRADUAL_DRIFT_UPWARD:
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.MFC_CALIBRATION_DRIFT,
                confidence=0.75,
                reasoning="Gradual thickness increase suggests MFC delivering more precursor",
                recommended_action="Calibrate MFC controllers, verify flow readings",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.TEMPERATURE_CONTROLLER_DRIFT,
                confidence=0.5,
                reasoning="Temperature increase can raise deposition rate",
                recommended_action="Verify temperature setpoints and controller calibration",
            ))

        elif fault.fault_type == FaultType.GRADUAL_DRIFT_DOWNWARD:
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.PRECURSOR_DEPLETION,
                confidence=0.7,
                reasoning="Gradual thickness decrease may indicate precursor bottle depletion",
                recommended_action="Check precursor levels, replace if low",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.MFC_CALIBRATION_DRIFT,
                confidence=0.6,
                reasoning="MFC drift could reduce precursor delivery",
                recommended_action="Calibrate MFC controllers",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.CHAMBER_CONTAMINATION,
                confidence=0.4,
                reasoning="Contamination can reduce deposition efficiency",
                recommended_action="Inspect chamber, perform wet clean if needed",
            ))

        elif fault.fault_type == FaultType.SUDDEN_SHIFT_UPWARD:
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.RECIPE_ERROR,
                confidence=0.6,
                reasoning="Sudden increase may indicate recipe parameter change",
                recommended_action="Verify recipe parameters against golden recipe",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.VM_MODEL_MISTUNE,
                confidence=0.5,
                reasoning="VM model mis-tuning could cause PCM overshoot",
                recommended_action="Re-tune VM model, verify PCM targets",
            ))

        elif fault.fault_type == FaultType.SUDDEN_SHIFT_DOWNWARD:
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.RECIPE_ERROR,
                confidence=0.6,
                reasoning="Sudden decrease may indicate recipe parameter change",
                recommended_action="Verify recipe parameters against golden recipe",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.GAS_LINE_CONTAMINATION,
                confidence=0.4,
                reasoning="Gas line blockage could reduce precursor delivery",
                recommended_action="Inspect gas lines, verify flow rates",
            ))

        elif fault.fault_type == FaultType.INCREASED_VARIATION:
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.TEMPERATURE_CONTROLLER_DRIFT,
                confidence=0.7,
                reasoning="Increased thickness variation suggests temperature instability",
                recommended_action="Check heater performance, verify PID tuning",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.MFC_CALIBRATION_DRIFT,
                confidence=0.6,
                reasoning="Flow instability causes deposition rate variation",
                recommended_action="Check MFC stability, calibrate if needed",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.PRESSURE_CONTROLLER_DRIFT,
                confidence=0.5,
                reasoning="Pressure fluctuations affect deposition rate",
                recommended_action="Verify pressure control loop stability",
            ))

        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        return hypotheses


class StressFaultClassifier(FaultClassifier):
    """
    Classifier for stress-related faults

    Patterns:
    - Gradual stress increase (less compressive) → Thermal drift, composition shift
    - Gradual stress decrease (more compressive) → Temperature increase, power drift
    - Sudden shift → Recipe change, gas composition change
    """

    def classify(self, fault: Fault) -> List[RootCauseHypothesis]:
        hypotheses = []

        if fault.fault_type == FaultType.GRADUAL_DRIFT_UPWARD:
            # Stress becoming less compressive (or more tensile)
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.HEATER_DEGRADATION,
                confidence=0.65,
                reasoning="Heater degradation reduces deposition temperature, increasing tensile stress",
                recommended_action="Inspect heater elements, verify temperature uniformity",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.GAS_LINE_CONTAMINATION,
                confidence=0.5,
                reasoning="Gas composition change affects film stoichiometry and stress",
                recommended_action="Analyze film composition, inspect gas lines",
            ))

        elif fault.fault_type == FaultType.GRADUAL_DRIFT_DOWNWARD:
            # Stress becoming more compressive
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.TEMPERATURE_CONTROLLER_DRIFT,
                confidence=0.7,
                reasoning="Temperature increase raises compressive stress",
                recommended_action="Calibrate temperature controller, verify setpoints",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.RF_GENERATOR_DRIFT,
                confidence=0.6,
                reasoning="RF power drift affects ion bombardment and stress",
                recommended_action="Calibrate RF generator, verify power delivery",
            ))

        elif fault.fault_type in [FaultType.SUDDEN_SHIFT_UPWARD, FaultType.SUDDEN_SHIFT_DOWNWARD]:
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.RECIPE_ERROR,
                confidence=0.7,
                reasoning="Sudden stress change suggests recipe parameter modification",
                recommended_action="Verify all recipe parameters, especially temperature and power",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.VM_MODEL_MISTUNE,
                confidence=0.5,
                reasoning="VM model error could cause PCM to target wrong conditions",
                recommended_action="Re-validate VM model, check PCM targets",
            ))

        elif fault.fault_type == FaultType.INCREASED_VARIATION:
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.TEMPERATURE_CONTROLLER_DRIFT,
                confidence=0.75,
                reasoning="Temperature variation is primary driver of stress variation",
                recommended_action="Check temperature stability, verify PID controller",
            ))

        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        return hypotheses


class AdhesionFaultClassifier(FaultClassifier):
    """
    Classifier for adhesion-related faults

    Patterns:
    - Sudden adhesion drop → Contamination, pre-clean failure
    - Gradual adhesion decrease → Surface roughness increase, contamination buildup
    - High variation → Pre-clean inconsistency, substrate quality variation
    """

    def classify(self, fault: Fault) -> List[RootCauseHypothesis]:
        hypotheses = []

        if fault.fault_type == FaultType.SUDDEN_SHIFT_DOWNWARD:
            # Sudden adhesion drop - most critical
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.PRECLEAN_FAILURE,
                confidence=0.8,
                reasoning="Sudden adhesion loss strongly indicates pre-clean failure",
                recommended_action="Verify pre-clean chemistry and timing, inspect clean module",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.PARTICLE_GENERATION,
                confidence=0.6,
                reasoning="Particle contamination between substrate and film reduces adhesion",
                recommended_action="Check particle counts, inspect chamber for flaking",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.CHAMBER_CONTAMINATION,
                confidence=0.5,
                reasoning="Chamber contamination may deposit on substrate before deposition",
                recommended_action="Perform chamber clean, verify cleanliness",
            ))

        elif fault.fault_type == FaultType.GRADUAL_DRIFT_DOWNWARD:
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.CHAMBER_CONTAMINATION,
                confidence=0.7,
                reasoning="Gradual contamination buildup reduces adhesion over time",
                recommended_action="Perform chamber clean, establish clean frequency",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.PRECLEAN_FAILURE,
                confidence=0.6,
                reasoning="Pre-clean chemistry degradation reduces effectiveness",
                recommended_action="Check pre-clean solution concentration and age",
            ))

        elif fault.fault_type == FaultType.INCREASED_VARIATION:
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.PRECLEAN_FAILURE,
                confidence=0.75,
                reasoning="Inconsistent pre-clean causes adhesion variation",
                recommended_action="Verify pre-clean repeatability, check chemistry stability",
            ))

            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.PARTICLE_GENERATION,
                confidence=0.5,
                reasoning="Variable particle levels cause adhesion variation",
                recommended_action="Monitor particle counts, identify source",
            ))

        elif fault.fault_type == FaultType.OUT_OF_SPEC_LOW:
            # Critical low adhesion
            hypotheses.append(RootCauseHypothesis(
                root_cause=RootCause.PRECLEAN_FAILURE,
                confidence=0.85,
                reasoning="Low adhesion most commonly caused by pre-clean issues",
                recommended_action="URGENT: Verify pre-clean process, hold lots if needed",
            ))

        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        return hypotheses


def classify_fault_root_cause(fault: Fault) -> Optional[Fault]:
    """
    Classify fault to root cause and update fault object

    Args:
        fault: Detected fault

    Returns:
        Updated fault with root cause classification
    """
    # Select appropriate classifier based on metric
    if fault.metric in [SPCMetric.THICKNESS_MEAN, SPCMetric.THICKNESS_UNIFORMITY]:
        classifier = ThicknessFaultClassifier()

    elif fault.metric in [SPCMetric.STRESS_MEAN, SPCMetric.STRESS_ABS]:
        classifier = StressFaultClassifier()

    elif fault.metric in [SPCMetric.ADHESION_SCORE, SPCMetric.ADHESION_CLASS_DIST]:
        classifier = AdhesionFaultClassifier()

    else:
        return fault  # No classifier available

    # Get root cause hypotheses
    hypotheses = classifier.classify(fault)

    if hypotheses:
        # Use highest confidence hypothesis
        top_hypothesis = hypotheses[0]

        fault.root_cause = top_hypothesis.root_cause
        fault.root_cause_confidence = top_hypothesis.confidence
        fault.recommended_action = top_hypothesis.recommended_action

        # Store all hypotheses in statistics
        fault.statistics["root_cause_hypotheses"] = [
            {
                "root_cause": h.root_cause.value,
                "confidence": h.confidence,
                "reasoning": h.reasoning,
                "action": h.recommended_action,
            }
            for h in hypotheses
        ]

    return fault
