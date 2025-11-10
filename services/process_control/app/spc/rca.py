"""Root Cause Analysis (RCA) playbooks and alert triage.

Provides structured troubleshooting guidance when SPC alerts occur,
including symptom-to-cause mapping and recommended corrective actions.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

from .charts import SPCAlert, RuleViolation, AlertSeverity
from .monitors import IonParameter, RTPParameter


# ============================================================================
# RCA Data Structures
# ============================================================================

class RootCause(Enum):
    """Common root causes in semiconductor processing."""
    # Ion Implantation
    ION_SOURCE_DEGRADATION = "ion_source_degradation"
    ION_OPTICS_MISALIGNMENT = "ion_optics_misalignment"
    ION_VACUUM_LEAK = "vacuum_leak"
    ION_MAGNET_DRIFT = "magnet_drift"
    ION_WAFER_HANDLING = "wafer_handling_issue"
    ION_SCAN_MALFUNCTION = "scan_malfunction"

    # RTP
    RTP_LAMP_FAILURE = "lamp_failure"
    RTP_PYROMETER_DRIFT = "pyrometer_drift"
    RTP_EMISSIVITY_CHANGE = "emissivity_change"
    RTP_GAS_FLOW_ISSUE = "gas_flow_issue"
    RTP_POWER_SUPPLY = "power_supply_issue"
    RTP_CHAMBER_CONTAMINATION = "chamber_contamination"
    RTP_THERMOCOUPLE_FAILURE = "thermocouple_failure"

    # Common
    EQUIPMENT_CALIBRATION = "calibration_drift"
    RECIPE_PARAMETER_ERROR = "recipe_parameter_error"
    OPERATOR_ERROR = "operator_error"
    MAINTENANCE_OVERDUE = "maintenance_overdue"
    ENVIRONMENTAL = "environmental_factor"
    UNKNOWN = "unknown"


@dataclass
class CorrectiveAction:
    """A corrective action to resolve an issue."""
    action_id: str
    description: str
    priority: int  # 1 = highest
    estimated_time_minutes: int
    requires_maintenance: bool = False
    requires_calibration: bool = False
    requires_approval: bool = False


@dataclass
class RCAPlaybook:
    """Root cause analysis playbook for a specific symptom."""
    symptom_pattern: str  # Regex pattern to match alert messages
    parameter_types: List[str]  # Affected parameter types
    likely_causes: List[RootCause]
    corrective_actions: List[CorrectiveAction]
    severity_escalation: Dict[str, int]  # Maps conditions to escalation level


@dataclass
class TriageResult:
    """Result of alert triage."""
    alert: SPCAlert
    matched_playbooks: List[RCAPlaybook]
    recommended_causes: List[RootCause]
    recommended_actions: List[CorrectiveAction]
    escalation_level: int  # 1=low, 2=medium, 3=high, 4=critical
    auto_resolvable: bool
    notes: str = ""


# ============================================================================
# Ion Implantation RCA Playbooks
# ============================================================================

ION_PLAYBOOKS = [
    # Beam current instability
    RCAPlaybook(
        symptom_pattern=r"beam_current.*beyond|beam_current.*trend",
        parameter_types=[IonParameter.BEAM_CURRENT_MA.value],
        likely_causes=[
            RootCause.ION_SOURCE_DEGRADATION,
            RootCause.ION_OPTICS_MISALIGNMENT,
            RootCause.ION_VACUUM_LEAK,
        ],
        corrective_actions=[
            CorrectiveAction(
                action_id="ION_CA_001",
                description="Check ion source filament/cathode condition and hours",
                priority=1,
                estimated_time_minutes=10,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="ION_CA_002",
                description="Verify extraction and acceleration voltages are at setpoint",
                priority=1,
                estimated_time_minutes=5,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="ION_CA_003",
                description="Check vacuum levels in source, analyzer, and process chambers",
                priority=2,
                estimated_time_minutes=10,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="ION_CA_004",
                description="Perform source maintenance (replace filament/clean cathode)",
                priority=3,
                estimated_time_minutes=120,
                requires_maintenance=True,
                requires_approval=True
            ),
        ],
        severity_escalation={"consecutive_alerts_3": 2, "consecutive_alerts_5": 3}
    ),

    # Dose uniformity issues
    RCAPlaybook(
        symptom_pattern=r"dose_uniformity.*beyond|uniformity.*low",
        parameter_types=[IonParameter.DOSE_UNIFORMITY_PCT.value],
        likely_causes=[
            RootCause.ION_SCAN_MALFUNCTION,
            RootCause.ION_OPTICS_MISALIGNMENT,
            RootCause.ION_WAFER_HANDLING,
        ],
        corrective_actions=[
            CorrectiveAction(
                action_id="ION_CA_010",
                description="Check scan waveforms (X and Y) for proper amplitude and frequency",
                priority=1,
                estimated_time_minutes=5,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="ION_CA_011",
                description="Verify wafer tilt and rotation settings (should be 7° tilt)",
                priority=1,
                estimated_time_minutes=5,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="ION_CA_012",
                description="Run uniformity calibration wafer and analyze dose map",
                priority=2,
                estimated_time_minutes=30,
                requires_calibration=True
            ),
            CorrectiveAction(
                action_id="ION_CA_013",
                description="Perform beam steering/parallelism calibration",
                priority=3,
                estimated_time_minutes=60,
                requires_calibration=True,
                requires_approval=True
            ),
        ],
        severity_escalation={"consecutive_alerts_2": 2, "consecutive_alerts_4": 3}
    ),

    # Vacuum issues
    RCAPlaybook(
        symptom_pattern=r"pressure.*beyond|pressure.*high",
        parameter_types=[
            IonParameter.SOURCE_PRESSURE_MTORR.value,
            IonParameter.ANALYZER_PRESSURE_MTORR.value,
            IonParameter.PROCESS_PRESSURE_MTORR.value,
        ],
        likely_causes=[
            RootCause.ION_VACUUM_LEAK,
            RootCause.EQUIPMENT_CALIBRATION,
            RootCause.MAINTENANCE_OVERDUE,
        ],
        corrective_actions=[
            CorrectiveAction(
                action_id="ION_CA_020",
                description="Check cryopump and turbopump operation",
                priority=1,
                estimated_time_minutes=10,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="ION_CA_021",
                description="Inspect O-rings and seals on chamber doors",
                priority=2,
                estimated_time_minutes=20,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="ION_CA_022",
                description="Perform helium leak check on vacuum system",
                priority=2,
                estimated_time_minutes=45,
                requires_maintenance=True
            ),
            CorrectiveAction(
                action_id="ION_CA_023",
                description="Calibrate vacuum gauges against reference standard",
                priority=3,
                estimated_time_minutes=30,
                requires_calibration=True
            ),
        ],
        severity_escalation={"consecutive_alerts_2": 3, "consecutive_alerts_3": 4}
    ),

    # Analyzer magnet drift
    RCAPlaybook(
        symptom_pattern=r"analyzer_field.*beyond|analyzer.*drift",
        parameter_types=[IonParameter.ANALYZER_FIELD_T.value],
        likely_causes=[
            RootCause.ION_MAGNET_DRIFT,
            RootCause.EQUIPMENT_CALIBRATION,
            RootCause.ENVIRONMENTAL,
        ],
        corrective_actions=[
            CorrectiveAction(
                action_id="ION_CA_030",
                description="Check magnet power supply current and voltage",
                priority=1,
                estimated_time_minutes=5,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="ION_CA_031",
                description="Verify Hall probe calibration and position",
                priority=2,
                estimated_time_minutes=15,
                requires_calibration=True
            ),
            CorrectiveAction(
                action_id="ION_CA_032",
                description="Check cooling water temperature (magnet stability)",
                priority=2,
                estimated_time_minutes=10,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="ION_CA_033",
                description="Recalibrate magnet field with reference probe",
                priority=3,
                estimated_time_minutes=60,
                requires_calibration=True,
                requires_approval=True
            ),
        ],
        severity_escalation={"consecutive_alerts_3": 2, "consecutive_alerts_5": 3}
    ),
]


# ============================================================================
# RTP RCA Playbooks
# ============================================================================

RTP_PLAYBOOKS = [
    # Ramp tracking error
    RCAPlaybook(
        symptom_pattern=r"ramp_tracking_error.*beyond|tracking.*high",
        parameter_types=[RTPParameter.RAMP_TRACKING_ERROR_C.value],
        likely_causes=[
            RootCause.RTP_PYROMETER_DRIFT,
            RootCause.RTP_LAMP_FAILURE,
            RootCause.RTP_POWER_SUPPLY,
            RootCause.EQUIPMENT_CALIBRATION,
        ],
        corrective_actions=[
            CorrectiveAction(
                action_id="RTP_CA_001",
                description="Verify all lamps are operational (check for failures)",
                priority=1,
                estimated_time_minutes=5,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="RTP_CA_002",
                description="Check pyrometer signal quality and lens cleanliness",
                priority=1,
                estimated_time_minutes=10,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="RTP_CA_003",
                description="Verify PID controller gains (Kp, Ki, Kd) are appropriate",
                priority=2,
                estimated_time_minutes=15,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="RTP_CA_004",
                description="Calibrate pyrometer against reference blackbody source",
                priority=3,
                estimated_time_minutes=45,
                requires_calibration=True,
                requires_approval=True
            ),
        ],
        severity_escalation={"consecutive_alerts_2": 2, "consecutive_alerts_4": 3}
    ),

    # Overshoot issues
    RCAPlaybook(
        symptom_pattern=r"overshoot.*beyond|overshoot.*high",
        parameter_types=[RTPParameter.OVERSHOOT_PCT.value],
        likely_causes=[
            RootCause.RTP_PYROMETER_DRIFT,
            RootCause.RTP_EMISSIVITY_CHANGE,
            RootCause.RECIPE_PARAMETER_ERROR,
            RootCause.EQUIPMENT_CALIBRATION,
        ],
        corrective_actions=[
            CorrectiveAction(
                action_id="RTP_CA_010",
                description="Review recipe ramp rate (reduce if too aggressive)",
                priority=1,
                estimated_time_minutes=5,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="RTP_CA_011",
                description="Reduce PID integral gain (Ki) to minimize overshoot",
                priority=1,
                estimated_time_minutes=10,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="RTP_CA_012",
                description="Check wafer emissivity setting (should match actual surface)",
                priority=2,
                estimated_time_minutes=10,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="RTP_CA_013",
                description="Enable MPC (Model Predictive Control) for overshoot constraints",
                priority=3,
                estimated_time_minutes=20,
                requires_maintenance=False
            ),
        ],
        severity_escalation={"consecutive_alerts_3": 2, "consecutive_alerts_5": 3}
    ),

    # Lamp power issues
    RCAPlaybook(
        symptom_pattern=r"lamp_power.*beyond|lamp.*high|lamp.*low",
        parameter_types=[RTPParameter.LAMP_POWER_PCT.value],
        likely_causes=[
            RootCause.RTP_LAMP_FAILURE,
            RootCause.RTP_EMISSIVITY_CHANGE,
            RootCause.RTP_CHAMBER_CONTAMINATION,
            RootCause.MAINTENANCE_OVERDUE,
        ],
        corrective_actions=[
            CorrectiveAction(
                action_id="RTP_CA_020",
                description="Check individual lamp power outputs (identify failed lamps)",
                priority=1,
                estimated_time_minutes=10,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="RTP_CA_021",
                description="Inspect quartz window for contamination or damage",
                priority=2,
                estimated_time_minutes=15,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="RTP_CA_022",
                description="Clean chamber and quartz window",
                priority=2,
                estimated_time_minutes=60,
                requires_maintenance=True
            ),
            CorrectiveAction(
                action_id="RTP_CA_023",
                description="Replace failed lamps and re-balance lamp zones",
                priority=3,
                estimated_time_minutes=120,
                requires_maintenance=True,
                requires_approval=True
            ),
        ],
        severity_escalation={"consecutive_alerts_2": 3, "consecutive_alerts_3": 4}
    ),

    # Emissivity drift
    RCAPlaybook(
        symptom_pattern=r"emissivity.*drift|emissivity.*beyond",
        parameter_types=[RTPParameter.EMISSIVITY_DRIFT.value],
        likely_causes=[
            RootCause.RTP_EMISSIVITY_CHANGE,
            RootCause.RTP_PYROMETER_DRIFT,
            RootCause.RTP_CHAMBER_CONTAMINATION,
        ],
        corrective_actions=[
            CorrectiveAction(
                action_id="RTP_CA_030",
                description="Verify wafer backside coating/condition (oxidation affects emissivity)",
                priority=1,
                estimated_time_minutes=10,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="RTP_CA_031",
                description="Update emissivity setting in recipe based on R2R data",
                priority=1,
                estimated_time_minutes=5,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="RTP_CA_032",
                description="Clean pyrometer optics and verify signal quality",
                priority=2,
                estimated_time_minutes=20,
                requires_maintenance=True
            ),
            CorrectiveAction(
                action_id="RTP_CA_033",
                description="Calibrate pyrometer with known emissivity reference",
                priority=3,
                estimated_time_minutes=45,
                requires_calibration=True,
                requires_approval=True
            ),
        ],
        severity_escalation={"consecutive_alerts_4": 2, "consecutive_alerts_6": 3}
    ),

    # Gas flow deviations
    RCAPlaybook(
        symptom_pattern=r"gas_flow.*deviation|gas.*beyond",
        parameter_types=[RTPParameter.GAS_FLOW_DEVIATION_SCCM.value],
        likely_causes=[
            RootCause.RTP_GAS_FLOW_ISSUE,
            RootCause.EQUIPMENT_CALIBRATION,
            RootCause.MAINTENANCE_OVERDUE,
        ],
        corrective_actions=[
            CorrectiveAction(
                action_id="RTP_CA_040",
                description="Check MFC (Mass Flow Controller) setpoint vs actual",
                priority=1,
                estimated_time_minutes=5,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="RTP_CA_041",
                description="Inspect gas lines for restrictions or leaks",
                priority=2,
                estimated_time_minutes=20,
                requires_maintenance=False
            ),
            CorrectiveAction(
                action_id="RTP_CA_042",
                description="Calibrate MFCs against reference flow meter",
                priority=2,
                estimated_time_minutes=30,
                requires_calibration=True
            ),
            CorrectiveAction(
                action_id="RTP_CA_043",
                description="Replace MFC if calibration fails",
                priority=3,
                estimated_time_minutes=90,
                requires_maintenance=True,
                requires_approval=True
            ),
        ],
        severity_escalation={"consecutive_alerts_2": 2, "consecutive_alerts_4": 3}
    ),
]


# ============================================================================
# Alert Triage Engine
# ============================================================================

class AlertTriageEngine:
    """Engine for triaging SPC alerts and recommending RCA playbooks."""

    def __init__(self, process_type: str = "ion"):
        """Initialize triage engine.

        Args:
            process_type: "ion" or "rtp"
        """
        self.process_type = process_type

        if process_type == "ion":
            self.playbooks = ION_PLAYBOOKS
        elif process_type == "rtp":
            self.playbooks = RTP_PLAYBOOKS
        else:
            self.playbooks = []

        # Alert history for escalation tracking
        self.alert_history: Dict[str, List[SPCAlert]] = {}

    def triage_alert(self, alert: SPCAlert) -> TriageResult:
        """Triage an SPC alert and recommend actions.

        Args:
            alert: SPCAlert to triage

        Returns:
            TriageResult with matched playbooks and recommendations
        """
        # Match playbooks
        matched_playbooks = self._match_playbooks(alert)

        # Extract recommended causes and actions
        recommended_causes = []
        recommended_actions = []

        for playbook in matched_playbooks:
            recommended_causes.extend(playbook.likely_causes)
            recommended_actions.extend(playbook.corrective_actions)

        # Deduplicate and sort by priority
        recommended_causes = list(set(recommended_causes))
        recommended_actions = sorted(
            recommended_actions,
            key=lambda a: a.priority
        )

        # Calculate escalation level
        escalation_level = self._calculate_escalation(alert, matched_playbooks)

        # Determine if auto-resolvable
        auto_resolvable = self._is_auto_resolvable(recommended_actions)

        # Generate notes
        notes = self._generate_notes(alert, matched_playbooks, escalation_level)

        result = TriageResult(
            alert=alert,
            matched_playbooks=matched_playbooks,
            recommended_causes=recommended_causes,
            recommended_actions=recommended_actions[:5],  # Top 5 actions
            escalation_level=escalation_level,
            auto_resolvable=auto_resolvable,
            notes=notes
        )

        # Store in history
        param_key = alert.parameter_name
        if param_key not in self.alert_history:
            self.alert_history[param_key] = []
        self.alert_history[param_key].append(alert)

        return result

    def _match_playbooks(self, alert: SPCAlert) -> List[RCAPlaybook]:
        """Match alert to applicable RCA playbooks."""
        matched = []

        for playbook in self.playbooks:
            # Check if message matches pattern
            if re.search(playbook.symptom_pattern, alert.message, re.IGNORECASE):
                matched.append(playbook)
                continue

            # Check if parameter type matches
            for param_type in playbook.parameter_types:
                if param_type in alert.parameter_name:
                    matched.append(playbook)
                    break

        return matched

    def _calculate_escalation(self, alert: SPCAlert, playbooks: List[RCAPlaybook]) -> int:
        """Calculate escalation level based on alert history and severity."""
        base_level = 1

        # Severity-based escalation
        if alert.severity == AlertSeverity.CRITICAL:
            base_level = 3
        elif alert.severity == AlertSeverity.WARNING:
            base_level = 2

        # History-based escalation
        param_key = alert.parameter_name
        if param_key in self.alert_history:
            recent_alerts = [
                a for a in self.alert_history[param_key][-10:]
                if a.rule_violated == alert.rule_violated
            ]

            consecutive_count = len(recent_alerts)

            for playbook in playbooks:
                for condition, level in playbook.severity_escalation.items():
                    if "consecutive_alerts" in condition:
                        threshold = int(condition.split("_")[-1])
                        if consecutive_count >= threshold:
                            base_level = max(base_level, level)

        return min(base_level, 4)  # Cap at 4

    def _is_auto_resolvable(self, actions: List[CorrectiveAction]) -> bool:
        """Determine if issue can be auto-resolved."""
        if not actions:
            return False

        # Top priority action should not require maintenance or approval
        top_action = actions[0]
        return not (top_action.requires_maintenance or top_action.requires_approval)

    def _generate_notes(
        self,
        alert: SPCAlert,
        playbooks: List[RCAPlaybook],
        escalation_level: int
    ) -> str:
        """Generate triage notes."""
        notes = []

        notes.append(f"Alert: {alert.message}")
        notes.append(f"Severity: {alert.severity.value}")
        notes.append(f"Rule: {alert.rule_violated.value}")
        notes.append(f"Escalation Level: {escalation_level}/4")

        if playbooks:
            notes.append(f"Matched {len(playbooks)} RCA playbook(s)")
        else:
            notes.append("No specific playbook matched - manual investigation required")

        if escalation_level >= 3:
            notes.append("⚠️ High escalation - notify process engineer")

        return " | ".join(notes)

    def get_alert_summary(self, lookback_count: int = 20) -> Dict:
        """Get summary of recent alerts and triage results."""
        summary = {
            "total_parameters": len(self.alert_history),
            "total_alerts": sum(len(alerts) for alerts in self.alert_history.values()),
            "parameters_with_alerts": len([k for k, v in self.alert_history.items() if v]),
        }

        # Count by severity (last N alerts)
        recent_all = []
        for alerts in self.alert_history.values():
            recent_all.extend(alerts[-lookback_count:])

        summary["recent_critical"] = sum(1 for a in recent_all if a.severity == AlertSeverity.CRITICAL)
        summary["recent_warning"] = sum(1 for a in recent_all if a.severity == AlertSeverity.WARNING)

        return summary


# Export
__all__ = [
    "RootCause",
    "CorrectiveAction",
    "RCAPlaybook",
    "TriageResult",
    "AlertTriageEngine",
    "ION_PLAYBOOKS",
    "RTP_PLAYBOOKS",
]
