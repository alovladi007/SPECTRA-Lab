"""Automated retraining scheduler for VM models.

Manages retraining triggers based on drift detection, time-based schedules,
and performance degradation.
"""

import json
import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import time


# ============================================================================
# Retraining Scheduler Data Structures
# ============================================================================

class RetrainingTrigger(Enum):
    """Types of retraining triggers."""
    TIME_BASED = "time_based"  # Periodic retraining (e.g., weekly)
    DRIFT_BASED = "drift_based"  # Triggered by drift alerts
    PERFORMANCE_BASED = "performance_based"  # Triggered by performance degradation
    MANUAL = "manual"  # Manually triggered by operator


class RetrainingStatus(Enum):
    """Status of retraining job."""
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RetrainingPolicy:
    """Policy for model retraining."""
    model_name: str
    model_type: str  # "ion_vm", "rtp_vm"

    # Time-based triggers
    time_based_enabled: bool = True
    retraining_interval_days: int = 7  # Retrain weekly by default
    last_training_date: Optional[str] = None

    # Drift-based triggers
    drift_based_enabled: bool = True
    drift_threshold_psi: float = 0.35  # PSI threshold for retraining
    consecutive_drift_alerts: int = 3  # Number of consecutive alerts to trigger

    # Performance-based triggers
    performance_based_enabled: bool = True
    performance_metric: str = "mae"  # "mae", "rmse", "r2"
    performance_threshold: float = 20.0  # % degradation threshold
    baseline_performance: Optional[float] = None

    # Configuration
    min_training_samples: int = 1000  # Minimum samples required
    data_date_range_days: int = 90  # Use last 90 days of data
    auto_deploy: bool = False  # Auto-deploy after successful training
    notification_emails: List[str] = field(default_factory=list)


@dataclass
class RetrainingJob:
    """Retraining job record."""
    job_id: str
    model_name: str
    model_type: str
    trigger: RetrainingTrigger
    status: RetrainingStatus

    # Timing
    scheduled_time: str
    started_time: Optional[str] = None
    completed_time: Optional[str] = None

    # Training info
    training_samples: int = 0
    training_duration_sec: float = 0.0
    new_model_version: Optional[str] = None

    # Performance
    old_performance: Optional[Dict[str, float]] = None
    new_performance: Optional[Dict[str, float]] = None
    improvement_pct: Optional[float] = None

    # Metadata
    trigger_reason: str = ""
    error_message: Optional[str] = None
    created_by: str = "system"


# ============================================================================
# Retraining Scheduler
# ============================================================================

class RetrainingScheduler:
    """Automated retraining scheduler for VM models.

    Monitors drift alerts, performance metrics, and time-based schedules
    to automatically trigger model retraining.
    """

    def __init__(self, scheduler_dir: str = "./retraining_scheduler"):
        """Initialize retraining scheduler.

        Args:
            scheduler_dir: Directory to store scheduler state
        """
        self.scheduler_dir = scheduler_dir
        self.policies_dir = os.path.join(scheduler_dir, "policies")
        self.jobs_dir = os.path.join(scheduler_dir, "jobs")

        # Create directories
        os.makedirs(self.policies_dir, exist_ok=True)
        os.makedirs(self.jobs_dir, exist_ok=True)

        # In-memory state
        self.policies: Dict[str, RetrainingPolicy] = {}
        self.jobs: Dict[str, RetrainingJob] = {}
        self.drift_alert_history: Dict[str, List[Dict]] = {}  # model_name -> alerts

        # Training callback (set by user)
        self.training_callback: Optional[Callable] = None

        self._load_policies()
        self._load_jobs()

    def _load_policies(self):
        """Load retraining policies from disk."""
        for filename in os.listdir(self.policies_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.policies_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    policy = RetrainingPolicy(**data)
                    self.policies[policy.model_name] = policy

    def _load_jobs(self):
        """Load retraining jobs from disk."""
        for filename in os.listdir(self.jobs_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.jobs_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    job = RetrainingJob(
                        **data,
                        trigger=RetrainingTrigger(data["trigger"]),
                        status=RetrainingStatus(data["status"])
                    )
                    self.jobs[job.job_id] = job

    def register_policy(self, policy: RetrainingPolicy):
        """Register a retraining policy for a model.

        Args:
            policy: RetrainingPolicy to register
        """
        self.policies[policy.model_name] = policy

        # Save to disk
        filepath = os.path.join(self.policies_dir, f"{policy.model_name}_policy.json")
        with open(filepath, 'w') as f:
            json.dump(asdict(policy), f, indent=2)

    def set_training_callback(self, callback: Callable):
        """Set callback function for executing training.

        Callback signature: callback(model_name: str, model_type: str, config: Dict) -> Dict
        Returns: {"success": bool, "model_version": str, "metrics": Dict, "error": str}

        Args:
            callback: Training function
        """
        self.training_callback = callback

    def check_time_based_trigger(self, model_name: str) -> bool:
        """Check if time-based retraining is due.

        Args:
            model_name: Model name to check

        Returns:
            True if retraining is due
        """
        policy = self.policies.get(model_name)
        if not policy or not policy.time_based_enabled:
            return False

        if not policy.last_training_date:
            # Never trained, trigger immediately
            return True

        last_training = datetime.fromisoformat(policy.last_training_date)
        next_training = last_training + timedelta(days=policy.retraining_interval_days)

        return datetime.now() >= next_training

    def check_drift_based_trigger(self, model_name: str) -> bool:
        """Check if drift-based retraining should trigger.

        Args:
            model_name: Model name to check

        Returns:
            True if drift threshold exceeded
        """
        policy = self.policies.get(model_name)
        if not policy or not policy.drift_based_enabled:
            return False

        # Get recent drift alerts
        recent_alerts = self.drift_alert_history.get(model_name, [])

        if len(recent_alerts) < policy.consecutive_drift_alerts:
            return False

        # Check last N alerts
        last_n = recent_alerts[-policy.consecutive_drift_alerts:]

        # All must exceed threshold
        all_exceed = all(
            alert.get("psi_score", 0) >= policy.drift_threshold_psi
            for alert in last_n
        )

        return all_exceed

    def check_performance_based_trigger(
        self,
        model_name: str,
        current_performance: float
    ) -> bool:
        """Check if performance degradation triggers retraining.

        Args:
            model_name: Model name
            current_performance: Current performance metric value

        Returns:
            True if performance degraded beyond threshold
        """
        policy = self.policies.get(model_name)
        if not policy or not policy.performance_based_enabled:
            return False

        if policy.baseline_performance is None:
            return False

        # Calculate degradation percentage
        if policy.performance_metric in ["mae", "rmse"]:
            # Lower is better
            degradation_pct = ((current_performance - policy.baseline_performance)
                             / policy.baseline_performance * 100)
        else:  # r2
            # Higher is better
            degradation_pct = ((policy.baseline_performance - current_performance)
                             / policy.baseline_performance * 100)

        return degradation_pct >= policy.performance_threshold

    def report_drift_alert(self, model_name: str, drift_alert: Dict):
        """Report a drift alert for trigger evaluation.

        Args:
            model_name: Model name
            drift_alert: Drift alert dictionary with "psi_score", "timestamp", etc.
        """
        if model_name not in self.drift_alert_history:
            self.drift_alert_history[model_name] = []

        self.drift_alert_history[model_name].append(drift_alert)

        # Keep only recent alerts (last 24 hours)
        cutoff = time.time() - 86400
        self.drift_alert_history[model_name] = [
            alert for alert in self.drift_alert_history[model_name]
            if alert.get("timestamp", 0) >= cutoff
        ]

    def schedule_retraining(
        self,
        model_name: str,
        trigger: RetrainingTrigger,
        trigger_reason: str = "",
        created_by: str = "system"
    ) -> str:
        """Schedule a retraining job.

        Args:
            model_name: Model to retrain
            trigger: Type of trigger
            trigger_reason: Reason for retraining
            created_by: User/system that triggered

        Returns:
            Job ID
        """
        policy = self.policies.get(model_name)
        if not policy:
            raise ValueError(f"No retraining policy found for model '{model_name}'")

        # Generate job ID
        job_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create job
        job = RetrainingJob(
            job_id=job_id,
            model_name=model_name,
            model_type=policy.model_type,
            trigger=trigger,
            status=RetrainingStatus.SCHEDULED,
            scheduled_time=datetime.now().isoformat(),
            trigger_reason=trigger_reason,
            created_by=created_by
        )

        self.jobs[job_id] = job
        self._save_job(job)

        return job_id

    def execute_retraining(self, job_id: str) -> bool:
        """Execute a scheduled retraining job.

        Args:
            job_id: Job ID to execute

        Returns:
            True if training succeeded
        """
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if self.training_callback is None:
            raise RuntimeError("No training callback set. Use set_training_callback().")

        # Update job status
        job.status = RetrainingStatus.RUNNING
        job.started_time = datetime.now().isoformat()
        self._save_job(job)

        try:
            # Get policy
            policy = self.policies[job.model_name]

            # Prepare training config
            config = {
                "min_samples": policy.min_training_samples,
                "data_date_range_days": policy.data_date_range_days,
                "auto_deploy": policy.auto_deploy
            }

            # Execute training callback
            start_time = time.time()
            result = self.training_callback(job.model_name, job.model_type, config)
            training_duration = time.time() - start_time

            # Update job with results
            job.training_duration_sec = training_duration
            job.completed_time = datetime.now().isoformat()

            if result.get("success"):
                job.status = RetrainingStatus.COMPLETED
                job.new_model_version = result.get("model_version")
                job.new_performance = result.get("metrics", {})

                # Update policy
                policy.last_training_date = datetime.now().isoformat()
                if job.new_performance and policy.performance_metric in job.new_performance:
                    policy.baseline_performance = job.new_performance[policy.performance_metric]

                self._save_policy(policy)

                # Clear drift alert history after successful retraining
                self.drift_alert_history[job.model_name] = []

                return True
            else:
                job.status = RetrainingStatus.FAILED
                job.error_message = result.get("error", "Unknown error")
                return False

        except Exception as e:
            job.status = RetrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_time = datetime.now().isoformat()
            return False

        finally:
            self._save_job(job)

    def run_scheduler_cycle(self) -> List[str]:
        """Run one scheduler cycle to check all triggers.

        Returns:
            List of job IDs that were scheduled
        """
        scheduled_jobs = []

        for model_name, policy in self.policies.items():
            # Check time-based trigger
            if self.check_time_based_trigger(model_name):
                job_id = self.schedule_retraining(
                    model_name,
                    RetrainingTrigger.TIME_BASED,
                    f"Scheduled retraining (interval: {policy.retraining_interval_days} days)"
                )
                scheduled_jobs.append(job_id)
                continue

            # Check drift-based trigger
            if self.check_drift_based_trigger(model_name):
                job_id = self.schedule_retraining(
                    model_name,
                    RetrainingTrigger.DRIFT_BASED,
                    f"Drift threshold exceeded (PSI ≥ {policy.drift_threshold_psi})"
                )
                scheduled_jobs.append(job_id)
                continue

        return scheduled_jobs

    def get_scheduled_jobs(self) -> List[RetrainingJob]:
        """Get all scheduled (not yet started) jobs.

        Returns:
            List of RetrainingJob objects
        """
        return [
            job for job in self.jobs.values()
            if job.status == RetrainingStatus.SCHEDULED
        ]

    def get_job_status(self, job_id: str) -> Optional[RetrainingJob]:
        """Get status of a retraining job.

        Args:
            job_id: Job ID

        Returns:
            RetrainingJob if found
        """
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str):
        """Cancel a scheduled job.

        Args:
            job_id: Job ID to cancel
        """
        job = self.jobs.get(job_id)
        if job and job.status == RetrainingStatus.SCHEDULED:
            job.status = RetrainingStatus.CANCELLED
            self._save_job(job)

    def get_retraining_history(
        self,
        model_name: Optional[str] = None,
        limit: int = 10
    ) -> List[RetrainingJob]:
        """Get retraining history.

        Args:
            model_name: Filter by model name (optional)
            limit: Maximum number of jobs to return

        Returns:
            List of RetrainingJob objects, sorted by scheduled time
        """
        jobs = list(self.jobs.values())

        if model_name:
            jobs = [j for j in jobs if j.model_name == model_name]

        # Sort by scheduled time (most recent first)
        jobs.sort(key=lambda j: j.scheduled_time, reverse=True)

        return jobs[:limit]

    def _save_policy(self, policy: RetrainingPolicy):
        """Save policy to disk."""
        filepath = os.path.join(self.policies_dir, f"{policy.model_name}_policy.json")
        with open(filepath, 'w') as f:
            json.dump(asdict(policy), f, indent=2)

    def _save_job(self, job: RetrainingJob):
        """Save job to disk."""
        filepath = os.path.join(self.jobs_dir, f"{job.job_id}.json")

        data = asdict(job)
        data["trigger"] = job.trigger.value
        data["status"] = job.status.value

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# Helper Functions
# ============================================================================

def create_default_ion_vm_policy() -> RetrainingPolicy:
    """Create default retraining policy for Ion VM."""
    return RetrainingPolicy(
        model_name="ion_vm",
        model_type="ion_vm",
        time_based_enabled=True,
        retraining_interval_days=7,  # Weekly retraining
        drift_based_enabled=True,
        drift_threshold_psi=0.35,
        consecutive_drift_alerts=3,
        performance_based_enabled=True,
        performance_metric="mae",
        performance_threshold=20.0,  # 20% degradation
        min_training_samples=1000,
        data_date_range_days=90,
        auto_deploy=False  # Require manual approval for production
    )


def create_default_rtp_vm_policy() -> RetrainingPolicy:
    """Create default retraining policy for RTP VM."""
    return RetrainingPolicy(
        model_name="rtp_vm",
        model_type="rtp_vm",
        time_based_enabled=True,
        retraining_interval_days=7,  # Weekly retraining
        drift_based_enabled=True,
        drift_threshold_psi=0.35,
        consecutive_drift_alerts=3,
        performance_based_enabled=True,
        performance_metric="mae",
        performance_threshold=20.0,  # 20% degradation
        min_training_samples=1000,
        data_date_range_days=90,
        auto_deploy=False
    )


# Export
__all__ = [
    "RetrainingScheduler",
    "RetrainingPolicy",
    "RetrainingJob",
    "RetrainingTrigger",
    "RetrainingStatus",
    "create_default_ion_vm_policy",
    "create_default_rtp_vm_policy",
]
