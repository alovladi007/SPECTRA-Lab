"""
A/B Testing API Router
Provides endpoints for model comparison, tournaments, and A/B experiments
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import uuid
from datetime import datetime
from pathlib import Path
import json
import joblib
import numpy as np

# Import tournament and evaluation modules
from app.ml.eval.winrate_tournament import WinRateTournament
from app.ml.eval.eval_metrics import calculate_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ab-testing", tags=["A/B Testing"])

# In-memory storage
ab_jobs = {}
experiments = {}

# Pydantic models
class TournamentConfig(BaseModel):
    """Tournament configuration"""
    model_paths: List[str] = Field(..., description="List of model file paths to compare")
    model_names: Optional[List[str]] = Field(None, description="Custom names for models")
    test_data_path: str = Field(..., description="Path to test data CSV")
    target_column: str = Field(..., description="Target variable column name")
    tournament_type: str = Field("round_robin", description="Tournament type: round_robin, single_elimination")
    metrics: List[str] = Field(["r2", "rmse", "mae"], description="Metrics to compare")
    output_dir: str = Field("tournament_results", description="Output directory")

class HeadToHeadConfig(BaseModel):
    """Head-to-head comparison configuration"""
    model_a_path: str = Field(..., description="Path to model A")
    model_b_path: str = Field(..., description="Path to model B")
    model_a_name: str = Field("Model A", description="Name for model A")
    model_b_name: str = Field("Model B", description="Name for model B")
    test_data_path: str = Field(..., description="Path to test data CSV")
    target_column: str = Field(..., description="Target variable column name")
    metrics: List[str] = Field(["r2", "rmse", "mae"], description="Metrics to compare")

class ExperimentConfig(BaseModel):
    """A/B experiment configuration"""
    experiment_name: str = Field(..., description="Name of the experiment")
    description: Optional[str] = Field(None, description="Experiment description")
    model_paths: List[str] = Field(..., description="List of model variants to test")
    model_names: List[str] = Field(..., description="Names for each variant")
    test_data_path: str = Field(..., description="Path to test data")
    target_column: str = Field(..., description="Target variable column name")
    traffic_split: Optional[List[float]] = Field(None, description="Traffic split percentages (must sum to 1.0)")
    success_metrics: List[str] = Field(["r2"], description="Primary metrics for success")
    duration_days: int = Field(7, description="Experiment duration in days")

class JobResponse(BaseModel):
    """Job submission response"""
    job_id: str
    status: str
    created_at: str
    message: str

class ExperimentResponse(BaseModel):
    """Experiment creation response"""
    experiment_id: str
    experiment_name: str
    status: str
    created_at: str
    message: str


# Background task functions

def run_tournament(job_id: str, config: TournamentConfig):
    """Background task to run model tournament"""
    try:
        ab_jobs[job_id]["status"] = "running"
        ab_jobs[job_id]["started_at"] = datetime.now().isoformat()

        logger.info(f"Starting tournament job {job_id}")

        # Load models
        models = []
        model_names = config.model_names or [f"Model_{i+1}" for i in range(len(config.model_paths))]

        for model_path, name in zip(config.model_paths, model_names):
            try:
                model = joblib.load(model_path)
                models.append({"name": name, "model": model, "path": model_path})
            except Exception as e:
                logger.warning(f"Failed to load model {model_path}: {str(e)}")

        if len(models) < 2:
            raise ValueError("Need at least 2 valid models for tournament")

        # Load test data
        import pandas as pd
        test_data = pd.read_csv(config.test_data_path)
        X_test = test_data.drop(columns=[config.target_column])
        y_test = test_data[config.target_column]

        # Run tournament
        tournament = WinRateTournament(
            models=models,
            X_test=X_test,
            y_test=y_test,
            metrics=config.metrics,
            tournament_type=config.tournament_type,
            output_dir=config.output_dir
        )

        results = tournament.run()

        # Calculate win rates
        win_rates = {}
        matchups = []

        for i, model_i in enumerate(models):
            wins = 0
            total_matches = 0

            for j, model_j in enumerate(models):
                if i == j:
                    continue

                # Get predictions
                pred_i = model_i["model"].predict(X_test)
                pred_j = model_j["model"].predict(X_test)

                # Calculate metrics
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

                metrics_i = {
                    "r2": r2_score(y_test, pred_i),
                    "rmse": np.sqrt(mean_squared_error(y_test, pred_i)),
                    "mae": mean_absolute_error(y_test, pred_i)
                }

                metrics_j = {
                    "r2": r2_score(y_test, pred_j),
                    "rmse": np.sqrt(mean_squared_error(y_test, pred_j)),
                    "mae": mean_absolute_error(y_test, pred_j)
                }

                # Determine winner (higher is better for r2, lower is better for errors)
                winner = None
                for metric in config.metrics:
                    if metric == "r2":
                        if metrics_i[metric] > metrics_j[metric]:
                            winner = model_i["name"]
                            wins += 1
                        elif metrics_j[metric] > metrics_i[metric]:
                            winner = model_j["name"]
                    else:  # rmse, mae - lower is better
                        if metrics_i[metric] < metrics_j[metric]:
                            winner = model_i["name"]
                            wins += 1
                        elif metrics_j[metric] < metrics_i[metric]:
                            winner = model_j["name"]

                total_matches += 1

                matchups.append({
                    "model_a": model_i["name"],
                    "model_b": model_j["name"],
                    "metrics_a": metrics_i,
                    "metrics_b": metrics_j,
                    "winner": winner
                })

            win_rates[model_i["name"]] = {
                "wins": wins,
                "total": total_matches,
                "win_rate": wins / total_matches if total_matches > 0 else 0
            }

        # Determine champion
        champion = max(win_rates.items(), key=lambda x: x[1]["win_rate"])

        ab_jobs[job_id]["status"] = "completed"
        ab_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        ab_jobs[job_id]["results"] = {
            "tournament_type": config.tournament_type,
            "num_models": len(models),
            "champion": {
                "name": champion[0],
                "win_rate": champion[1]["win_rate"],
                "wins": champion[1]["wins"],
                "total_matches": champion[1]["total"]
            },
            "win_rates": win_rates,
            "matchups": matchups,
            "models": [{"name": m["name"], "path": m["path"]} for m in models]
        }

        logger.info(f"Tournament job {job_id} completed. Champion: {champion[0]}")

    except Exception as e:
        logger.error(f"Tournament job {job_id} failed: {str(e)}")
        ab_jobs[job_id]["status"] = "failed"
        ab_jobs[job_id]["error"] = str(e)
        ab_jobs[job_id]["completed_at"] = datetime.now().isoformat()


def run_head_to_head(job_id: str, config: HeadToHeadConfig):
    """Background task to run head-to-head comparison"""
    try:
        ab_jobs[job_id]["status"] = "running"
        ab_jobs[job_id]["started_at"] = datetime.now().isoformat()

        logger.info(f"Starting head-to-head job {job_id}")

        # Load models
        model_a = joblib.load(config.model_a_path)
        model_b = joblib.load(config.model_b_path)

        # Load test data
        import pandas as pd
        test_data = pd.read_csv(config.test_data_path)
        X_test = test_data.drop(columns=[config.target_column])
        y_test = test_data[config.target_column]

        # Get predictions
        pred_a = model_a.predict(X_test)
        pred_b = model_b.predict(X_test)

        # Calculate metrics
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        metrics_a = {
            "r2": float(r2_score(y_test, pred_a)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred_a))),
            "mae": float(mean_absolute_error(y_test, pred_a))
        }

        metrics_b = {
            "r2": float(r2_score(y_test, pred_b)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred_b))),
            "mae": float(mean_absolute_error(y_test, pred_b))
        }

        # Determine winner for each metric
        winners = {}
        improvements = {}

        for metric in config.metrics:
            if metric == "r2":
                # Higher is better
                winners[metric] = config.model_a_name if metrics_a[metric] > metrics_b[metric] else config.model_b_name
                improvement = ((metrics_a[metric] - metrics_b[metric]) / abs(metrics_b[metric])) * 100
            else:
                # Lower is better (rmse, mae)
                winners[metric] = config.model_a_name if metrics_a[metric] < metrics_b[metric] else config.model_b_name
                improvement = ((metrics_b[metric] - metrics_a[metric]) / metrics_b[metric]) * 100

            improvements[metric] = float(improvement)

        # Overall winner
        wins_a = sum(1 for w in winners.values() if w == config.model_a_name)
        wins_b = sum(1 for w in winners.values() if w == config.model_b_name)

        overall_winner = config.model_a_name if wins_a > wins_b else (
            config.model_b_name if wins_b > wins_a else "Tie"
        )

        ab_jobs[job_id]["status"] = "completed"
        ab_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        ab_jobs[job_id]["results"] = {
            "model_a": {
                "name": config.model_a_name,
                "path": config.model_a_path,
                "metrics": metrics_a
            },
            "model_b": {
                "name": config.model_b_name,
                "path": config.model_b_path,
                "metrics": metrics_b
            },
            "winners": winners,
            "improvements": improvements,
            "overall_winner": overall_winner,
            "summary": f"{overall_winner} wins {max(wins_a, wins_b)}-{min(wins_a, wins_b)}"
        }

        logger.info(f"Head-to-head job {job_id} completed. Winner: {overall_winner}")

    except Exception as e:
        logger.error(f"Head-to-head job {job_id} failed: {str(e)}")
        ab_jobs[job_id]["status"] = "failed"
        ab_jobs[job_id]["error"] = str(e)
        ab_jobs[job_id]["completed_at"] = datetime.now().isoformat()


# API Endpoints

@router.post("/tournament", response_model=JobResponse)
async def start_tournament(config: TournamentConfig, background_tasks: BackgroundTasks):
    """
    Start a model tournament

    Compare multiple models in round-robin or elimination tournament format.
    """
    if len(config.model_paths) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 models for tournament")

    job_id = str(uuid.uuid4())

    ab_jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "config": config.dict(),
        "type": "tournament"
    }

    background_tasks.add_task(run_tournament, job_id, config)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=ab_jobs[job_id]["created_at"],
        message=f"Tournament with {len(config.model_paths)} models created successfully"
    )


@router.post("/head-to-head", response_model=JobResponse)
async def start_head_to_head(config: HeadToHeadConfig, background_tasks: BackgroundTasks):
    """
    Run head-to-head comparison between two models

    Compare two models across multiple metrics.
    """
    job_id = str(uuid.uuid4())

    ab_jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "config": config.dict(),
        "type": "head_to_head"
    }

    background_tasks.add_task(run_head_to_head, job_id, config)

    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=ab_jobs[job_id]["created_at"],
        message=f"Head-to-head comparison: {config.model_a_name} vs {config.model_b_name}"
    )


@router.post("/experiment/create", response_model=ExperimentResponse)
async def create_experiment(config: ExperimentConfig):
    """
    Create a new A/B testing experiment

    Set up a multi-variant experiment with traffic splitting and success metrics.
    """
    if len(config.model_paths) != len(config.model_names):
        raise HTTPException(
            status_code=400,
            detail="Number of model paths must match number of model names"
        )

    if config.traffic_split:
        if len(config.traffic_split) != len(config.model_paths):
            raise HTTPException(
                status_code=400,
                detail="Traffic split must have same length as number of models"
            )
        if not np.isclose(sum(config.traffic_split), 1.0):
            raise HTTPException(
                status_code=400,
                detail="Traffic split must sum to 1.0"
            )
    else:
        # Equal split
        config.traffic_split = [1.0 / len(config.model_paths)] * len(config.model_paths)

    experiment_id = str(uuid.uuid4())

    experiments[experiment_id] = {
        "experiment_id": experiment_id,
        "experiment_name": config.experiment_name,
        "description": config.description,
        "status": "active",
        "created_at": datetime.now().isoformat(),
        "config": config.dict(),
        "variants": [
            {
                "variant_id": f"variant_{i}",
                "name": name,
                "model_path": path,
                "traffic_allocation": split
            }
            for i, (name, path, split) in enumerate(
                zip(config.model_names, config.model_paths, config.traffic_split)
            )
        ],
        "results": {
            "observations": 0,
            "variant_metrics": {}
        }
    }

    return ExperimentResponse(
        experiment_id=experiment_id,
        experiment_name=config.experiment_name,
        status="active",
        created_at=experiments[experiment_id]["created_at"],
        message=f"Experiment created with {len(config.model_paths)} variants"
    )


@router.get("/experiment/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """Get results for an A/B experiment"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")

    experiment = experiments[experiment_id]

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment["experiment_name"],
        "status": experiment["status"],
        "created_at": experiment["created_at"],
        "variants": experiment["variants"],
        "results": experiment["results"],
        "config": experiment["config"]
    }


@router.patch("/experiment/{experiment_id}/status")
async def update_experiment_status(experiment_id: str, status: str):
    """Update experiment status (active, paused, completed, cancelled)"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")

    valid_statuses = ["active", "paused", "completed", "cancelled"]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    experiments[experiment_id]["status"] = status
    experiments[experiment_id]["updated_at"] = datetime.now().isoformat()

    return {
        "experiment_id": experiment_id,
        "status": status,
        "message": f"Experiment status updated to {status}"
    }


@router.get("/experiment/{experiment_id}/winner")
async def get_experiment_winner(experiment_id: str):
    """
    Determine the winning variant based on success metrics

    Uses statistical significance testing to declare a winner.
    """
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")

    experiment = experiments[experiment_id]

    # Placeholder for statistical analysis
    return {
        "experiment_id": experiment_id,
        "winner": None,
        "confidence": 0.0,
        "message": "Statistical analysis not yet implemented",
        "suggestion": "Implement Bayesian A/B testing or frequentist hypothesis testing"
    }


@router.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Get status of an A/B testing job"""
    if job_id not in ab_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = ab_jobs[job_id]

    return {
        "job_id": job_id,
        "type": job["type"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "error": job.get("error")
    }


@router.get("/job/{job_id}/results")
async def get_job_results(job_id: str):
    """Get results of an A/B testing job"""
    if job_id not in ab_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = ab_jobs[job_id]

    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job is still {job['status']}. Results not available yet."
        )

    return {
        "job_id": job_id,
        "type": job["type"],
        "status": job["status"],
        "results": job.get("results"),
        "error": job.get("error")
    }


@router.get("/experiments")
async def list_experiments(status: Optional[str] = None, limit: int = 50):
    """List all A/B experiments"""
    filtered = experiments

    if status:
        filtered = {
            exp_id: exp for exp_id, exp in experiments.items()
            if exp["status"] == status
        }

    sorted_experiments = sorted(
        filtered.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    )[:limit]

    return {
        "total": len(filtered),
        "experiments": [
            {
                "experiment_id": exp_id,
                "experiment_name": exp["experiment_name"],
                "status": exp["status"],
                "created_at": exp["created_at"],
                "num_variants": len(exp["variants"])
            }
            for exp_id, exp in sorted_experiments
        ]
    }


@router.get("/jobs")
async def list_jobs(status: Optional[str] = None, job_type: Optional[str] = None, limit: int = 50):
    """List all A/B testing jobs"""
    filtered = ab_jobs

    if status:
        filtered = {
            job_id: job for job_id, job in filtered.items()
            if job["status"] == status
        }

    if job_type:
        filtered = {
            job_id: job for job_id, job in filtered.items()
            if job["type"] == job_type
        }

    sorted_jobs = sorted(
        filtered.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    )[:limit]

    return {
        "total": len(filtered),
        "jobs": [
            {
                "job_id": job_id,
                "type": job["type"],
                "status": job["status"],
                "created_at": job["created_at"],
                "completed_at": job.get("completed_at")
            }
            for job_id, job in sorted_jobs
        ]
    }


@router.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete an A/B testing job"""
    if job_id not in ab_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if ab_jobs[job_id]["status"] == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running job")

    del ab_jobs[job_id]
    return {"message": f"Job {job_id} deleted successfully"}


@router.delete("/experiment/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an experiment"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if experiments[experiment_id]["status"] == "active":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete active experiment. Pause or complete it first."
        )

    del experiments[experiment_id]
    return {"message": f"Experiment {experiment_id} deleted successfully"}
