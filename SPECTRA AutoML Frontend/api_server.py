"""
Flask API Backend for AutoML Dashboard
Connects the React frontend to the Python AutoML pipeline
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sys
import os
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from automl.train_automl import AutoMLPipeline
from semiconductor.data_handler import load_semiconductor_data

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Storage for active jobs
jobs = {}
job_lock = threading.Lock()


class JobManager:
    """Manages AutoML pipeline jobs"""
    
    def __init__(self):
        self.jobs = {}
    
    def create_job(self, config):
        """Create a new AutoML job"""
        job_id = str(uuid.uuid4())
        
        job = {
            'id': job_id,
            'status': 'queued',
            'progress': 0,
            'stage': 'Initializing...',
            'config': config,
            'created_at': datetime.now().isoformat(),
            'results': None,
            'error': None
        }
        
        self.jobs[job_id] = job
        return job_id
    
    def update_job(self, job_id, **kwargs):
        """Update job status"""
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)
    
    def get_job(self, job_id):
        """Get job details"""
        return self.jobs.get(job_id)


job_manager = JobManager()


def run_automl_pipeline(job_id, config):
    """Run AutoML pipeline in background thread"""
    try:
        # Update status
        job_manager.update_job(job_id, status='running', progress=0)
        
        # Load data
        job_manager.update_job(job_id, stage='Loading data...', progress=10)
        X_train, X_val, X_test, y_train, y_val, y_test = load_semiconductor_data(
            data_type=config.get('dataType', 'synthetic_yield'),
            test_size=0.2,
            val_size=0.1
        )
        
        # Model Selection
        if config.get('runModelSelection', True):
            job_manager.update_job(job_id, stage='Auto Model Selection...', progress=30)
            
            from automl.model_selection.auto_selector import AutoModelSelector
            selector = AutoModelSelector(
                task_type='regression',
                metric=config.get('metric', 'r2'),
                cv_folds=config.get('cvFolds', 5)
            )
            selection_results = selector.fit(X_train, y_train, X_test, y_test)
            best_model_type = selection_results['best_model']
        else:
            best_model_type = 'RandomForest'
            selection_results = None
        
        # Hyperparameter Tuning
        hyperopt_results = None
        if config.get('runHyperparameterTuning', True):
            job_manager.update_job(job_id, stage='Hyperparameter Tuning...', progress=60)
            
            from automl.hyperopt.tuner import AutoHyperparameterTuner
            tuner = AutoHyperparameterTuner(
                model_type=best_model_type,
                metric=config.get('metric', 'r2'),
                n_trials=config.get('nTrials', 50),
                cv_folds=config.get('cvFolds', 5),
                n_jobs=-1
            )
            hyperopt_results = tuner.optimize(X_train, y_train, X_test, y_test)
        
        # Neural Architecture Search
        nas_results = None
        if config.get('runNAS', False):
            job_manager.update_job(job_id, stage='Neural Architecture Search...', progress=80)
            
            from automl.nas.architecture_search import NeuralArchitectureSearch
            nas = NeuralArchitectureSearch(
                input_dim=X_train.shape[1],
                search_strategy='evolutionary'
            )
            nas_results = nas.search(X_train, y_train, X_val, y_val)
        
        # Compile results
        results = {
            'modelSelection': selection_results,
            'hyperparameterTuning': hyperopt_results,
            'nas': nas_results,
            'completed_at': datetime.now().isoformat()
        }
        
        # Update job as complete
        job_manager.update_job(
            job_id,
            status='complete',
            progress=100,
            stage='Complete!',
            results=results
        )
        
    except Exception as e:
        # Handle errors
        job_manager.update_job(
            job_id,
            status='failed',
            stage='Error',
            error=str(e)
        )


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AutoML Backend API',
        'version': '1.0.0'
    })


@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline():
    """Start a new AutoML pipeline"""
    try:
        config = request.json.get('pipeline_config', {})
        
        # Create job
        job_id = job_manager.create_job(config)
        
        # Start pipeline in background thread
        thread = threading.Thread(
            target=run_automl_pipeline,
            args=(job_id, config)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Pipeline started successfully'
        }), 202
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to start pipeline'
        }), 500


@app.route('/api/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    """Get pipeline progress"""
    job = job_manager.get_job(job_id)
    
    if not job:
        return jsonify({
            'error': 'Job not found'
        }), 404
    
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'stage': job['stage'],
        'created_at': job['created_at']
    })


@app.route('/api/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """Get pipeline results"""
    job = job_manager.get_job(job_id)
    
    if not job:
        return jsonify({
            'error': 'Job not found'
        }), 404
    
    if job['status'] != 'complete':
        return jsonify({
            'error': 'Job not complete',
            'status': job['status']
        }), 400
    
    return jsonify(job['results'])


@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    return jsonify([
        {
            'job_id': job_id,
            'status': job['status'],
            'progress': job['progress'],
            'created_at': job['created_at']
        }
        for job_id, job in job_manager.jobs.items()
    ])


@app.route('/api/job/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job"""
    if job_id in job_manager.jobs:
        del job_manager.jobs[job_id]
        return jsonify({
            'message': 'Job deleted successfully'
        })
    
    return jsonify({
        'error': 'Job not found'
    }), 404


@app.route('/api/config/presets', methods=['GET'])
def get_config_presets():
    """Get configuration presets"""
    return jsonify({
        'quickstart': {
            'runModelSelection': True,
            'runHyperparameterTuning': True,
            'runNAS': False,
            'nTrials': 20,
            'cvFolds': 3,
            'metric': 'r2'
        },
        'balanced': {
            'runModelSelection': True,
            'runHyperparameterTuning': True,
            'runNAS': False,
            'nTrials': 50,
            'cvFolds': 5,
            'metric': 'r2'
        },
        'thorough': {
            'runModelSelection': True,
            'runHyperparameterTuning': True,
            'runNAS': True,
            'nTrials': 100,
            'cvFolds': 5,
            'metric': 'r2'
        }
    })


@app.route('/api/models/available', methods=['GET'])
def get_available_models():
    """Get list of available models"""
    return jsonify({
        'models': [
            {
                'name': 'RandomForest',
                'description': 'Robust to outliers, good for tabular data',
                'speed': 'fast',
                'accuracy': 'high'
            },
            {
                'name': 'GradientBoosting',
                'description': 'High accuracy for complex patterns',
                'speed': 'medium',
                'accuracy': 'very_high'
            },
            {
                'name': 'MLP',
                'description': 'Neural network for large datasets',
                'speed': 'medium',
                'accuracy': 'very_high'
            },
            {
                'name': 'Ridge',
                'description': 'Fast linear model, interpretable',
                'speed': 'very_fast',
                'accuracy': 'medium'
            },
            {
                'name': 'SVR',
                'description': 'Non-linear, medium datasets',
                'speed': 'slow',
                'accuracy': 'high'
            }
        ]
    })


if __name__ == '__main__':
    print("="*60)
    print("AutoML Backend API Server")
    print("="*60)
    print(f"Server running on: http://localhost:8000")
    print(f"Health check: http://localhost:8000/api/health")
    print(f"API documentation: See BACKEND_API.md")
    print("="*60)
    
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True,
        threaded=True
    )
