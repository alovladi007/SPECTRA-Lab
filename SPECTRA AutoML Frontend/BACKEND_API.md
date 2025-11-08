# AutoML Backend API Documentation

Complete API reference for the AutoML Flask backend server.

---

## Base URL

```
http://localhost:8000/api
```

---

## Authentication

Currently: **None** (development mode)

For production, implement JWT tokens or API keys.

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "AutoML Backend API",
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK` - Service is healthy

---

### 2. Run Pipeline

**POST** `/run-pipeline`

Start a new AutoML pipeline execution.

**Request Body:**
```json
{
  "pipeline_config": {
    "runModelSelection": true,
    "runHyperparameterTuning": true,
    "runNAS": false,
    "dataType": "synthetic_yield",
    "metric": "r2",
    "nTrials": 50,
    "cvFolds": 5,
    "device": "cpu"
  }
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Pipeline started successfully"
}
```

**Status Codes:**
- `202 Accepted` - Pipeline started
- `500 Internal Server Error` - Failed to start

---

### 3. Get Progress

**GET** `/progress/<job_id>`

Get the current progress of a running pipeline.

**Parameters:**
- `job_id` (path) - UUID of the job

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 65,
  "stage": "Hyperparameter Tuning...",
  "created_at": "2024-11-06T14:30:00.000Z"
}
```

**Status Values:**
- `queued` - Waiting to start
- `running` - Currently executing
- `complete` - Finished successfully
- `failed` - Error occurred

**Status Codes:**
- `200 OK` - Progress retrieved
- `404 Not Found` - Job not found

---

### 4. Get Results

**GET** `/results/<job_id>`

Get the final results of a completed pipeline.

**Parameters:**
- `job_id` (path) - UUID of the job

**Response:**
```json
{
  "modelSelection": {
    "bestModel": "RandomForest",
    "bestScore": 0.9234,
    "allCandidates": [
      {
        "name": "RandomForest",
        "cvScore": 0.9234,
        "testScore": 0.9156,
        "inferenceTime": 0.234,
        "complexity": 50000
      }
    ]
  },
  "hyperparameterTuning": {
    "modelType": "RandomForest",
    "bestCvScore": 0.9412,
    "nTrials": 50,
    "bestParams": {
      "n_estimators": 350,
      "max_depth": 15,
      "min_samples_split": 3
    },
    "testMetrics": {
      "r2": 0.9412,
      "rmse": 2.34,
      "mae": 1.87
    },
    "paramImportance": {
      "n_estimators": 0.35,
      "max_depth": 0.28
    }
  },
  "completed_at": "2024-11-06T15:00:00.000Z"
}
```

**Status Codes:**
- `200 OK` - Results retrieved
- `400 Bad Request` - Job not complete
- `404 Not Found` - Job not found

---

### 5. List Jobs

**GET** `/jobs`

Get a list of all jobs.

**Response:**
```json
[
  {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "complete",
    "progress": 100,
    "created_at": "2024-11-06T14:30:00.000Z"
  },
  {
    "job_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
    "status": "running",
    "progress": 45,
    "created_at": "2024-11-06T15:00:00.000Z"
  }
]
```

**Status Codes:**
- `200 OK` - Jobs retrieved

---

### 6. Delete Job

**DELETE** `/job/<job_id>`

Delete a job and its results.

**Parameters:**
- `job_id` (path) - UUID of the job

**Response:**
```json
{
  "message": "Job deleted successfully"
}
```

**Status Codes:**
- `200 OK` - Job deleted
- `404 Not Found` - Job not found

---

### 7. Get Configuration Presets

**GET** `/config/presets`

Get predefined configuration presets.

**Response:**
```json
{
  "quickstart": {
    "runModelSelection": true,
    "runHyperparameterTuning": true,
    "runNAS": false,
    "nTrials": 20,
    "cvFolds": 3,
    "metric": "r2"
  },
  "balanced": {
    "runModelSelection": true,
    "runHyperparameterTuning": true,
    "runNAS": false,
    "nTrials": 50,
    "cvFolds": 5,
    "metric": "r2"
  },
  "thorough": {
    "runModelSelection": true,
    "runHyperparameterTuning": true,
    "runNAS": true,
    "nTrials": 100,
    "cvFolds": 5,
    "metric": "r2"
  }
}
```

**Status Codes:**
- `200 OK` - Presets retrieved

---

### 8. Get Available Models

**GET** `/models/available`

Get information about available ML models.

**Response:**
```json
{
  "models": [
    {
      "name": "RandomForest",
      "description": "Robust to outliers, good for tabular data",
      "speed": "fast",
      "accuracy": "high"
    },
    {
      "name": "GradientBoosting",
      "description": "High accuracy for complex patterns",
      "speed": "medium",
      "accuracy": "very_high"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Models retrieved

---

## Usage Examples

### Example 1: Run Complete Pipeline

```bash
# Start pipeline
curl -X POST http://localhost:8000/api/run-pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_config": {
      "runModelSelection": true,
      "runHyperparameterTuning": true,
      "runNAS": false,
      "nTrials": 50,
      "metric": "r2"
    }
  }'

# Response:
# {"job_id": "abc-123", "status": "queued"}

# Check progress
curl http://localhost:8000/api/progress/abc-123

# Get results when complete
curl http://localhost:8000/api/results/abc-123
```

### Example 2: JavaScript/Fetch

```javascript
// Start pipeline
const response = await fetch('http://localhost:8000/api/run-pipeline', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    pipeline_config: {
      runModelSelection: true,
      runHyperparameterTuning: true,
      nTrials: 50
    }
  })
});

const { job_id } = await response.json();

// Poll for progress
const checkProgress = async () => {
  const progressRes = await fetch(`http://localhost:8000/api/progress/${job_id}`);
  const data = await progressRes.json();
  
  console.log(`Progress: ${data.progress}% - ${data.stage}`);
  
  if (data.status === 'complete') {
    const resultsRes = await fetch(`http://localhost:8000/api/results/${job_id}`);
    const results = await resultsRes.json();
    console.log('Results:', results);
  } else {
    setTimeout(checkProgress, 2000); // Check every 2 seconds
  }
};

checkProgress();
```

### Example 3: Python/Requests

```python
import requests
import time

# Start pipeline
response = requests.post(
    'http://localhost:8000/api/run-pipeline',
    json={
        'pipeline_config': {
            'runModelSelection': True,
            'runHyperparameterTuning': True,
            'nTrials': 50
        }
    }
)

job_id = response.json()['job_id']
print(f"Job started: {job_id}")

# Poll for progress
while True:
    progress_response = requests.get(
        f'http://localhost:8000/api/progress/{job_id}'
    )
    data = progress_response.json()
    
    print(f"Progress: {data['progress']}% - {data['stage']}")
    
    if data['status'] == 'complete':
        # Get results
        results_response = requests.get(
            f'http://localhost:8000/api/results/{job_id}'
        )
        results = results_response.json()
        print("Results:", results)
        break
    
    time.sleep(2)
```

---

## Error Responses

All errors return this format:

```json
{
  "error": "Error description",
  "message": "User-friendly message"
}
```

**Common Error Codes:**
- `400 Bad Request` - Invalid input
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

---

## Rate Limiting

**Current:** No rate limiting (development)

**Production:** Implement rate limiting:
- 100 requests per minute per IP
- 10 concurrent pipelines per user

---

## CORS Configuration

**Current:** CORS enabled for all origins

**Production:** Restrict to specific domains:
```python
CORS(app, origins=['https://yourdomain.com'])
```

---

## WebSocket Support (Future)

For real-time progress updates:

```javascript
const socket = new WebSocket('ws://localhost:8000/ws/progress');

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress}%`);
};
```

---

## Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-flask

# Run tests
pytest tests/test_api.py -v
```

### Example Test

```python
def test_health_check(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_run_pipeline(client):
    response = client.post('/api/run-pipeline', json={
        'pipeline_config': {
            'runModelSelection': True,
            'nTrials': 10
        }
    })
    assert response.status_code == 202
    assert 'job_id' in response.json
```

---

## Production Deployment

### 1. Using Gunicorn

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:8000 api_server:app
```

### 2. Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "api_server:app"]
```

### 3. Environment Variables

```bash
# .env file
FLASK_ENV=production
API_PORT=8000
CORS_ORIGINS=https://yourdomain.com
MAX_CONCURRENT_JOBS=5
```

---

## Monitoring

### Health Check Endpoint

Monitor service health:
```bash
watch -n 5 'curl http://localhost:8000/api/health'
```

### Logging

Configure structured logging:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## Security Considerations

For production:

1. **Authentication**: Implement JWT tokens
2. **HTTPS**: Use SSL/TLS certificates
3. **Input Validation**: Sanitize all inputs
4. **Rate Limiting**: Prevent abuse
5. **CORS**: Restrict to known domains
6. **Logging**: Log all API access
7. **Error Messages**: Don't expose internal details

---

## Support

- **API Issues**: Check logs in `/var/log/automl-api.log`
- **Performance**: Monitor with `/api/health` endpoint
- **Questions**: See INTEGRATION_GUIDE.md

---

**API Version**: 1.0.0  
**Last Updated**: November 2024
