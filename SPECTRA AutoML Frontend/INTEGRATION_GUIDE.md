# Complete Integration Guide: Frontend + Backend

This guide shows you how to connect the React frontend to the Flask backend for a fully functional AutoML platform.

---

## 📋 Prerequisites

### Software Requirements
- Python 3.10+
- Node.js 16+ (optional, for development)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Python Dependencies
```bash
pip install flask flask-cors
pip install -r requirements.txt  # AutoML dependencies
```

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Start the Backend

```bash
# Navigate to your project directory
cd /path/to/automl-project

# Start the Flask API server
python api_server.py
```

**Expected Output:**
```
============================================================
AutoML Backend API Server
============================================================
Server running on: http://localhost:8000
Health check: http://localhost:8000/api/health
API documentation: See BACKEND_API.md
============================================================
 * Running on http://0.0.0.0:8000/ (Press CTRL+C to quit)
```

### Step 2: Open the Frontend

```bash
# Simply open the HTML file in your browser
open automl-dashboard.html

# Or on Windows
start automl-dashboard.html

# Or on Linux
xdg-open automl-dashboard.html
```

### Step 3: Test the Connection

1. **Configure Pipeline**: Adjust settings in the Configure tab
2. **Run Pipeline**: Click "Run AutoML Pipeline"
3. **Monitor Progress**: Watch real-time updates
4. **View Results**: See detailed analysis when complete

---

## 🔧 Configuration

### Backend Configuration (api_server.py)

Update these settings:

```python
# Server settings
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 8000       # API port

# CORS settings (for production)
CORS(app, origins=['https://yourdomain.com'])

# Job settings
MAX_CONCURRENT_JOBS = 5
JOB_TIMEOUT_HOURS = 2
```

### Frontend Configuration (automl-dashboard.html)

Update the API endpoint in the frontend code:

```javascript
// Find this section in the HTML file
const API_BASE_URL = 'http://localhost:8000/api';

// For production
const API_BASE_URL = 'https://api.yourdomain.com/api';
```

---

## 🔌 Connecting Frontend to Backend

### Method 1: Direct HTML Modification

Edit `automl-dashboard.html` and update the `runPipeline` function:

```javascript
const runPipeline = async () => {
  setIsRunning(true);
  setActiveTab('monitor');
  
  try {
    // Call backend API
    const response = await fetch('http://localhost:8000/api/run-pipeline', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        pipeline_config: pipeline
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    const jobId = data.job_id;
    
    // Poll for progress
    const pollInterval = setInterval(async () => {
      try {
        const progressRes = await fetch(
          `http://localhost:8000/api/progress/${jobId}`
        );
        const progressData = await progressRes.json();
        
        setProgress(progressData.progress);
        setCurrentStage(progressData.stage);
        
        if (progressData.status === 'complete') {
          clearInterval(pollInterval);
          
          // Get results
          const resultsRes = await fetch(
            `http://localhost:8000/api/results/${jobId}`
          );
          const resultsData = await resultsRes.json();
          
          setResults(resultsData);
          setIsRunning(false);
          setActiveTab('results');
        } else if (progressData.status === 'failed') {
          clearInterval(pollInterval);
          alert('Pipeline failed: ' + progressData.error);
          setIsRunning(false);
        }
      } catch (error) {
        console.error('Progress check error:', error);
      }
    }, 2000); // Poll every 2 seconds
    
  } catch (error) {
    console.error('Pipeline start error:', error);
    alert('Failed to start pipeline: ' + error.message);
    setIsRunning(false);
  }
};
```

### Method 2: Create Separate JS File

Create `automl-api.js`:

```javascript
// automl-api.js
const AutoMLAPI = {
  baseURL: 'http://localhost:8000/api',
  
  async startPipeline(config) {
    const response = await fetch(`${this.baseURL}/run-pipeline`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pipeline_config: config })
    });
    return response.json();
  },
  
  async getProgress(jobId) {
    const response = await fetch(`${this.baseURL}/progress/${jobId}`);
    return response.json();
  },
  
  async getResults(jobId) {
    const response = await fetch(`${this.baseURL}/results/${jobId}`);
    return response.json();
  },
  
  async listJobs() {
    const response = await fetch(`${this.baseURL}/jobs`);
    return response.json();
  }
};

// Usage in main component
const runPipeline = async () => {
  const data = await AutoMLAPI.startPipeline(pipeline);
  const jobId = data.job_id;
  
  // Poll for updates
  const poll = setInterval(async () => {
    const progress = await AutoMLAPI.getProgress(jobId);
    updateUI(progress);
    
    if (progress.status === 'complete') {
      clearInterval(poll);
      const results = await AutoMLAPI.getResults(jobId);
      displayResults(results);
    }
  }, 2000);
};
```

---

## 📊 Data Flow Architecture

```
┌─────────────────┐
│  React Frontend │
│  (Browser)      │
└────────┬────────┘
         │ HTTP/JSON
         ↓
┌─────────────────┐
│  Flask API      │
│  (Port 8000)    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  AutoML Pipeline│
│  (Background)   │
└─────────────────┘
```

### Request Flow

1. **User Action**: Click "Run Pipeline"
2. **Frontend**: Send POST to `/api/run-pipeline`
3. **Backend**: Create job, start thread, return job_id
4. **Frontend**: Poll `/api/progress/{job_id}` every 2s
5. **Backend**: Update progress in background
6. **Frontend**: When complete, fetch `/api/results/{job_id}`
7. **Backend**: Return final results
8. **Frontend**: Display charts and metrics

---

## 🔒 Security Configuration

### Development Mode

```python
# api_server.py
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins
```

### Production Mode

```python
# api_server.py
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=[
    'https://yourdomain.com',
    'https://app.yourdomain.com'
])

# Add authentication
from functools import wraps
from flask import request

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/run-pipeline', methods=['POST'])
@require_api_key
def run_pipeline():
    # ... existing code
```

---

## 🐛 Troubleshooting

### Issue 1: CORS Errors

**Error**: `Access-Control-Allow-Origin` header is missing

**Solution**:
```bash
# Install flask-cors
pip install flask-cors

# In api_server.py
from flask_cors import CORS
CORS(app)
```

### Issue 2: Connection Refused

**Error**: `Failed to connect to localhost:8000`

**Solution**:
```bash
# Check if backend is running
curl http://localhost:8000/api/health

# If not running, start it
python api_server.py
```

### Issue 3: Module Not Found

**Error**: `ModuleNotFoundError: No module named 'automl'`

**Solution**:
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/project"

# Or in api_server.py
import sys
sys.path.append('/path/to/project')
```

### Issue 4: Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
python api_server.py --port 8001
```

---

## 🚀 Deployment

### Option 1: Simple Deployment

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 api_server:app
```

### Option 2: Docker Deployment

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install flask flask-cors gunicorn

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "api_server:app"]
```

**Build and run**:
```bash
# Build image
docker build -t automl-backend .

# Run container
docker run -p 8000:8000 automl-backend
```

### Option 3: Cloud Deployment

#### Deploy to Heroku

```bash
# Create Procfile
echo "web: gunicorn api_server:app" > Procfile

# Deploy
heroku create automl-backend
git push heroku main
```

#### Deploy to AWS EC2

```bash
# SSH into EC2 instance
ssh ec2-user@your-instance.amazonaws.com

# Clone repository
git clone https://github.com/your-repo/automl.git
cd automl

# Install dependencies
pip install -r requirements.txt

# Run with systemd
sudo nano /etc/systemd/system/automl-api.service

# Add service configuration
[Unit]
Description=AutoML API Server
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/automl
ExecStart=/usr/bin/python3 api_server.py
Restart=always

[Install]
WantedBy=multi-user.target

# Start service
sudo systemctl start automl-api
sudo systemctl enable automl-api
```

---

## 📈 Monitoring

### Health Check

```bash
# Periodic health check
watch -n 5 'curl -s http://localhost:8000/api/health | jq'
```

### Logging

```python
# api_server.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automl-api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline():
    logger.info(f"Pipeline started with config: {request.json}")
    # ... existing code
```

---

## 🧪 Testing

### Test Backend

```bash
# Health check
curl http://localhost:8000/api/health

# Start pipeline
curl -X POST http://localhost:8000/api/run-pipeline \
  -H "Content-Type: application/json" \
  -d '{"pipeline_config": {"nTrials": 10}}'

# Check progress
curl http://localhost:8000/api/progress/<job_id>
```

### Test Frontend

1. Open `automl-dashboard.html` in browser
2. Open Developer Tools (F12)
3. Go to Network tab
4. Run pipeline and observe API calls
5. Check Console for any errors

---

## 📚 Additional Resources

- **Frontend README**: `FRONTEND_README.md`
- **Backend API**: `BACKEND_API.md`
- **AutoML Guide**: `AUTOML_README.md`
- **Quick Reference**: `QUICK_REFERENCE.md`

---

## 🎯 Next Steps

1. ✅ **Test Locally**: Run backend and frontend together
2. ✅ **Connect APIs**: Update frontend to use backend
3. ✅ **Add Features**: File upload, export, history
4. ✅ **Deploy**: Choose deployment method
5. ✅ **Monitor**: Set up logging and health checks

---

## 💡 Pro Tips

1. **Development**: Use `flask run --debug` for auto-reload
2. **Testing**: Keep browser DevTools open to debug API calls
3. **Performance**: Use WebSockets for real-time updates
4. **Security**: Always use HTTPS in production
5. **Caching**: Cache model results to avoid retraining

---

## 📞 Support

**Having issues?**
1. Check browser console (F12)
2. Check backend logs (`automl-api.log`)
3. Verify CORS is enabled
4. Test API with curl first
5. Refer to troubleshooting section

---

**Integration Complete! 🎉**

Your AutoML platform is now fully connected and ready to use!
