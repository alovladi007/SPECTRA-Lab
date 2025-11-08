# AutoML Frontend Documentation 🎨

## Overview

A modern, interactive React-based dashboard for the AutoML semiconductor manufacturing platform. This frontend provides a complete user interface for configuring, monitoring, and analyzing AutoML pipeline results.

---

## 🚀 Quick Start

### Option 1: Standalone HTML (Easiest)

Simply open the HTML file in your browser:

```bash
# Open in default browser
open automl-dashboard.html

# Or on Windows
start automl-dashboard.html

# Or on Linux
xdg-open automl-dashboard.html
```

**That's it!** The dashboard runs entirely in the browser with no build process needed.

### Option 2: Local Development Server

For better performance and hot reloading:

```bash
# Using Python
python -m http.server 8000

# Then open: http://localhost:8000/automl-dashboard.html
```

---

## 🎯 Features

### 1. **Configuration Tab** ⚙️

Configure your AutoML pipeline with an intuitive interface:

- **Data Source Selection**
  - Use synthetic data (Wafer Yield or Defect Detection)
  - Upload custom CSV files
  
- **Pipeline Stages**
  - ☑️ Auto Model Selection (9+ algorithms)
  - ☑️ Hyperparameter Tuning (Bayesian optimization)
  - ☑️ Neural Architecture Search (optional)

- **Advanced Settings**
  - Optimization metric (R², RMSE, MAE, Accuracy)
  - Hardware selection (CPU/GPU)
  - Number of optimization trials (20-200)
  - Cross-validation folds (3-10)

- **Runtime Estimation**
  - Real-time estimate based on configuration
  - Helps plan pipeline execution

### 2. **Monitor Tab** 📊

Real-time pipeline execution monitoring:

- **Progress Bar** - Visual pipeline progress (0-100%)
- **Stage Tracking** - Current execution stage
- **Status Cards** - Individual stage completion status
- **Live Metrics**
  - Models evaluated
  - Best score so far
  - Trials completed
  - Time elapsed

### 3. **Results Tab** 📈

Comprehensive results visualization:

#### Summary Cards
- Best model identified
- Final R² score
- RMSE (error metric)
- Total execution time

#### Interactive Charts
1. **Model Performance Comparison** (Bar Chart)
   - Compare CV scores vs Test scores
   - All evaluated models side-by-side

2. **Optimization Progress** (Line Chart)
   - R² score improvement over trials
   - Visualize convergence

3. **Parameter Importance** (Progress Bars)
   - Which hyperparameters matter most
   - Percentage contribution

#### Data Tables
- **Detailed Model Comparison**
  - CV Score, Test Score, Inference Time, Complexity
  - Sortable, comprehensive view

- **Best Hyperparameters**
  - Final optimized parameter values
  - Easy to copy for reproduction

#### Export Options
- Download trained model (.pkl)
- Download PDF report
- Export configuration for reuse

---

## 🎨 UI/UX Features

### Design Elements
- **Modern Gradient UI** - Professional blue/indigo color scheme
- **Responsive Layout** - Works on desktop and tablets
- **Smooth Animations** - Progress bars, state transitions
- **Interactive Elements** - Hover effects, disabled states
- **Status Indicators** - Color-coded (green=complete, blue=running, gray=waiting)

### User Experience
- **Clear Navigation** - Tab-based interface
- **Real-time Feedback** - Live progress updates
- **Visual Hierarchy** - Important info stands out
- **Accessibility** - Keyboard navigation, clear labels
- **Error Prevention** - Disabled states, validation

---

## 📊 Data Flow

### 1. Configuration Phase
```
User Input → Pipeline Config → Validation → Ready to Run
```

### 2. Execution Phase
```
Run Button → Progress Updates → Stage Transitions → Completion
```

### 3. Results Phase
```
Complete → Generate Charts → Display Metrics → Export Options
```

---

## 🔧 Technical Stack

### Core Technologies
- **React 18** - UI framework
- **Recharts 2.5** - Data visualization
- **Tailwind CSS 3** - Styling
- **Lucide React** - Icons

### Key Libraries
- `react` - Component-based UI
- `recharts` - Bar charts, line charts
- `lucide-react` - Beautiful icons
- `tailwindcss` - Utility-first CSS

### Browser Support
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

---

## 🔌 Backend Integration

### Current Implementation (Demo Mode)

The frontend currently runs in **demo mode** with simulated data. Perfect for:
- UI/UX testing
- Feature demonstrations
- Design validation
- User training

### Connecting to Real Backend

To connect to your Python backend, modify the `runPipeline` function:

```javascript
const runPipeline = async () => {
  setIsRunning(true);
  setActiveTab('monitor');
  
  try {
    // Send configuration to backend
    const response = await fetch('http://localhost:8000/api/run-pipeline', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pipeline_config: pipeline,
        uploaded_file: uploadedFile
      })
    });
    
    const jobId = await response.json();
    
    // Poll for progress
    const pollInterval = setInterval(async () => {
      const progressRes = await fetch(`http://localhost:8000/api/progress/${jobId}`);
      const progressData = await progressRes.json();
      
      setProgress(progressData.progress);
      setCurrentStage(progressData.stage);
      
      if (progressData.status === 'complete') {
        clearInterval(pollInterval);
        
        // Fetch final results
        const resultsRes = await fetch(`http://localhost:8000/api/results/${jobId}`);
        const resultsData = await resultsRes.json();
        
        setResults(resultsData);
        setIsRunning(false);
        setActiveTab('results');
      }
    }, 1000);
    
  } catch (error) {
    console.error('Pipeline error:', error);
    setIsRunning(false);
  }
};
```

### Backend API Endpoints (Required)

```python
# Flask example
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline():
    config = request.json['pipeline_config']
    # Start AutoML pipeline
    job_id = start_automl_job(config)
    return jsonify({'job_id': job_id})

@app.route('/api/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    progress_data = get_job_progress(job_id)
    return jsonify(progress_data)

@app.route('/api/results/<job_id>', methods=['GET'])
def get_results(job_id):
    results = get_job_results(job_id)
    return jsonify(results)

if __name__ == '__main__':
    app.run(port=8000)
```

---

## 📝 Configuration Schema

### Pipeline Config Object

```javascript
{
  runModelSelection: true,        // Enable model selection
  runHyperparameterTuning: true,  // Enable hyperparameter tuning
  runNAS: false,                  // Enable neural architecture search
  dataType: 'synthetic_yield',    // Data type
  metric: 'r2',                   // Optimization metric
  nTrials: 50,                    // Number of trials
  cvFolds: 5,                     // Cross-validation folds
  device: 'cpu'                   // Hardware (cpu/cuda)
}
```

### Results Object Schema

```javascript
{
  modelSelection: {
    bestModel: string,
    bestScore: number,
    allCandidates: [{
      name: string,
      cvScore: number,
      testScore: number,
      inferenceTime: number,
      complexity: number
    }]
  },
  hyperparameterTuning: {
    modelType: string,
    bestCvScore: number,
    nTrials: number,
    bestParams: object,
    testMetrics: {
      r2: number,
      rmse: number,
      mae: number
    },
    paramImportance: object
  },
  optimizationHistory: [{
    trial: number,
    score: number
  }],
  timeMetrics: {
    totalTime: string,
    modelSelectionTime: string,
    hyperparameterTuningTime: string
  }
}
```

---

## 🎨 Customization

### Color Scheme

Current palette:
```css
Primary Blue: #3b82f6
Primary Indigo: #6366f1
Success Green: #10b981
Warning Orange: #f97316
Purple Accent: #8b5cf6
```

To change colors, modify the Tailwind classes:
- `bg-blue-600` → `bg-purple-600`
- `text-green-600` → `text-emerald-600`

### Branding

Update the header section:
```javascript
<h1 className="text-2xl font-bold text-gray-900">
  Your Company Name
</h1>
<p className="text-sm text-gray-600">
  Your Tagline
</p>
```

### Charts

Modify Recharts components:
```javascript
// Change chart colors
<Bar dataKey="cvScore" fill="#3b82f6" /> // Your color here

// Adjust dimensions
<ResponsiveContainer width="100%" height={400}>
```

---

## 🐛 Troubleshooting

### Charts Not Displaying
**Issue**: Recharts charts appear blank
**Solution**: Ensure Recharts CDN is loaded before React code
```html
<script src="https://unpkg.com/recharts@2.5.0/dist/Recharts.js"></script>
```

### Icons Not Showing
**Issue**: Lucide icons don't render
**Solution**: Check Lucide CDN is loaded
```html
<script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>
```

### Styling Issues
**Issue**: Tailwind classes not working
**Solution**: Verify Tailwind CDN link
```html
<script src="https://cdn.tailwindcss.com"></script>
```

### File Upload Not Working
**Issue**: CSV upload doesn't trigger
**Solution**: Check file input is not disabled and accept attribute is set
```html
<input type="file" accept=".csv" onChange={handleUpload} />
```

---

## 🚀 Deployment

### Static Hosting (Simplest)

1. **GitHub Pages**
   ```bash
   # Commit automl-dashboard.html to repo
   # Enable GitHub Pages in repo settings
   # Access at: https://username.github.io/repo/automl-dashboard.html
   ```

2. **Netlify**
   ```bash
   # Drag and drop automl-dashboard.html to Netlify
   # Get instant URL
   ```

3. **Vercel**
   ```bash
   vercel deploy automl-dashboard.html
   ```

### With Backend API

1. Update API endpoints in code
2. Enable CORS on backend
3. Deploy both frontend and backend
4. Use environment variables for API URL

---

## 📱 Mobile Responsiveness

The dashboard is optimized for:
- **Desktop**: Full layout (1920x1080+)
- **Laptop**: Compact layout (1366x768+)
- **Tablet**: Responsive grid (768x1024+)
- **Mobile**: Not optimized (use desktop view)

---

## 🎯 Future Enhancements

Potential improvements:
- [ ] Real-time WebSocket updates
- [ ] Dark mode toggle
- [ ] Export charts as images
- [ ] Compare multiple runs
- [ ] Saved configurations
- [ ] User authentication
- [ ] Mobile-optimized layout
- [ ] Multi-language support
- [ ] Keyboard shortcuts
- [ ] Advanced filtering

---

## 📄 License

MIT License - Same as the AutoML backend

---

## 🤝 Contributing

To add features:
1. Modify the React component
2. Test in browser
3. Update this documentation
4. Submit changes

---

## 📞 Support

- **Documentation**: This file
- **Demo**: Open `automl-dashboard.html`
- **Issues**: Check browser console for errors

---

## 🎉 Quick Tips

1. **Fast Testing**: Just open the HTML file, no build needed
2. **Custom Data**: Use the file upload for your own CSVs
3. **Export Results**: Download models and reports directly
4. **Configuration**: Save pipeline configs for later reuse
5. **Live Demo**: Run pipeline to see simulated execution

---

**Enjoy your AutoML Dashboard! 🚀**
