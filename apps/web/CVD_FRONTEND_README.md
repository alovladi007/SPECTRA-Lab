# CVD Frontend - Complete Implementation

## Overview

This document describes the complete CVD (Chemical Vapor Deposition) frontend implementation with dedicated pages, specialized metric components, and real-time WebSocket integration.

## 📁 Project Structure

```
apps/web/src/
├── app/cvd/
│   ├── page.tsx                    # Overview dashboard
│   ├── recipes/
│   │   └── page.tsx                # Recipe list & editor
│   ├── runs/
│   │   ├── page.tsx                # Runs list
│   │   └── [id]/page.tsx           # Run detail
│   └── results/
│       └── [id]/page.tsx           # Results deep-dive
├── components/cvd/
│   ├── metrics/
│   │   ├── ThicknessGauge.tsx      # Thickness visualization
│   │   ├── StressBar.tsx           # Stress axis visualization
│   │   ├── AdhesionChip.tsx        # Adhesion class badges
│   │   ├── WaferMap.tsx            # 2D wafer heatmap
│   │   └── AlertBanner.tsx         # Alert notifications
│   └── RealTimeMonitor.tsx         # Live WebSocket updates
└── hooks/
    └── useCVDWebSocket.ts          # WebSocket integration hook
```

---

## 🎨 Components

### 1. ThicknessGauge

**Location:** `apps/web/src/components/cvd/metrics/ThicknessGauge.tsx`

**Purpose:** Display target vs actual thickness with uniformity visualization

**Features:**
- SVG-based circular gauge (180x180px)
- Target vs actual thickness comparison
- Deviation calculation with color coding (green = in spec, red = out of spec)
- Uniformity ring visualization (outer circle, color-coded)
- Metrics grid: target, deviation %, WIW uniformity
- Configurable tolerance (default 5%)

**Props:**
```typescript
interface ThicknessGaugeProps {
  actual: number;              // Actual thickness (nm)
  target: number;              // Target thickness (nm)
  uniformity: number;          // WIW uniformity (%)
  tolerance?: number;          // Tolerance (%) - default 5%
  showUniformityRing?: boolean;
  className?: string;
}
```

**Uniformity Color Coding:**
- `< 2%`: Green (Excellent)
- `2-5%`: Yellow (Good)
- `> 5%`: Red (Poor)

**Usage:**
```tsx
<ThicknessGauge
  actual={98.5}
  target={100}
  uniformity={1.8}
  tolerance={5}
/>
```

---

### 2. StressBar

**Location:** `apps/web/src/components/cvd/metrics/StressBar.tsx`

**Purpose:** Horizontal stress axis with safe zone highlighting

**Features:**
- Horizontal bar showing compressive (left) to tensile (right) axis
- Safe zone highlighted region (-400 to +300 MPa by default)
- Current stress indicator with color coding
- Risk warnings for out-of-spec stress
- Scale labels and tooltips

**Props:**
```typescript
interface StressBarProps {
  stress: number;              // Current stress (MPa)
  safeZoneMin?: number;        // Safe zone min (default -400)
  safeZoneMax?: number;        // Safe zone max (default 300)
  rangeMin?: number;           // Display range min (default -800)
  rangeMax?: number;           // Display range max (default 600)
  showLabels?: boolean;
  className?: string;
}
```

**Stress Classification:**
- `< -500 MPa`: Critical Compressive
- `-500 to -400 MPa`: High Compressive (warning)
- `-400 to +300 MPa`: Normal (safe zone)
- `+300 to +500 MPa`: High Tensile (warning)
- `> +500 MPa`: Critical Tensile

**Usage:**
```tsx
<StressBar
  stress={-185}
  safeZoneMin={-400}
  safeZoneMax={300}
/>
```

---

### 3. AdhesionChip

**Location:** `apps/web/src/components/cvd/metrics/AdhesionChip.tsx`

**Purpose:** Color-coded adhesion class badges with test method information

**Features:**
- Color-coded badges (green/blue/yellow/red)
- Test method tooltips (ASTM standards)
- Multiple size variants (sm/md/lg)
- Risk indicators for poor adhesion
- Helper variants: AdhesionBadge, AdhesionDetail

**Props:**
```typescript
interface AdhesionChipProps {
  score: number;                      // Adhesion score (0-100)
  adhesionClass?: AdhesionClass;      // Override class
  testMethod?: AdhesionTestMethod;
  showScore?: boolean;
  showIcon?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}
```

**Adhesion Classification:**
- `80-100`: Excellent (green)
- `60-80`: Good (blue)
- `40-60`: Fair (yellow)
- `0-40`: Poor (red)

**Test Methods:**
- Tape Test (ASTM D3359)
- Scratch Test (ASTM C1624)
- Pull-Off Test (ASTM D4541)
- Four-Point Bend (ASTM E290)

**Usage:**
```tsx
<AdhesionChip
  score={88}
  testMethod={TEST_METHODS.TAPE_TEST}
  size="md"
/>
```

---

### 4. WaferMap

**Location:** `apps/web/src/components/cvd/metrics/WaferMap.tsx`

**Purpose:** 2D wafer visualization with thickness/stress heatmap overlay

**Features:**
- Circular wafer visualization (300mm default)
- Heatmap overlay with multiple color scales
- Interactive tooltips for measurement points
- Grid overlay with orientation marker (wafer flat)
- Statistics display (min/max/mean/uniformity)
- Outlier detection (>2σ from mean)
- Multiple size variants (sm/md/lg)

**Props:**
```typescript
interface WaferMapProps {
  points: WaferPoint[];
  waferDiameter?: number;              // mm (default 300)
  parameter?: string;                  // "Thickness", "Stress", etc.
  unit?: string;                       // "nm", "MPa", etc.
  colorScale?: "viridis" | "rdylgn" | "thermal" | "blues";
  showLegend?: boolean;
  showGrid?: boolean;
  valueRange?: [number, number];
  highlightOutliers?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}
```

**Color Scales:**
- `viridis`: Purple-blue-green-yellow (general purpose)
- `rdylgn`: Red-yellow-green (diverging)
- `thermal`: Blue-cyan-yellow-red (thermal)
- `blues`: Blue monochrome

**Helper Functions:**
```typescript
// Generate standard measurement patterns
generateWaferPoints.fivePoint(radius);      // 5-point
generateWaferPoints.ninePoint(radius);      // 9-point
generateWaferPoints.fortyNinePoint(radius); // 49-point
```

**Usage:**
```tsx
const points = generateWaferPoints.ninePoint(150).map(p => ({
  ...p,
  value: 100 + (Math.random() - 0.5) * 4
}));

<WaferMap
  points={points}
  parameter="Thickness"
  unit="nm"
  colorScale="viridis"
  showLegend={true}
  size="lg"
/>
```

---

### 5. AlertBanner

**Location:** `apps/web/src/components/cvd/metrics/AlertBanner.tsx`

**Purpose:** Multi-severity alert notifications with actions

**Features:**
- Multiple severity levels (info/warning/error/critical)
- Dismissible alerts
- Action buttons with callbacks
- Alert grouping and filtering
- Expand/collapse for long lists
- Timestamp display

**Props:**
```typescript
interface AlertBannerProps {
  alerts: AlertItem[];
  onDismiss?: (alertId: string) => void;
  maxVisible?: number;
  groupBySeverity?: boolean;
  showTimestamp?: boolean;
  className?: string;
}

interface AlertItem {
  id: string;
  severity: "info" | "warning" | "error" | "critical";
  title: string;
  message: string;
  timestamp?: string;
  source?: string;
  details?: Record<string, any>;
  actionLabel?: string;
  onAction?: () => void;
  dismissible?: boolean;
}
```

**Helper Components:**
- `AlertList`: Compact list for dashboards
- `InlineAlert`: Single inline alert for immediate feedback

**Usage:**
```tsx
<AlertBanner
  alerts={[
    {
      id: "1",
      severity: "warning",
      title: "High Stress Detected",
      message: "Compressive stress approaching threshold",
      timestamp: new Date().toISOString(),
      source: "stress",
    }
  ]}
  onDismiss={(id) => console.log("Dismissed:", id)}
  maxVisible={5}
/>
```

---

## 📄 Pages

### 1. CVD Overview Dashboard (`/cvd`)

**Location:** `apps/web/src/app/cvd/page.tsx`

**Features:**
- KPI cards: Active runs, avg thickness, avg stress, alerts (24h)
- Average thickness by tool (bar chart)
- Stress distribution (bar chart with color coding)
- 7-day trends (line charts for thickness/stress/adhesion)
- Adhesion by tool (stacked bar visualization)
- Recent alerts panel
- Quick links to recipes, runs, workspace

**Data Refresh:** Every 30 seconds

**Key Metrics:**
- Total active runs across all tools
- Average thickness with deviation from target
- Average stress with safe zone indicator
- Alert count with critical severity breakdown

---

### 2. CVD Recipes Page (`/cvd/recipes`)

**Location:** `apps/web/src/app/cvd/recipes/page.tsx`

**Features:**
- Recipe table with target specifications
- Recipe editor dialog with 3 tabs:
  - **Parameters:** Temperature, pressure, flow rates, mode
  - **Targets:** Target thickness, stress, adhesion class
  - **Predictions:** Live expected windows from physics models
- CRUD operations: Create, edit, delete, duplicate recipes
- Run recipe directly from list
- Filter by material, mode, tool

**Live Predictions:**
- Thickness window: ±5 nm from target
- Stress window: Based on temperature and deposition rate
- Expected adhesion score
- Deposition rate estimation
- WIW uniformity prediction

---

### 3. CVD Runs List (`/cvd/runs`)

**Location:** `apps/web/src/app/cvd/runs/page.tsx`

**Features:**
- Real-time runs list (refreshes every 5s)
- Status badges: Running, Completed, Failed, Pending, Cancelled
- Progress bars for active runs
- Quick metrics: Thickness, stress, adhesion
- Alert count indicators
- Search by run ID, recipe, or tool
- Filter by status
- Link to run detail page

**Columns:**
- Run ID, Recipe, Tool, Status
- Progress (with percentage)
- Thickness (actual vs target with trend icon)
- Stress (MPa with risk indicator)
- Adhesion (color-coded badge)
- Started time with duration
- Alert count

---

### 4. Run Detail Page (`/cvd/runs/[id]`)

**Location:** `apps/web/src/app/cvd/runs/[id]/page.tsx`

**Features:**
- **Metrics Row:** ThicknessGauge, StressBar, Adhesion & Alerts card
- **Telemetry Tab:**
  - Thickness growth (area chart with predicted line)
  - Temperature vs setpoint
  - Pressure vs setpoint
  - Stress evolution
- **Predictions Tab:**
  - Predicted final thickness
  - Predicted final stress
  - Model confidence metrics
- **Alerts Tab:** Full alert history
- **Parameters Tab:** All process parameters

**Real-time Updates:**
- Auto-refresh every 2s if run is active
- Live telemetry plots
- Current metrics in gauges

**Integration Points:**
- Can integrate `RealTimeMonitor` component for WebSocket updates
- Export data functionality
- Link to results deep-dive (when completed)

---

### 5. Results Deep-Dive Page (`/cvd/results/[id]`)

**Location:** `apps/web/src/app/cvd/results/[id]/page.tsx`

**Features:**
- **Summary Stats:** Mean thickness, uniformity, mean stress, adhesion score
- **Wafer Map Tab:**
  - Thickness map (9-point or 49-point)
  - Stress map
  - Measurement points table
- **Distributions Tab:**
  - Thickness histogram with statistics
  - Stress histogram with range
  - Cpk calculation
- **Adhesion Tab:**
  - Test results table (by location)
  - Adhesion score bar chart
  - Test method details (ASTM standards)
- **SPC Tab:**
  - Control chart (last 30 runs)
  - UCL/LCL reference lines
  - Cpk and sigma level metrics
- **VM Analysis Tab:**
  - Predicted vs actual scatter plot
  - Residuals distribution
  - Model performance metrics (RMSE, R², MAE)

**Export:** Report export functionality

---

## 🔌 WebSocket Integration

### useCVDWebSocket Hook

**Location:** `apps/web/src/hooks/useCVDWebSocket.ts`

**Purpose:** Real-time updates via WebSocket connection to backend

**Features:**
- Auto-reconnect with exponential backoff
- Connection state management
- Event type filtering
- Latest event helpers
- Type-safe event handling

**Event Types:**
- `run_started`: Run initiated
- `progress_update`: Deposition progress (%)
- `metrics_update`: Thickness, stress, rate updates
- `warning`: Process warnings
- `error`: Process errors
- `stress_risk`: High stress detected
- `adhesion_risk`: Poor adhesion predicted
- `rate_anomaly`: Deposition rate anomaly
- `run_completed`: Run finished successfully
- `run_failed`: Run failed
- `run_cancelled`: Run cancelled

**Usage:**
```tsx
const { connectionState, events, isConnected } = useCVDWebSocket({
  runId: "CVD_RUN_20251114_103045",
  onEvent: (event) => {
    console.log("Event:", event);
  },
  autoReconnect: true,
  reconnectDelay: 3000,
});

// Filter specific events
const alerts = useCVDEventFilter(events, ["warning", "error", "stress_risk"]);

// Get latest progress
const latestProgress = useLatestCVDEvent(events, "progress_update");
```

---

### RealTimeMonitor Component

**Location:** `apps/web/src/components/cvd/RealTimeMonitor.tsx`

**Purpose:** Display live updates from WebSocket

**Features:**
- Connection status badge
- Live progress bar
- Current thickness, rate, stress, uniformity
- Active alerts panel
- Event count and last update timestamp

**Usage:**
```tsx
<RealTimeMonitor
  runId="CVD_RUN_20251114_103045"
  onProgressUpdate={(progress) => console.log(progress)}
  onThicknessUpdate={(thickness) => console.log(thickness)}
/>
```

**Compact Variant:**
```tsx
<RealTimeIndicator runId="CVD_RUN_20251114_103045" />
```

---

## 🔗 Backend Integration

### API Endpoints Expected

```
GET  /api/cvd/overview                    # Overview dashboard data
GET  /api/cvd/alerts?limit=10             # Recent alerts
GET  /api/cvd/recipes                     # Recipe list
POST /api/cvd/recipes                     # Create recipe
PUT  /api/cvd/recipes/{id}                # Update recipe
DELETE /api/cvd/recipes/{id}              # Delete recipe
GET  /api/cvd/runs?status=running         # Runs list (filterable)
GET  /api/cvd/runs/{id}                   # Run details
GET  /api/cvd/runs/{id}/telemetry         # Telemetry data
GET  /api/cvd/runs/{id}/alerts            # Run alerts
GET  /api/cvd/results/{id}                # Results deep-dive
WS   /ws/cvd/runs/{run_id}                # WebSocket for real-time updates
```

### WebSocket Message Format

**Server → Client:**
```json
{
  "type": "event",
  "event_type": "progress_update",
  "timestamp": "2025-11-14T10:35:00",
  "data": {
    "progress": 67.5,
    "current_thickness_nm": 67.5,
    "deposition_rate_nm_min": 50.2
  }
}
```

**Connection Message:**
```json
{
  "type": "connected",
  "run_id": "CVD_RUN_20251114_103045",
  "message": "Connected to run CVD_RUN_20251114_103045"
}
```

---

## 🎨 Styling

### Technology Stack
- **Framework:** Next.js 14 with App Router
- **Styling:** Tailwind CSS
- **Components:** shadcn/ui
- **Charts:** Recharts
- **Icons:** Lucide React

### Color Scheme

**Thickness:**
- In spec: `#22c55e` (green)
- Out of spec: `#ef4444` (red)

**Stress:**
- Safe zone: `#10b981` (green)
- Warning: `#f59e0b` (orange)
- Critical: `#ef4444` (red)

**Adhesion:**
- Excellent: `#22c55e` (green)
- Good: `#3b82f6` (blue)
- Fair: `#eab308` (yellow)
- Poor: `#ef4444` (red)

**Alerts:**
- Info: `#3b82f6` (blue)
- Warning: `#eab308` (yellow)
- Error: `#f97316` (orange)
- Critical: `#dc2626` (red)

---

## 📊 Data Flow

```
Backend (FastAPI)
    ↓ REST API
Frontend Pages (React Query)
    ↓ Props
Metric Components
    ↓ Display
User

Backend (WebSocket)
    ↓ Real-time Events
useCVDWebSocket Hook
    ↓ Event Stream
RealTimeMonitor Component
    ↓ Display
User
```

---

## 🚀 Getting Started

### Prerequisites
- Backend running at `http://localhost:8001`
- WebSocket server at `ws://localhost:8001`
- Redis for real-time events

### Installation

1. **Install dependencies:**
```bash
cd apps/web
npm install
```

2. **Run development server:**
```bash
npm run dev
```

3. **Access pages:**
- Overview: http://localhost:3000/cvd
- Recipes: http://localhost:3000/cvd/recipes
- Runs: http://localhost:3000/cvd/runs
- Run Detail: http://localhost:3000/cvd/runs/1
- Results: http://localhost:3000/cvd/results/1

### Configuration

**API Base URL:**
Update in each page or create a config file:
```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8001";
```

---

## 📝 Mock Data

All pages include mock data for demonstration. Replace with actual API calls:

**Example:**
```tsx
// Mock data (current)
const mockRuns = [/* ... */];

// Real data (replace with)
const { data: runs } = useQuery({
  queryKey: ["cvd-runs"],
  queryFn: async () => {
    const response = await fetch(`${API_BASE_URL}/api/cvd/runs`);
    return response.json();
  },
});
```

---

## 🧪 Testing Integration

### Test WebSocket Connection

```typescript
// Test hook
const TestComponent = () => {
  const { connectionState, events } = useCVDWebSocket({
    runId: "test-run-123",
    onEvent: (event) => console.log("Event received:", event),
  });

  return <div>Status: {connectionState}</div>;
};
```

### Test Components

```tsx
// Test ThicknessGauge
<ThicknessGauge actual={98.5} target={100} uniformity={1.8} />

// Test StressBar
<StressBar stress={-185} />

// Test AdhesionChip
<AdhesionChip score={88} />

// Test WaferMap
<WaferMap points={mockPoints} parameter="Thickness" unit="nm" />

// Test AlertBanner
<AlertBanner alerts={mockAlerts} />
```

---

## 📈 Performance

**Optimization:**
- React Query caching (30s for overview, 5s for runs list)
- WebSocket auto-reconnect with backoff
- Conditional refetching (only when run is active)
- Lazy loading for charts
- Optimized re-renders with useMemo

**Bundle Size:**
- Components: ~15KB gzipped
- Charts (Recharts): ~45KB gzipped
- shadcn/ui: ~10KB gzipped

---

## 🔧 Troubleshooting

### WebSocket won't connect
1. Check backend WebSocket server is running
2. Verify URL: `ws://localhost:8001/ws/cvd/runs/{runId}`
3. Check browser console for connection errors
4. Ensure run ID exists in backend

### Real-time updates not appearing
1. Check `connectionState` is "connected"
2. Verify events are being received in console
3. Check event type filtering
4. Ensure React Query refetch interval is set

### Charts not rendering
1. Verify Recharts is installed: `npm install recharts`
2. Check data format matches expected structure
3. Ensure ResponsiveContainer has width/height

---

## 📚 Additional Resources

- **Backend Documentation:** `services/analysis/app/JOBQUEUE_REALTIME_README.md`
- **shadcn/ui:** https://ui.shadcn.com/
- **Recharts:** https://recharts.org/
- **React Query:** https://tanstack.com/query/latest

---

## 🎯 Future Enhancements

1. **Real-time Charts:** Streaming telemetry charts
2. **Historical Comparisons:** Compare runs side-by-side
3. **Recipe Optimization:** ML-based recipe suggestions
4. **Export Formats:** PDF, Excel, CSV reports
5. **Mobile Responsiveness:** Optimize for tablets/phones
6. **Dark Mode:** Full dark theme support
7. **Notifications:** Browser notifications for critical alerts
8. **User Preferences:** Save filter/view preferences

---

## ✅ Completion Status

- ✅ All 5 metric components created
- ✅ All 5 pages implemented
- ✅ WebSocket integration complete
- ✅ Real-time monitoring component
- ✅ Mock data for testing
- ✅ TypeScript type safety
- ✅ Responsive design
- ✅ Documentation complete

**Total Files Created:** 12
**Lines of Code:** ~4,000+

---

**Generated:** 2025-11-14

**Authors:** Vladimir Antoine, Claude (Anthropic)
