# SemiconductorLab Platform - Architecture Documentation

**Version:** 1.0  
**Date:** October 21, 2025  
**Authors:** Platform Architecture Team

-----

## Table of Contents

1. [Architecture Overview](#architecture-overview)
1. [C4 Model Diagrams](#c4-model-diagrams)
1. [Technology Stack](#technology-stack)
1. [Component Details](#component-details)
1. [Data Flow](#data-flow)
1. [Security Architecture](#security-architecture)
1. [Deployment Architecture](#deployment-architecture)
1. [Scalability Strategy](#scalability-strategy)
1. [Disaster Recovery](#disaster-recovery)

-----

## Architecture Overview

### Design Principles

1. **Modularity:** Plugin-based instrument drivers; swappable analysis engines
1. **Separation of Concerns:** Frontend ↔ API Gateway ↔ Backend Services ↔ Data Layer
1. **Event-Driven:** Asynchronous processing for long-running analyses
1. **API-First:** OpenAPI spec drives development; versioned endpoints
1. **Observability:** Metrics, logs, traces for all components
1. **Security by Default:** Zero-trust; encrypted at rest and in transit; RBAC everywhere

### Architectural Style

**Hybrid:**

- **Microservices-lite** for core domains (Auth, Instruments, Analysis, Reporting)
- **Monorepo** for development simplicity (Turborepo/NX)
- **Event Sourcing** for audit trail and reproducibility
- **CQRS-lite** for read-heavy analytics queries (materialized views)

### Quality Attributes (Priority Order)

1. **Reliability:** Zero data loss, 99.5% uptime
1. **Accuracy:** Measurements validated against references
1. **Security:** Compliance with 21 CFR Part 11, ISO 17025
1. **Performance:** <1s API response, real-time streaming
1. **Maintainability:** Clear boundaries, comprehensive tests
1. **Extensibility:** New methods in <5 days

-----

## C4 Model Diagrams

### Level 1: System Context Diagram

```mermaid
graph TB
    subgraph "Lab Environment"
        LabTech[Lab Technician<br/>Runs experiments]
        Engineer[Process Engineer<br/>Analyzes data]
        QA[QA Manager<br/>Reviews reports]
        Admin[System Admin<br/>Manages infrastructure]
    end

    subgraph "External Systems"
        AD[Active Directory<br/>LDAP/SSO]
        ELN_Ext[External ELN<br/>LabArchives]
        MES[MES System<br/>Camstar]
        Email[Email Server<br/>SMTP]
    end

    subgraph "Instruments"
        SMU[SMU<br/>Keithley 2400]
        Spec[Spectrometer<br/>Ocean Optics]
        Ellips[Ellipsometer<br/>J.A. Woollam]
        XRD[XRD<br/>Bruker D8]
        OtherInst[... 10+ more]
    end

    SemiconductorLab[SemiconductorLab<br/>Platform<br/><br/>Integrated characterization<br/>data management & analysis]

    LabTech -->|Runs experiments| SemiconductorLab
    Engineer -->|Analyzes data| SemiconductorLab
    QA -->|Reviews reports| SemiconductorLab
    Admin -->|Configures system| SemiconductorLab

    SemiconductorLab -->|Authenticates users| AD
    SemiconductorLab -->|Exports entries| ELN_Ext
    SemiconductorLab -->|Sends lot data| MES
    SemiconductorLab -->|Sends alerts| Email

    SemiconductorLab -->|Controls via VISA| SMU
    SemiconductorLab -->|Acquires spectra| Spec
    SemiconductorLab -->|Measures Ψ/Δ| Ellips
    SemiconductorLab -->|Gets XRD patterns| XRD
    SemiconductorLab -->|Connects to| OtherInst

    style SemiconductorLab fill:#4A90E2,stroke:#2E5C8A,color:#fff
```

**Key Interactions:**

- **Users** interact via web browser (HTTPS)
- **Instruments** communicate via VISA/SCPI, USB, Ethernet
- **External Systems** integrate via OAuth, REST APIs, SMTP

-----

### Level 2: Container Diagram

```mermaid
graph TB
    subgraph "User Devices"
        Browser[Web Browser<br/>Chrome, Firefox, Safari]
    end

    subgraph "SemiconductorLab Platform"
        subgraph "Frontend Layer"
            WebApp[Web Application<br/>Next.js, TypeScript<br/>Port 3000<br/>Interactive UI, real-time plots]
        end

        subgraph "API Layer"
            APIGateway[API Gateway<br/>Kong / Nginx<br/>Port 443<br/>Rate limiting, auth, routing]
        end

        subgraph "Backend Services"
            AuthService[Auth Service<br/>Keycloak / Auth0<br/>OAuth2/OIDC, RBAC]

            InstrumentService[Instrument Service<br/>FastAPI, Python<br/>Driver management, acquisition]

            AnalysisService[Analysis Service<br/>FastAPI, Python<br/>Curve fitting, SPC, ML]

            ReportService[Report Service<br/>FastAPI, Python<br/>PDF generation, exports]

            NotificationService[Notification Service<br/>FastAPI, Python<br/>Email, Slack, webhooks]
        end

        subgraph "Data Layer"
            PostgresDB[(PostgreSQL<br/>+ TimescaleDB<br/>Port 5432<br/>Structured data, time-series)]

            ObjectStore[(Object Storage<br/>MinIO / S3<br/>Port 9000<br/>Raw files, images, reports)]

            RedisCache[(Redis<br/>Port 6379<br/>Session cache, job queue)]
        end

        subgraph "Messaging Layer"
            MessageBus[Message Bus<br/>NATS / Kafka<br/>Port 4222/9092<br/>Event streaming]
        end

        subgraph "Observability"
            Prometheus[Prometheus<br/>Metrics collection]
            Grafana[Grafana<br/>Dashboards]
            Loki[Loki<br/>Log aggregation]
        end

        subgraph "Worker Layer"
            Workers[Analysis Workers<br/>Celery, Python<br/>Background jobs, ML training]
        end
    end

    subgraph "External"
        Instruments[Instruments<br/>VISA/USB/Ethernet]
        AD[Active Directory]
        SMTP[Email Server]
    end

    Browser -->|HTTPS| APIGateway
    APIGateway -->|Routes| WebApp
    APIGateway -->|Routes| AuthService
    APIGateway -->|Routes| InstrumentService
    APIGateway -->|Routes| AnalysisService
    APIGateway -->|Routes| ReportService

    WebApp -->|Auth requests| AuthService
    WebApp -->|API calls| InstrumentService
    WebApp -->|API calls| AnalysisService
    WebApp -->|API calls| ReportService

    AuthService -->|Verifies| AD
    AuthService -.->|Stores sessions| RedisCache

    InstrumentService -->|Reads/writes| PostgresDB
    InstrumentService -->|Stores raw data| ObjectStore
    InstrumentService -->|Publishes events| MessageBus
    InstrumentService -->|Connects to| Instruments

    AnalysisService -->|Reads/writes| PostgresDB
    AnalysisService -->|Reads raw data| ObjectStore
    AnalysisService -->|Subscribes to| MessageBus
    AnalysisService -->|Enqueues jobs| RedisCache

    ReportService -->|Reads| PostgresDB
    ReportService -->|Reads| ObjectStore
    ReportService -->|Stores PDFs| ObjectStore

    NotificationService -->|Subscribes to| MessageBus
    NotificationService -->|Sends emails| SMTP

    Workers -->|Consumes jobs| RedisCache
    Workers -->|Processes data| AnalysisService
    Workers -->|Reads/writes| PostgresDB
    Workers -->|Reads/writes| ObjectStore

    Prometheus -->|Scrapes metrics| InstrumentService
    Prometheus -->|Scrapes metrics| AnalysisService
    Prometheus -->|Scrapes metrics| ReportService
    Prometheus -->|Scrapes metrics| Workers

    Grafana -->|Queries| Prometheus
    Grafana -->|Queries| Loki

    style WebApp fill:#61DAFB,stroke:#20232A,color:#000
    style InstrumentService fill:#4CAF50,stroke:#2E7D32,color:#fff
    style AnalysisService fill:#4CAF50,stroke:#2E7D32,color:#fff
    style ReportService fill:#4CAF50,stroke:#2E7D32,color:#fff
    style PostgresDB fill:#336791,stroke:#1A3A52,color:#fff
    style MessageBus fill:#FF6B6B,stroke:#C92A2A,color:#fff
```

**Container Descriptions:**

|Container                   |Technology                |Responsibility                                                    |
|----------------------------|--------------------------|------------------------------------------------------------------|
|**Web Application**         |Next.js, React, TypeScript|User interface; server-side rendering; responsive design          |
|**API Gateway**             |Kong or Nginx             |Rate limiting, TLS termination, request routing, auth verification|
|**Auth Service**            |Keycloak or Auth0         |User authentication, token issuance, RBAC                         |
|**Instrument Service**      |FastAPI, Python           |Driver management, instrument communication, data acquisition     |
|**Analysis Service**        |FastAPI, Python           |Algorithm execution, SPC, ML inference, curve fitting             |
|**Report Service**          |FastAPI, Python           |PDF generation, export formats, templating                        |
|**Notification Service**    |FastAPI, Python           |Alert dispatching (email, Slack, webhooks)                        |
|**PostgreSQL + TimescaleDB**|SQL database              |Structured data, time-series optimization, ACID transactions      |
|**Object Storage**          |MinIO or S3               |Large binary objects (raw data, images, PDFs)                     |
|**Redis**                   |In-memory cache           |Session storage, job queue (Celery broker)                        |
|**Message Bus**             |NATS or Kafka             |Event streaming, pub/sub, decoupling services                     |
|**Analysis Workers**        |Celery, Python            |Background processing, long-running analyses, ML training         |
|**Prometheus**              |Metrics DB                |Time-series metrics collection                                    |
|**Grafana**                 |Visualization             |Dashboards for metrics and logs                                   |
|**Loki**                    |Log aggregation           |Centralized logging                                               |

-----

### Level 3: Component Diagram (Instrument Service)

```mermaid
graph TB
    subgraph "Instrument Service"
        API[API Controller<br/>FastAPI routes]
        
        subgraph "Core Logic"
            InstrMgr[Instrument Manager<br/>Registry, lifecycle]
            DriverMgr[Driver Manager<br/>Plugin loader, versioning]
            ConnMgr[Connection Manager<br/>VISA, serial, TCP/IP]
            CalibMgr[Calibration Manager<br/>Certificate tracking]
        end

        subgraph "Drivers"
            SMUDriver[SMU Driver<br/>Keithley 2400/2600]
            SpecDriver[Spectrometer Driver<br/>Ocean Optics]
            EllipsDriver[Ellipsometer Driver<br/>J.A. Woollam]
            GenericDriver[Generic SCPI Driver<br/>Fallback]
        end

        subgraph "HIL Simulators"
            SMUSim[SMU Simulator<br/>I-V models]
            SpecSim[Spectrometer Simulator<br/>Gaussian peaks]
            EllipsSim[Ellipsometer Simulator<br/>Fresnel layers]
        end

        subgraph "Data Handlers"
            DataValidator[Data Validator<br/>Range checks, unit validation]
            DataWriter[Data Writer<br/>HDF5, CSV, JSONL]
            EventPublisher[Event Publisher<br/>Publishes to NATS/Kafka]
        end
    end

    subgraph "External"
        DB[(PostgreSQL)]
        Storage[(Object Storage)]
        MsgBus[Message Bus]
        PhysicalInst[Physical Instruments]
    end

    API -->|Registers| InstrMgr
    API -->|Starts run| InstrMgr

    InstrMgr -->|Loads plugin| DriverMgr
    InstrMgr -->|Opens connection| ConnMgr
    InstrMgr -->|Checks calibration| CalibMgr

    DriverMgr -.->|Instantiates| SMUDriver
    DriverMgr -.->|Instantiates| SpecDriver
    DriverMgr -.->|Instantiates| EllipsDriver
    DriverMgr -.->|Fallback| GenericDriver

    ConnMgr -->|VISA commands| PhysicalInst
    ConnMgr -.->|Dev mode| SMUSim
    ConnMgr -.->|Dev mode| SpecSim
    ConnMgr -.->|Dev mode| EllipsSim

    SMUDriver -->|Acquires data| DataValidator
    SpecDriver -->|Acquires data| DataValidator
    EllipsDriver -->|Acquires data| DataValidator

    DataValidator -->|Writes| DataWriter
    DataWriter -->|Stores metadata| DB
    DataWriter -->|Stores raw files| Storage

    DataValidator -->|Triggers| EventPublisher
    EventPublisher -->|Publishes run.completed| MsgBus

    style InstrMgr fill:#FFD93D,stroke:#F39C12,color:#000
    style DriverMgr fill:#FFD93D,stroke:#F39C12,color:#000
    style DataValidator fill:#A8E6CF,stroke:#56C596,color:#000
```

**Component Responsibilities:**

- **API Controller:** REST endpoints (POST /instruments, GET /runs, etc.)
- **Instrument Manager:** Tracks registered instruments, orchestrates acquisition
- **Driver Manager:** Dynamic plugin loading, version compatibility checks
- **Connection Manager:** VISA/serial/TCP abstraction, error handling, timeouts
- **Calibration Manager:** Tracks cal certificates, alerts on expiry
- **Drivers:** Vendor-specific command sequences, response parsing
- **HIL Simulators:** Physics-based models for dev/test without hardware
- **Data Validator:** Range checks, unit validation, outlier flagging
- **Data Writer:** Serialization to HDF5/CSV, metadata generation
- **Event Publisher:** Publishes events to message bus (run started, completed, failed)

-----

## Technology Stack

### Frontend

|Component            |Technology             |Rationale                                              |
|---------------------|-----------------------|-------------------------------------------------------|
|**Framework**        |Next.js 14 (App Router)|SSR/SSG, React Server Components, built-in optimization|
|**Language**         |TypeScript 5.0+        |Type safety, IntelliSense, maintainability             |
|**Styling**          |Tailwind CSS 3.x       |Utility-first, rapid prototyping, consistent design    |
|**Component Library**|shadcn/ui              |High-quality, accessible, customizable components      |
|**Charts**           |Plotly.js, Recharts    |Interactive plots, export to PNG/SVG, scientific viz   |
|**State Management** |Zustand, TanStack Query|Simple state, server state caching                     |
|**Forms**            |React Hook Form + Zod  |Validation, type-safe schemas                          |

### Backend

|Component          |Technology        |Rationale                                             |
|-------------------|------------------|------------------------------------------------------|
|**API Framework**  |FastAPI 0.110+    |High performance, auto OpenAPI docs, async support    |
|**Language**       |Python 3.11+      |Rich scientific ecosystem (NumPy, SciPy, scikit-learn)|
|**ORM**            |SQLAlchemy 2.0    |Mature, supports complex queries, migrations (Alembic)|
|**Validation**     |Pydantic v2       |Type-safe data validation, serialization              |
|**Background Jobs**|Celery 5.x + Redis|Distributed task queue, retries, priorities           |
|**Testing**        |pytest, httpx     |Unit, integration, E2E tests                          |

### Data Layer

|Component         |Technology           |Rationale                                                |
|------------------|---------------------|---------------------------------------------------------|
|**Database**      |PostgreSQL 15+       |ACID, JSON support, mature, high performance             |
|**Time-Series**   |TimescaleDB 2.x      |Hypertables, continuous aggregates, compression          |
|**Object Storage**|MinIO (on-prem) or S3|Scalable, S3-compatible API, versioning                  |
|**Cache**         |Redis 7.x            |In-memory, low-latency, pub/sub                          |
|**Message Bus**   |NATS 2.x or Kafka 3.x|Event streaming; NATS for <10 instruments, Kafka for >100|

### Data Science / ML

|Component               |Technology            |Rationale                                  |
|------------------------|----------------------|-------------------------------------------|
|**Scientific Computing**|NumPy, SciPy, Pandas  |Array operations, curve fitting, statistics|
|**ML Framework**        |scikit-learn, LightGBM|Wide algorithm coverage, production-ready  |
|**DL Framework**        |PyTorch (optional)    |Autoencoders, LSTMs for time-series        |
|**Model Export**        |ONNX                  |Interoperability, fast inference           |
|**Experiment Tracking** |MLflow (optional)     |Model registry, versioning, A/B tests      |

### Instrument Integration

|Component |Technology       |Rationale                                     |
|----------|-----------------|----------------------------------------------|
|**VISA**  |PyVISA 1.13+     |Industry standard for instrument communication|
|**SCPI**  |Custom parser    |Text-based commands for most lab equipment    |
|**Serial**|pySerial         |RS-232/USB-to-serial devices                  |
|**USB**   |libusb via ctypes|Direct USB communication (e.g., Ocean Optics) |

### DevOps / Infrastructure

|Component           |Technology                       |Rationale                                                |
|--------------------|---------------------------------|---------------------------------------------------------|
|**Containerization**|Docker 24+, Docker Compose       |Reproducible builds, local dev parity                    |
|**Orchestration**   |Kubernetes 1.28+ (K3s or EKS/GKE)|Horizontal scaling, self-healing, rolling updates        |
|**CI/CD**           |GitHub Actions                   |Integrated with repo, good free tier, self-hosted runners|
|**IaC**             |Helm 3.x, Terraform (optional)   |Declarative deployments, version control                 |
|**Secrets**         |SOPS or HashiCorp Vault          |Encrypted secrets in repo or external vault              |
|**Monitoring**      |Prometheus, Grafana, Loki        |Open-source observability stack                          |
|**Tracing**         |OpenTelemetry (optional)         |Distributed tracing for debugging                        |

-----

## Component Details

### Frontend Architecture

**Directory Structure:**

```
apps/web/
├── app/                    # Next.js app router
│   ├── (auth)/            # Auth pages (login, register)
│   ├── dashboard/         # Main dashboard
│   ├── experiments/       # Experiment builder
│   ├── results/           # Results explorer
│   ├── spc/              # SPC hub
│   ├── vm/               # Virtual metrology studio
│   └── admin/            # Admin panel
├── components/           # Shared components
│   ├── ui/              # shadcn/ui components
│   ├── charts/          # Chart wrappers (Plotly, Recharts)
│   ├── forms/           # Form components
│   └── layouts/         # Page layouts
├── lib/                 # Utilities
│   ├── api.ts          # API client (fetch wrappers)
│   ├── auth.ts         # Auth helpers
│   └── utils.ts        # Common utilities
├── hooks/              # Custom React hooks
├── stores/             # Zustand stores
└── types/              # TypeScript types
```

**Key Patterns:**

- **Server Components:** Default; use Client Components only when needed (interactivity)
- **Data Fetching:** TanStack Query for client-side; native fetch in Server Components
- **Optimistic Updates:** For real-time feel (e.g., updating run status before server confirms)
- **Error Boundaries:** Catch rendering errors, show fallback UI
- **Suspense:** Loading states for async components

-----

### Backend Architecture

**Directory Structure:**

```
services/
├── auth/              # Authentication service
│   ├── main.py       # FastAPI app
│   ├── routers/      # API endpoints
│   ├── models/       # SQLAlchemy models
│   ├── schemas/      # Pydantic schemas
│   └── services/     # Business logic
├── instruments/      # Instrument service
│   ├── main.py
│   ├── routers/
│   ├── drivers/      # Instrument drivers
│   │   ├── base.py  # Abstract base class
│   │   ├── keithley.py
│   │   ├── oceanoptics.py
│   │   └── ...
│   ├── simulators/   # HIL simulators
│   └── plugins/      # User-added drivers (loaded at runtime)
├── analysis/         # Analysis service
│   ├── main.py
│   ├── methods/      # Analysis pipelines per method
│   │   ├── electrical/
│   │   ├── optical/
│   │   ├── structural/
│   │   └── chemical/
│   ├── spc/         # SPC algorithms
│   ├── ml/          # ML models
│   └── utils/       # Curve fitting, statistics
├── reporting/       # Report service
│   ├── main.py
│   ├── templates/   # Jinja2 templates
│   └── generators/  # PDF, CSV, MATLAB export
└── notifications/   # Notification service
    ├── main.py
    └── channels/    # Email, Slack, webhook handlers
```

**Common Patterns:**

- **Dependency Injection:** FastAPI’s DI for database sessions, services
- **Repository Pattern:** Separate data access logic from business logic
- **Service Layer:** Thin controllers, fat services
- **Event Handlers:** Subscribers listen to message bus, trigger actions
- **Background Jobs:** Celery tasks for long-running operations (ML training, batch reports)

-----

## Data Flow

### Example: Running an I-V Sweep

```mermaid
sequenceDiagram
    actor User
    participant WebApp
    participant API Gateway
    participant Instrument Service
    participant Driver
    participant Physical Instrument
    participant Message Bus
    participant Analysis Service
    participant PostgreSQL
    participant Object Storage
    participant WebApp as WebApp (SSE)

    User->>WebApp: Click "Start I-V Sweep"
    WebApp->>API Gateway: POST /runs {method: 'iv', params: {...}}
    API Gateway->>Instrument Service: Forward request (+ auth token)
    
    Instrument Service->>PostgreSQL: Create run record (status: 'pending')
    Instrument Service->>Driver: Initialize with params
    Driver->>Physical Instrument: Send SCPI commands (setup)
    Physical Instrument-->>Driver: ACK
    
    Instrument Service->>Message Bus: Publish 'run.started'
    Instrument Service-->>API Gateway: 202 Accepted {run_id: '123'}
    API Gateway-->>WebApp: Return run_id
    WebApp->>WebApp (SSE): Subscribe to /runs/123/stream

    loop Sweep points
        Driver->>Physical Instrument: Set voltage, measure current
        Physical Instrument-->>Driver: Current value
        Driver->>Instrument Service: Append data point
        Instrument Service->>Message Bus: Publish 'run.data' {run_id, V, I}
        Message Bus-->>WebApp (SSE): Stream data point
        WebApp (SSE)->>User: Update live plot
    end

    Driver->>Instrument Service: Sweep complete
    Instrument Service->>Object Storage: Save raw data (HDF5)
    Instrument Service->>PostgreSQL: Update run (status: 'completed', file_uri)
    Instrument Service->>Message Bus: Publish 'run.completed'

    Message Bus->>Analysis Service: Trigger analysis
    Analysis Service->>Object Storage: Load raw data
    Analysis Service->>Analysis Service: Fit diode model (extract Is, n)
    Analysis Service->>PostgreSQL: Save analysis results
    Analysis Service->>Message Bus: Publish 'analysis.completed'

    Message Bus-->>WebApp (SSE): Notify analysis done
    WebApp (SSE)->>User: Show extracted parameters
```

**Key Points:**

- **Asynchronous:** Run creation returns immediately; actual acquisition happens in background
- **Real-time Updates:** SSE (Server-Sent Events) for live data streaming to UI
- **Event-Driven:** Completion of run triggers analysis automatically
- **Decoupled:** Instrument Service doesn’t know about Analysis Service; message bus mediates

-----

## Security Architecture

### Zero-Trust Model

- **No Implicit Trust:** All requests authenticated/authorized, even internal
- **Least Privilege:** Users/services granted minimum required permissions
- **Defense in Depth:** Multiple layers (network, application, data)

### Authentication Flow (OAuth2/OIDC)

```mermaid
sequenceDiagram
    actor User
    participant Browser
    participant WebApp
    participant Auth Service (Keycloak)
    participant API Gateway
    participant Backend Services

    User->>Browser: Navigate to /dashboard
    Browser->>WebApp: GET /dashboard
    WebApp->>Auth Service (Keycloak): Redirect to /auth/login
    Auth Service (Keycloak)->>User: Show login form
    User->>Auth Service (Keycloak): Submit credentials
    Auth Service (Keycloak)->>Auth Service (Keycloak): Verify against AD/LDAP
    Auth Service (Keycloak)->>Browser: Redirect to /callback?code=...
    Browser->>WebApp: GET /callback?code=...
    WebApp->>Auth Service (Keycloak): Exchange code for tokens
    Auth Service (Keycloak)-->>WebApp: Access token + ID token + Refresh token
    WebApp->>Browser: Set cookie (encrypted session)
    Browser->>API Gateway: GET /api/runs (+ cookie)
    API Gateway->>Auth Service (Keycloak): Validate token (introspection)
    Auth Service (Keycloak)-->>API Gateway: Token valid, user roles
    API Gateway->>Backend Services: Forward request (+ user context)
    Backend Services-->>API Gateway: Response
    API Gateway-->>Browser: Response
```

**Key Security Features:**

- **Short-lived Access Tokens:** 15 min expiry; refresh tokens for renewal
- **PKCE (Proof Key for Code Exchange):** Prevents authorization code interception
- **Secure Cookies:** HttpOnly, SameSite=Strict, Secure flag
- **Token Introspection:** API Gateway validates every request
- **Logout:** Token revocation + session cleanup

### RBAC (Role-Based Access Control)

|Role                           |Permissions                                               |Typical User                    |
|-------------------------------|----------------------------------------------------------|--------------------------------|
|**Admin**                      |Full system access; user management; instrument config    |IT admin, platform owner        |
|**PI (Principal Investigator)**|Project management; approve results; view all project data|Lab manager, senior scientist   |
|**Engineer**                   |Create/run experiments; analyze data; generate reports    |Process engineers, researchers  |
|**Technician**                 |Run pre-configured experiments; view results              |Lab technicians                 |
|**Viewer**                     |Read-only access to public data                           |External collaborators, auditors|

**Permission Model:**

- **Resource-Based:** Permissions tied to Projects, Samples, Runs
- **Hierarchical:** PI inherits all permissions of Engineer; Engineer inherits Technician
- **Explicit Denies:** Can revoke specific permissions (e.g., Technician cannot delete runs)

### Data Encryption

**At Rest:**

- **Database:** PostgreSQL TDE (Transparent Data Encryption) or pgcrypto
- **Object Storage:** Server-side encryption (SSE-S3 or SSE-KMS)
- **Backups:** Encrypted before offsite transfer (GPG or Restic)

**In Transit:**

- **TLS 1.3:** All HTTP traffic (frontend ↔ backend, backend ↔ database)
- **mTLS (optional):** Service-to-service communication in K8s
- **VISA over TCP:** Encrypted if supported by instrument (rare; document risk)

**Secrets Management:**

- **Development:** SOPS (encrypted files in repo, decrypted at runtime)
- **Production:** HashiCorp Vault (dynamic secrets, rotation, audit log)
- **Never in Code:** No hardcoded passwords/API keys; CI fails on secret scan

-----

## Deployment Architecture

### Development Environment

```yaml
# docker-compose.yml (simplified)
services:
  web:
    image: semiconductorlab/web:dev
    ports: ["3000:3000"]
    environment:
      - API_URL=http://api-gateway:8000
  
  api-gateway:
    image: kong:3.4
    ports: ["8000:8000"]
  
  auth:
    image: keycloak:22
    ports: ["8080:8080"]
  
  instruments:
    image: semiconductorlab/instruments:dev
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/lab
      - NATS_URL=nats://nats:4222
  
  analysis:
    image: semiconductorlab/analysis:dev
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/lab
  
  reporting:
    image: semiconductorlab/reporting:dev
  
  postgres:
    image: timescale/timescaledb:latest-pg15
    ports: ["5432:5432"]
    volumes: ["postgres-data:/var/lib/postgresql/data"]
  
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
  
  nats:
    image: nats:2.10
    ports: ["4222:4222"]
  
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports: ["9000:9000", "9001:9001"]
    volumes: ["minio-data:/data"]
  
  prometheus:
    image: prom/prometheus:latest
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
    ports: ["9090:9090"]
  
  grafana:
    image: grafana/grafana:latest
    ports: ["3001:3000"]
```

**One-Command Startup:**

```bash
make dev-up   # Starts all services
make dev-down # Stops and removes containers
make dev-logs # Tail logs
```

-----

### Production Environment (Kubernetes)

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "Ingress"
            IngressCtrl[Ingress Controller<br/>Nginx/Traefik<br/>TLS termination]
        end

        subgraph "Frontend Namespace"
            WebPods[Web App Pods<br/>x3 replicas<br/>HPA: 3-10]
        end

        subgraph "Backend Namespace"
            AuthPods[Auth Pods<br/>x2 replicas]
            InstrPods[Instrument Pods<br/>x3 replicas<br/>HPA: 3-10]
            AnalysisPods[Analysis Pods<br/>x3 replicas<br/>HPA: 3-20]
            ReportPods[Report Pods<br/>x2 replicas]
            NotifyPods[Notify Pods<br/>x2 replicas]
            WorkerPods[Worker Pods<br/>x5 replicas<br/>HPA: 5-50]
        end

        subgraph "Data Namespace"
            PostgresStateful[PostgreSQL<br/>StatefulSet x3<br/>Patroni HA]
            RedisStateful[Redis<br/>StatefulSet x3<br/>Sentinel]
            NATSStateful[NATS<br/>StatefulSet x3<br/>JetStream]
        end

        subgraph "Storage"
            PVCPostgres[PVC: postgres-data]
            PVCRedis[PVC: redis-data]
            PVCMinio[PVC: minio-data]
        end

        subgraph "Monitoring Namespace"
            PromPods[Prometheus Pods x2]
            GrafanaPods[Grafana Pods x2]
            LokiPods[Loki Pods x2]
        end
    end

    subgraph "External"
        Users[Users]
        Instruments[Instruments<br/>Lab Network]
        S3External[External S3<br/>or MinIO cluster]
    end

    Users -->|HTTPS| IngressCtrl
    IngressCtrl -->|Routes /| WebPods
    IngressCtrl -->|Routes /api| AuthPods
    IngressCtrl -->|Routes /api| InstrPods
    IngressCtrl -->|Routes /api| AnalysisPods
    IngressCtrl -->|Routes /api| ReportPods

    WebPods -->|Calls| AuthPods
    WebPods -->|Calls| InstrPods
    WebPods -->|Calls| AnalysisPods

    InstrPods -->|Connects to| Instruments
    InstrPods -->|Reads/writes| PostgresStateful
    InstrPods -->|Publishes| NATSStateful
    InstrPods -->|Stores files| S3External

    AnalysisPods -->|Reads/writes| PostgresStateful
    AnalysisPods -->|Subscribes| NATSStateful
    AnalysisPods -->|Enqueues| RedisStateful

    WorkerPods -->|Consumes| RedisStateful
    WorkerPods -->|Processes| AnalysisPods

    PostgresStateful -.->|Mounts| PVCPostgres
    RedisStateful -.->|Mounts| PVCRedis

    PromPods -->|Scrapes| InstrPods
    PromPods -->|Scrapes| AnalysisPods
    PromPods -->|Scrapes| WorkerPods
    GrafanaPods -->|Queries| PromPods
    GrafanaPods -->|Queries| LokiPods

    style IngressCtrl fill:#FF6B6B,stroke:#C92A2A,color:#fff
    style PostgresStateful fill:#336791,stroke:#1A3A52,color:#fff
    style NATSStateful fill:#27AAE1,stroke:#1A6FA1,color:#fff
```

**Key Features:**

- **High Availability:** Multi-replica for all stateless services; StatefulSet for databases
- **Auto-Scaling:** HPA based on CPU/memory or custom metrics (queue depth)
- **Rolling Updates:** Zero-downtime deployments
- **Affinity Rules:** Spread replicas across nodes for fault tolerance
- **Resource Limits:** Prevent noisy neighbors; guaranteed QoS
- **Persistent Volumes:** Data survives pod restarts
- **Secrets:** Kubernetes Secrets or external Vault integration

**Helm Chart Structure:**

```
helm/semiconductorlab/
├── Chart.yaml
├── values.yaml              # Default values
├── values-prod.yaml         # Production overrides
├── templates/
│   ├── web-deployment.yaml
│   ├── instruments-deployment.yaml
│   ├── analysis-deployment.yaml
│   ├── postgres-statefulset.yaml
│   ├── redis-statefulset.yaml
│   ├── nats-statefulset.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml            # Horizontal Pod Autoscalers
│   ├── pvc.yaml            # Persistent Volume Claims
│   └── ...
└── charts/                 # Sub-charts (optional)
```

**Deployment Commands:**

```bash
# Install or upgrade
helm upgrade --install semiconductorlab ./helm/semiconductorlab \
  --namespace lab --create-namespace \
  --values ./helm/semiconductorlab/values-prod.yaml

# Rollback
helm rollback semiconductorlab 1 --namespace lab

# View status
helm status semiconductorlab --namespace lab
kubectl get pods -n lab
```

-----

## Scalability Strategy

### Horizontal Scaling Targets

|Component         |Baseline Replicas|Max Replicas|Scaling Metric                      |
|------------------|-----------------|------------|------------------------------------|
|Web App           |3                |10          |CPU > 70% or request latency > 500ms|
|Instrument Service|3                |10          |Active connections or queue depth   |
|Analysis Service  |3                |20          |CPU > 80% or job queue length       |
|Workers           |5                |50          |Celery queue depth                  |
|PostgreSQL        |3 (HA)           |3 (no HPA)  |Manual scale + read replicas        |
|Redis             |3 (Sentinel)     |3 (no HPA)  |Manual scale                        |
|NATS              |3 (cluster)      |5           |Manual scale                        |

### Vertical Scaling

- **Database:** Scale up to 64 cores / 256 GB RAM before sharding
- **Workers:** GPU-enabled nodes for ML training (NVIDIA T4/A100)
- **Storage:** Start 10 TB, expand to 500 TB+ with object storage tiers

### Caching Strategy

1. **Redis Cache:** Session data (TTL: 1 hour), frequently accessed runs (TTL: 5 min)
1. **CDN (optional):** Static assets, reports (for external users)
1. **Materialized Views:** Pre-aggregated SPC data, updated every 5 minutes
1. **Query Caching:** PostgreSQL query cache for repeated analytics queries

### Database Optimization

- **Partitioning:** Time-series tables partitioned by month
- **Indexes:** B-tree on foreign keys, GiST on spatial data, GIN on JSONB
- **Connection Pooling:** PgBouncer (max 100 connections per replica)
- **Read Replicas:** 2 read-only replicas for analytics queries

-----

## Disaster Recovery

### Backup Strategy

|Component     |Frequency                  |Retention                      |Method                       |
|--------------|---------------------------|-------------------------------|-----------------------------|
|PostgreSQL    |Continuous WAL + daily full|30 days rolling + 1 year annual|pgBackRest or Barman         |
|Object Storage|Daily incremental          |30 days rolling                |MinIO mirror or S3 versioning|
|Configurations|On change (Git)            |Forever                        |GitOps (Helm values in repo) |
|Secrets       |Manual (encrypted)         |Forever                        |SOPS or Vault snapshot       |

### Recovery Objectives

- **RTO (Recovery Time Objective):** 1 hour (time to restore service)
- **RPO (Recovery Point Objective):** 5 minutes (max data loss)

### DR Plan

1. **Primary Failure:** Automatic failover to standby PostgreSQL (Patroni); redirect traffic to backup region
1. **Data Corruption:** Restore from last known-good backup (point-in-time recovery)
1. **Region Failure:** Failover to secondary datacenter (manual if cross-region; requires geo-replication)
1. **Ransomware:** Restore from immutable backups (S3 Object Lock or tape)

### Testing

- **Monthly:** Restore test (random backup → test environment)
- **Quarterly:** Full DR drill (simulate primary datacenter failure)
- **Annual:** Tabletop exercise with stakeholders

-----

## Appendix: Design Decisions & Rationale

### Why FastAPI over Flask/Django?

- **Performance:** ASGI async support; handles concurrent requests better
- **Auto Docs:** OpenAPI spec generated automatically
- **Type Safety:** Pydantic integration; fewer runtime errors
- **Modern:** Active development, growing ecosystem

### Why NATS over Kafka?

- **Simplicity:** Easier to operate for <10 instrument scale
- **Low Latency:** Sub-millisecond message delivery
- **Built-in Persistence:** JetStream for event sourcing
- **Caveat:** Switch to Kafka if scale >100 instruments or multi-datacenter required

### Why Next.js over Create React App?

- **SSR/SSG:** Better initial load time, SEO (not critical for internal tool, but nice)
- **App Router:** React Server Components reduce bundle size
- **Optimizations:** Automatic code splitting, image optimization, font optimization
- **Ecosystem:** Vercel deployment (if cloud), strong community

### Why PostgreSQL over MongoDB?

- **ACID:** Critical for audit trail and compliance
- **Relational Model:** Many relationships (samples → runs → results)
- **JSON Support:** JSONB for flexible metadata (best of both worlds)
- **Maturity:** Battle-tested, excellent tooling (pg_dump, pgAdmin)

### Why Monorepo (Turborepo/NX)?

- **Shared Code:** Common types, utilities across frontend/backend
- **Atomic Changes:** Single PR can update API + UI
- **Build Caching:** Turborepo caches build artifacts; only rebuild changed packages
- **Caveat:** Can become unwieldy if >50 packages; monitor repo size

-----

## Next Steps

1. **Review & Approval:** Architecture review meeting with stakeholders
1. **Spike/PoC:** Build thin slice (single I-V sweep end-to-end) to validate architecture
1. **Repository Setup:** Create monorepo, CI/CD pipelines, Docker Compose
1. **Database Schema:** Implement core entities, migrations
1. **Driver SDK:** Build abstract base, 1-2 reference drivers, HIL simulator
1. **UI Shell:** Navigation, auth pages, experiment builder skeleton

-----

*End of Architecture Documentation*