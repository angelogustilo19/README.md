# Momentum AI

A full-stack conversational AI assistant with specialized financial calculation capabilities, deployed on Google Cloud Platform with Kubernetes.

![React](https://img.shields.io/badge/React-18.3-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Python-green) ![MySQL](https://img.shields.io/badge/MySQL-8+-orange) ![Kubernetes](https://img.shields.io/badge/Kubernetes-K3s-purple) ![GCP](https://img.shields.io/badge/GCP-Cloud-red)

## Overview

Momentum AI is a sophisticated conversational AI application that combines natural language processing with precise financial calculations. It provides users with an intelligent assistant capable of answering general knowledge questions and performing financial computations (debt payoff, loan amortization, monthly payments) through a hybrid processing approach.

## Key Features

### Intelligent Hybrid Processing
- **Intent Detection**: Automatically identifies whether queries are financial calculations or general knowledge questions
- **Financial Path**: Extracts parameters (principal, interest rate, monthly payment) and performs precise calculations
- **General Knowledge Path**: Routes to LLM for factual answers with RAG-enhanced context

### Financial Calculation Engine
- Debt payoff time calculation (years + months)
- Monthly payment calculation for fixed timeframes
- Repayment plan generation with total interest calculation
- Support for loans, credit cards, and mortgages

### User Experience
- User authentication with secure password hashing
- Per-user chat history persistence
- Command palette with keyboard shortcuts (Cmd/Ctrl+K)
- Real-time message formatting and validation
- Responsive Material-UI design

### Enterprise-Grade Infrastructure
- Multi-tier LLM fallback system (Gemini → OpenAI → Ollama)
- Rate limiting to prevent API abuse
- Connection pooling for database efficiency
- Kubernetes auto-healing and rolling updates

## Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| FastAPI | Web framework (Python 3.11+) |
| MySQL 8+ | Database with connection pooling |
| Google Gemini 2.5-Flash | Primary LLM |
| OpenAI GPT-3.5-Turbo | Fallback LLM |
| Ollama | Local LLM fallback |
| FAISS + LangChain | Vector search & RAG |
| bcrypt | Password hashing |
| SlowAPI | Rate limiting |

### Frontend
| Technology | Purpose |
|------------|---------|
| React 18.3.1 | UI framework |
| Material-UI (MUI) v7 | Component library |
| Axios | HTTP client |
| Tailwind CSS + Emotion | Styling |

### Infrastructure & DevOps
| Technology | Purpose |
|------------|---------|
| Docker | Containerization (multi-stage builds) |
| Kubernetes (K3s) | Container orchestration |
| GCP Compute Engine | K3s VM hosting |
| GCP Cloud SQL | Managed MySQL |
| GCP Artifact Registry | Container registry |
| GitHub Actions | CI/CD pipeline |
| Traefik | Ingress controller |
| Nginx | Frontend serving |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client (Browser)                         │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Traefik Ingress Controller                    │
│                       (SSL/TLS Termination)                      │
└─────────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
┌───────────────────────────┐     ┌───────────────────────────────┐
│   Frontend (React/Nginx)  │     │      Backend (FastAPI)        │
│   ────────────────────    │     │   ─────────────────────────   │
│   • Material-UI           │     │   • Intent Detection          │
│   • Command Palette       │     │   • Financial Calculator      │
│   • Chat Interface        │     │   • RAG Retrieval (FAISS)     │
│   • Auth Forms            │     │   • LLM Integration           │
└───────────────────────────┘     │   • Rate Limiting             │
                                  └───────────────────────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
                    ▼                           ▼                           ▼
        ┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
        │    Cloud SQL      │       │   Gemini API      │       │   FAISS Index     │
        │    (MySQL 8+)     │       │   (Primary LLM)   │       │   (Vector Store)  │
        └───────────────────┘       └───────────────────┘       └───────────────────┘
```

## Project Structure

```
Project-Momentum/
│
├── rag_app/                          # Backend application
│   ├── service.py                    # FastAPI endpoints & main app
│   ├── database.py                   # MySQL connection pool & queries
│   ├── debt_calculator.py            # Financial calculation engine
│   ├── prompts.py                    # LLM prompt templates
│   ├── retriever.py                  # FAISS retrieval chain
│   ├── agents.py                     # Agent definitions
│   ├── utils.py                      # Utility functions
│   └── domain/                       # Intent detection modules
│       ├── __init__.py
│       ├── banking.py                # Finance topic detection
│       ├── cloud.py                  # Cloud computing topics
│       └── general_query.py          # General knowledge routing
│
├── frontend/
│   ├── src/
│   │   ├── App.js                    # Main React component
│   │   ├── theme.js                  # MUI theme configuration
│   │   ├── index.js                  # Application entry point
│   │   ├── components/
│   │   │   ├── Chatbot.js            # Core chat interface
│   │   │   ├── CommandPalette.js     # Keyboard shortcuts (Cmd+K)
│   │   │   ├── EmptyState.js         # Empty chat state
│   │   │   ├── FormattedBotMessage.js # Message formatting
│   │   │   ├── Login.js              # Authentication
│   │   │   ├── Register.js           # User registration
│   │   │   └── chat.css              # Chat styling
│   │   └── hooks/
│   │       └── useHotkeys.js         # Keyboard shortcut hook
│   ├── public/
│   │   └── index.html                # HTML template
│   ├── nginx.conf                    # Production server config
│   ├── tailwind.config.js            # Tailwind CSS config
│   └── package.json                  # Dependencies
│
├── deploy/
│   └── k8s/                          # Kubernetes manifests
│       ├── namespace.yaml            # momentum namespace
│       ├── configmap.yaml            # Non-secret configuration
│       ├── secrets.example.yaml      # Secret template
│       ├── backend.yaml              # Backend Deployment + Service
│       ├── frontend.yaml             # Frontend Deployment + Service
│       ├── mysql.yaml                # Cloud SQL proxy
│       ├── ingress.yaml              # Traefik ingress rules
│       ├── kustomization.yaml        # Kustomize config
│       └── RUNBOOK.md                # Deployment guide
│
├── .github/
│   └── workflows/
│       └── deploy.yml                # GitHub Actions CI/CD
│
├── faiss_index/                      # Pre-built FAISS vector store
├── prompts/                          # System prompt overrides
├── scripts/
│   └── open-browser.js               # Dev browser auto-open
│
├── Dockerfile.backend                # Backend container image
├── Dockerfile.frontend               # Frontend container image
├── docker-compose.yml                # Local development setup
├── requirements.txt                  # Python dependencies
├── package.json                      # Root npm scripts
├── .env.template                     # Environment template
└── README.md                         # Documentation
```

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

**Build Stage**
- Multi-platform Docker builds using buildx
- Images pushed to Docker Hub with SHA tagging
- Build cache optimization

**Deploy Stage**
- Authenticate to GCP via service account
- SSH into K3s cluster
- Rolling restart of deployments
- Post-deployment verification

## Security Features

- Kubernetes Secrets for sensitive configuration
- Non-root Docker container execution
- CORS middleware protection
- bcrypt password hashing
- Rate limiting per endpoint (10 req/min for `/ask`, 5 req/min for `/login`)
- Environment-based configuration

---

**Author**: Angelo Gustilo
