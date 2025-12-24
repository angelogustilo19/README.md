# Angelo Gustilo
Full-Stack AI Systems Engineer

**Core Stack:** Google Cloud Platform (Compute Engine), Kubernetes (k3s), Docker, FastAPI, React, Gemini 2.5 Flash, MySQL

---

## Featured Project
### Cloud-Native Full-Stack RAG Platform

Designed, implemented, and deployed a production-style cloud-native Retrieval-Augmented Generation (RAG) system, re-architecting an initial prototype into a fully containerized and orchestrated platform running on Google Cloud infrastructure.

The system was rebuilt end-to-end to support authenticated access, persistent state, secure networking, and concurrent multi-user workloads under real deployment conditions.

---

## System Architecture & Implementation

### Backend & API Layer
- Implemented a FastAPI-based backend exposing authenticated REST endpoints for conversational AI workflows and retrieval-augmented reasoning.
- Integrated Google Gemini 2.5 Flash to perform structured reasoning, financial analysis, and domain-specific response generation.
- Designed system prompts and request pipelines to constrain model behavior toward cloud architecture concepts and real-world banking and finance scenarios.
- Implemented request validation, error handling, and fallback logic to maintain stability under transient service failures.

### Frontend Layer
- Developed a React-based client application enabling authenticated, real-time interaction with the AI system.
- Containerized frontend assets and deployed them as Kubernetes-managed services for consistent delivery and routing.

---

## Containerization & Orchestration

- Fully containerized frontend and backend services using Docker with environment-driven configuration.
- Published container images to Docker Hub and deployed them using Kubernetes manifests.
- Orchestrated services using k3s on a Google Cloud virtual machine, enabling lightweight yet production-style cluster management.
- Defined Kubernetes Deployments, Services, and Ingress resources to manage service discovery, internal routing, and external traffic exposure.
- Configured Traefik Ingress to securely expose the application to the public internet.

---

## State Management & Persistence

- Deployed MySQL as a stateful Kubernetes workload with persistent volume claims to ensure data durability across pod restarts.
- Designed schemas and query flows to support per-user authentication, session tracking, and persistent chat history.
- Maintained strict separation between stateless application services and stateful storage components.

---

## Security & Configuration Management

- Externalized configuration using Kubernetes ConfigMaps and Secrets to separate code from environment-specific values.
- Implemented secure credential handling for database access and external AI service integration.
- Enforced authenticated access to backend services and protected internal service communication within the cluster.

---

## Reliability, Debugging & Validation

- Diagnosed and resolved complex deployment issues including:
  - Container image pull authentication failures
  - Kubernetes DNS and service resolution errors
  - Environment variable injection mismatches
  - Database connectivity and persistence edge cases
- Validated system reliability through live multi-user deployment and demonstration with concurrent users accessing the platform simultaneously.
- Verified stability under concurrent load without service crashes or data loss.

---

## Cost & Resource Optimization

- Designed the platform to operate entirely within Google Cloud free-tier constraints.
- Achieved a fully functional cloud AI deployment at zero infrastructure cost through efficient container sizing, lightweight orchestration, and resource-aware architecture.

---

## Experience
**AI Solutions — Customer Echoes**  
*May 2025 – Present*

- Designed AI-assisted sentiment analysis workflows to process unstructured customer feedback from reviews, surveys, and social channels.
- Translated AI-derived insights into churn risk assessments, competitive benchmarking, and executive-facing CX strategy briefs.
- Collaborated cross-functionally to align AI-generated signals with customer journey optimization and retention initiatives.

**Tools:** ChatGPT, Perplexity AI, Microsoft Excel

---

## Technical Skills
- **Cloud & Infrastructure:** Google Cloud Platform, Kubernetes (k3s), Docker
- **Backend & APIs:** FastAPI, RESTful services
- **Frontend:** React
- **AI & Data:** Gemini, Python, SQL, applied machine learning fundamentals
- **Databases:** MySQL
- **Tooling:** Git, Jupyter Notebook

---

## Background
Master’s student in Information Systems Management with a foundation in business administration and applied cloud-native AI systems.
