<p align="center">
<pre align="center">
   █████╗ ███╗   ██╗ ██████╗ ███████╗██╗      ██████╗
  ██╔══██╗████╗  ██║██╔════╝ ██╔════╝██║     ██╔═══██╗
  ███████║██╔██╗ ██║██║  ███╗█████╗  ██║     ██║   ██║
  ██╔══██║██║╚██╗██║██║   ██║██╔══╝  ██║     ██║   ██║
  ██║  ██║██║ ╚████║╚██████╔╝███████╗███████╗╚██████╔╝
  ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚══════╝ ╚═════╝
 ██████╗ ██╗   ██╗███████╗████████╗██╗██╗      ██████╗
██╔════╝ ██║   ██║██╔════╝╚══██╔══╝██║██║     ██╔═══██╗
██║  ███╗██║   ██║███████╗   ██║   ██║██║     ██║   ██║
██║   ██║██║   ██║╚════██║   ██║   ██║██║     ██║   ██║
╚██████╔╝╚██████╔╝███████║   ██║   ██║███████╗╚██████╔╝
 ╚═════╝  ╚═════╝ ╚══════╝   ╚═╝   ╚═╝╚══════╝ ╚═════╝
</pre>
</p>

<div align="center">

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Russo+One&weight=700&size=28&duration=3000&pause=1000&color=DC143C&center=true&vCenter=true&random=false&width=600&lines=FULL-STACK+AI+SYSTEMS+ENGINEER;BUILDING+CLOUD-NATIVE+INTELLIGENCE;TURNING+IDEAS+INTO+DEPLOYED+REALITY)](https://git.io/typing-svg)

**I don't just build AI systems. I ship them.**

<img src="https://raw.githubusercontent.com/angelogustilo19/angelogustilo19/output/github-snake-dark.svg" alt="Snake animation" />

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

## THE SHORT VERSION

I architect and deploy production AI systems on cloud infrastructure. Not demos. Not notebooks. **Real systems** with auth, persistence, orchestration, and users hitting them concurrently.

My sweet spot? Taking an idea from "wouldn't it be cool if..." to a containerized, Kubernetes-orchestrated reality running on GCP — often at zero infrastructure cost.

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

## TECH I SHIP WITH

<div align="center">

**CLOUD & INFRA**

![GCP](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**BACKEND & AI**

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_2.5-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Kafka](https://img.shields.io/badge/Apache_Kafka-231F20?style=for-the-badge&logo=apachekafka&logoColor=white)

**FRONTEND & DATA**

![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

## WHAT I'VE BUILT

### MOMENTUM AI — Intelligent Financial Assistant

> *When a calculator meets a conversational AI, and both actually work in production.*

A hybrid AI system that knows the difference between "explain compound interest" and "calculate my debt payoff schedule" — and handles both flawlessly.

<div align="center">
<img src="assets/momentum-login.png" alt="Momentum AI Login" width="500"/>
<br><br>
<img src="assets/momentum-ai.png" alt="Momentum AI Chat" width="700"/>
<br><br>
<img src="assets/rag-platform.png" alt="Momentum AI Conversation" width="700"/>
</div>

<details>
<summary><b>SEE WHAT'S UNDER THE HOOD</b></summary>

<br>

**THE PROBLEM:** Users need both precise financial calculations AND conversational AI — but most systems do one or the other, poorly.

**THE SOLUTION:** Intent detection that routes queries to either a specialized financial engine (debt payoff, amortization, loan math) or an LLM with RAG enhancement.

**KEY ENGINEERING:**
- Multi-tier LLM fallback: Gemini 2.5 Flash → GPT-3.5 → Ollama (never fails silently)
- FAISS vector search for retrieval-augmented responses
- Per-user auth, chat history, and session persistence
- Command palette with keyboard shortcuts (Cmd/Ctrl+K)
- Full CI/CD via GitHub Actions → Docker → K3s on GCP

**STACK:** FastAPI | React 18 | Material-UI | MySQL | Kubernetes | Traefik | GCP

</details>

<div align="center">

[![Repo](https://img.shields.io/badge/VIEW_REPO-DC143C?style=for-the-badge&logo=github&logoColor=white)](https://github.com/angelogustilo19/momentum-ai-portfolio)

</div>

---

### CLOUD-NATIVE RAG PLATFORM

> *Production-grade retrieval-augmented generation. Not a Jupyter notebook — an actual deployed system.*

Re-architected from prototype to fully containerized platform handling concurrent multi-user workloads.


<details>
<summary><b>SEE WHAT'S UNDER THE HOOD</b></summary>

<br>

**THE CHALLENGE:** Turn a working prototype into something that survives real users, real load, and real production conditions.

**WHAT I BUILT:**
- FastAPI backend with authenticated REST endpoints for conversational AI
- Gemini 2.5 Flash integration with structured reasoning pipelines
- System prompts constrained for domain-specific responses (cloud architecture, finance)
- React frontend deployed as K8s-managed service
- MySQL with persistent volumes — data survives pod restarts
- Traefik ingress exposing the whole thing securely to the internet

**BATTLE SCARS (PROBLEMS I SOLVED):**
- Container image pull auth failures
- K8s DNS and service resolution nightmares
- Environment variable injection mismatches
- Database connectivity edge cases under concurrent load

**RESULT:** Multi-user deployment validated live. Zero crashes. Zero data loss.

</details>

---

### SALES PREDICTION WITH TENSORFLOW

> *When you want to know what's going to sell — and by how much.*

Deep learning regression model for sales forecasting, built with TensorFlow and grounded in solid exploratory data analysis.

<details>
<summary><b>SEE WHAT'S UNDER THE HOOD</b></summary>

<br>

**THE GOAL:** Predict sales outcomes using historical data and deep learning.

**APPROACH:**
- Exploratory data analysis to understand patterns, distributions, and feature relationships
- Feature engineering informed by EDA insights
- TensorFlow regression model for sales prediction
- Model evaluation and performance analysis

**STACK:** Python | TensorFlow | Pandas | Jupyter Notebook

</details>

<div align="center">

[![Repo](https://img.shields.io/badge/VIEW_REPO-DC143C?style=for-the-badge&logo=github&logoColor=white)](https://github.com/angelogustilo19/EDA)

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

## CURRENTLY

- Engineering AI solutions at **Customer Echoes** — building dashboards, lead management systems, and translating AI specs into business outcomes
- Obsessing over the gap between "it works on my machine" and "it works in production"
- Exploring what happens when you throw real constraints at AI systems

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

## LET'S CONNECT

I'm always down to talk about cloud architecture, AI systems that actually ship, or why Kubernetes is simultaneously the best and worst thing ever.

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/angelogustilo)
[![Email](https://img.shields.io/badge/Email-DC143C?style=for-the-badge&logo=gmail&logoColor=white)](mailto:angelogustilo19@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/angelogustilo19)

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

<div align="center">

**"The best code is code that's running somewhere, doing something, for someone."**

</div>
