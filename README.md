<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:000000,50:1a1a1a,100:2E8B57&height=180&section=header&text=ANGELO%20GUSTILO&fontSize=50&fontColor=ffffff&fontAlign=50&fontAlignY=45&desc=FULL-STACK%20AI%20SYSTEMS%20ENGINEER&descSize=18&descAlignY=70"/>

<div align="center">

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Russo+One&weight=700&size=28&duration=3000&pause=1000&color=2E8B57&center=true&vCenter=true&random=false&width=600&lines=FULL-STACK+AI+SYSTEMS+ENGINEER;BUILDING+CLOUD-NATIVE+INTELLIGENCE;TURNING+IDEAS+INTO+DEPLOYED+REALITY)](https://git.io/typing-svg)

**I don't just build AI systems. I ship them.**

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

## THE SHORT VERSION

I architect and deploy production AI systems on cloud infrastructure. Not demos. Not notebooks. **Real systems** with auth, persistence, orchestration, and users hitting them concurrently.

My sweet spot? Taking an idea from "wouldn't it be cool if..." to a containerized, Kubernetes-orchestrated reality running on GCP - often at zero infrastructure cost.

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

### MOMENTUM AI - Cloud Computing & Banking Expert

> *Enterprise-grade AI that speaks cloud architecture and banking - deployed in production.*

A hybrid AI system built for cloud computing guidance and banking expertise, handling everything from GCP infrastructure questions to financial analysis with precision.

<div align="center">
<img src="assets/momentum-login.png" alt="Momentum AI Login" width="500"/>
<br><br>

![Momentum AI Demo](assets/chatbot-demo.gif)

</div>

<details>
<summary><b>SEE WHAT'S UNDER THE HOOD</b></summary>

<br>

**THE PROBLEM:** Users need expert guidance on cloud architecture AND banking concepts - but most AI systems lack domain-specific depth.

**THE SOLUTION:** Intent detection that routes queries to specialized engines for cloud computing (GCP, Kubernetes, infrastructure) or banking expertise (financial analysis, loan calculations) with RAG enhancement.

**KEY ENGINEERING:**
- Multi-tier LLM fallback: Gemini 2.5 Flash → GPT-3.5 → Ollama (never fails silently)
- FAISS vector search for retrieval-augmented responses
- Per-user auth, chat history, and session persistence
- Command palette with keyboard shortcuts (Cmd/Ctrl+K)
- Full CI/CD via GitHub Actions → Docker → K3s on GCP

**STACK:** FastAPI | React 18 | Material-UI | MySQL | Kubernetes | Traefik | GCP

</details>

<div align="center">

[![Repo](https://img.shields.io/badge/VIEW_REPO-2E8B57?style=for-the-badge&logo=github&logoColor=white)](https://github.com/angelogustilo19/momentum-ai-portfolio)

</div>

---

### CLOUD-NATIVE RAG PLATFORM

> *Production-grade retrieval-augmented generation. Not a Jupyter notebook - an actual deployed system.*

Re-architected from prototype to fully containerized platform handling concurrent multi-user workloads.

<div align="center">
<img src="assets/k8s-deployment.png" alt="Kubernetes Deployment on GCP" width="700"/>
</div>

<details>
<summary><b>SEE WHAT'S UNDER THE HOOD</b></summary>

<br>

**THE CHALLENGE:** Turn a working prototype into something that survives real users, real load, and real production conditions.

**WHAT I BUILT:**
- FastAPI backend with authenticated REST endpoints for conversational AI
- Gemini 2.5 Flash integration with structured reasoning pipelines
- System prompts constrained for domain-specific responses (cloud architecture, finance)
- React frontend deployed as K8s-managed service
- MySQL with persistent volumes - data survives pod restarts
- Traefik ingress exposing the whole thing securely to the internet

**BATTLE SCARS (PROBLEMS I SOLVED):**
- Container image pull auth failures
- K8s DNS and service resolution nightmares
- Environment variable injection mismatches
- Database connectivity edge cases under concurrent load

**RESULT:** Multi-user deployment validated live. Zero crashes. Zero data loss.

</details>

---

### SALES PREDICTION WITH REAL-TIME KAFKA STREAMING

> *Predict sales. Stream predictions. Monitor in real-time.*

End-to-end ML pipeline combining TensorFlow regression with Apache Kafka for live prediction streaming and monitoring.

<div align="center">
<img src="assets/sales-prediction-analysis.png" alt="Sales Prediction Analysis" width="700"/>
<br><br>
<img src="assets/kafdrop-streaming.png" alt="Kafdrop Real-Time Streaming" width="700"/>
</div>

<details>
<summary><b>SEE WHAT'S UNDER THE HOOD</b></summary>

<br>

**THE GOAL:** Predict aggregated sales performance and stream predictions in real-time for live monitoring.

**DATA PROCESSING:**
- EDA on 500 customer transaction records revealing seasonality patterns, marketing ROI, and churn demographics
- Missing value imputation and outlier removal using IQR method
- Feature engineering informed by correlation analysis

**ML PIPELINE:**
- Linear regression baseline for interpretability
- TensorFlow neural network (2 hidden layers) capturing non-linear relationships between marketing spend, seasonality, and sales

**REAL-TIME STREAMING:**
- Kafka message broker streams 81 prediction records to `sales_predictions` topic
- Kafdrop web UI visualizes predictions with error analysis
- Schema includes prediction index, predicted/actual sales, and error metrics

**STACK:** TensorFlow | Pandas | Apache Kafka | Kafdrop | Docker | Docker Compose

</details>

<div align="center">

[![Repo](https://img.shields.io/badge/VIEW_REPO-2E8B57?style=for-the-badge&logo=github&logoColor=white)](https://github.com/angelogustilo19/EDA)

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

## CURRENTLY

- **Open to work** - yes, this is my way of saying *please hire me*
- Mass applying to jobs like my Kafka producer pushes messages - high throughput, hoping for at least one successful delivery
- Still obsessing over the gap between "it works on my machine" and "it works in production"
- Available immediately. Will deploy AI systems for coffee. Or money. Preferably both.

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

## LET'S CONNECT

I'm always down to talk about cloud architecture, AI systems that actually ship, why Kubernetes is simultaneously the best and worst thing ever - or **job opportunities** (seriously, my inbox is ready).

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/angelogustilo)
[![Email](https://img.shields.io/badge/Email-2E8B57?style=for-the-badge&logo=gmail&logoColor=white)](mailto:angelogustilo19@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/angelogustilo19)

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">

<div align="center">

**"The best code is code that's running somewhere, doing something, for someone."**

</div>
