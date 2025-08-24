
# MindCare AI – HR Copilot for Decision Minds

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit 1.28+](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)](https://streamlit.io)
[![Ollama Local LLM](https://img.shields.io/badge/Ollama-Local%20LLM-2ea44f.svg)](https://ollama.ai)

> **Predict attrition, personalize growth, and answer HR queries — locally with Ollama.**

MindCare AI is an **offline-first** HR analytics platform that combines predictive modeling with **local** Large Language Models to help organizations retain talent, identify growth opportunities, and streamline HR operations **without compromising data privacy**.

https://huggingface.co/spaces/ganesund/Mindcare_AI
---

## 🎯 Key Features

- **🚨 Attrition Risk Radar**: Predict attrition with explainable insights  
- **📈 Sentiment & Theme Mining**: Analyze feedback with local NLP  
- **🎓 Career Pathing & Upskilling**: AI-powered development recommendations  
- **🤖 HR Policy Copilot**: RAG-powered chatbot for instant policy Q&A  
- **🔒 Privacy-First**: 100% local processing, no external API calls

---

## 🏗️ Architecture

```

┌────────────────────────────────────────┐
│           HR Data Sources (local)      │
│  • Pulse surveys (CSV)                 │
│  • HRIS exports (CSV/Parquet)          │
│  • PTO & timesheets                    │
│  • L\&D catalog (CSV/Docs)              │
│  • HR policy PDFs/Docs                 │
└────────────────────────────────────────┘
│  (local file share)
▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   Data & Feature Layer (offline)                         │
│  • Ingestion: Python + DuckDB / SQLite                                   │
│  • Cleansing & Anonymization: PII masking, hashing                       │
│  • Features: survey\_sentiment, workload\_ratio, skill\_gap\_score,          │
│              manager\_1on1\_cadence, tenure\_weeks, internal\_moves          │
│  • Embeddings store for RAG: local (FAISS/Chroma on disk)                │
└──────────────────────────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────┐     ┌──────────────────────────────────────────┐
│  Ollama (LLMs local)     │     │   Classical Models (local, Python)      │
│  • mistral / llama3      │     │  • Sentiment: VADER/TextBlob/spaCy      │
│  • codellama for prompts │     │  • Attrition: XGBoost/LogReg (sklearn)  │
│  • RAG over HR docs      │     │  • Recommender: rules + similarity      │
└──────────────────────────┘     └──────────────────────────────────────────┘
│                           │
└──────────────┬────────────┘
▼
┌───────────────────────────────────┐
│     Streamlit/Gradio UI (local)   │
│  • Risk dashboard & drilldowns     │
│  • HR Copilot chat (RAG+LLM)       │
│  • Career paths & export (CSV/JSON)│
└───────────────────────────────────┘

````

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google AI Studio's Gemini 2.5 Pro
- [Ollama](https://ollama.ai) installed locally
- 8GB+ RAM (recommended), 10GB+ free disk space

### Installation

1) **Clone the repository**
```bash
git clone https://github.com/yourusername/mindcare-ai.git
cd mindcare-ai
````

2. **Set up Python environment**

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

3. **Install and verify Ollama**

```bash
# Follow install steps at https://ollama.ai
ollama pull mistral
ollama pull llama3

# Quick check
ollama run mistral "Hello, test message"
```

4. **Prepare sample data (optional)**

```bash
python scripts/generate_sample_data.py
```

5. **Launch the app**

```bash
streamlit run app/ui_app.py
```

6. **Open the dashboard** at `http://localhost:8501`

---

## 📁 Project Structure

```
mindcare/
├── app/
│   ├── ui_app.py           # Main Streamlit application
│   ├── rag.py              # RAG pipeline for HR Copilot
│   ├── attrition.py        # Attrition prediction models
│   ├── features.py         # Feature engineering utilities
│   ├── recommender.py      # Career pathing recommendations
│   └── storage.py          # Data management utilities
├── data/
│   ├── employees.csv       # Employee master data
│   ├── surveys.csv         # Pulse survey responses
│   ├── timesheets.csv      # Work hours tracking
│   ├── skills.csv          # Employee skills matrix
│   └── policies/           # HR policy documents
├── models/
│   ├── vector_index/       # FAISS/Chroma embeddings
│   └── attrition_lr.joblib # Trained attrition model
├── scripts/
│   ├── generate_sample_data.py
│   └── data_pipeline.py
├── tests/
├── requirements.txt
└── README.md
```

---

## 🎮 Usage Guide

### 1) Risk Radar Dashboard

* Org-wide attrition risk heatmap
* Drilldowns by practice/team/individual
* Top risk drivers (explainable AI)
* Export as CSV

### 2) Career Pathing

* Personalized development plans
* Course/mentor recommendations
* Project rotation suggestions
* Track skill progression

### 3) HR Policy Copilot

* Natural-language Q\&A over policy docs
* Source citations with RAG
* Handles complex scenarios
* Maintains conversation context

### 4) Sample Queries

**Risk Analysis**

* “Show me teams with highest attrition risk”
* “What are the top drivers for the Cloud practice?”

**Career Development**

* “Recommend growth plan for employee E\_1042”
* “What skills are most in demand for Data Engineers?”

**Policy Questions**

* “What is our parental leave policy?”
* “How many vacation days do I get?”
* “What’s the process for requesting sabbatical?”

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the repo root:

```env
# Ollama Configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=mistral

# Data Configuration
DATA_PATH=./data
VECTOR_INDEX_PATH=./models/vector_index

# Privacy Settings
ENABLE_PII_MASKING=true
ANONYMIZATION_LEVEL=high
```

### Model Configuration

Edit `config/models.yaml`:

```yaml
attrition:
  model_type: "logistic_regression"
  features:
    - "tenure_weeks"
    - "survey_sentiment"
    - "workload_ratio"
    - "manager_1on1_cadence"
  threshold: 0.3

rag:
  chunk_size: 512
  chunk_overlap: 50
  top_k: 4
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
```

---

## 📊 Sample Outputs

**Attrition Risk Export** (`attrition_risk_by_team.csv`)

| team  | avg\_risk | top\_driver\_1   | top\_driver\_2  | top\_driver\_3     |
| ----- | --------- | ---------------- | --------------- | ------------------ |
| Cloud | 0.31      | workload\_ratio  | low\_1on1       | low\_sentiment     |
| Data  | 0.27      | low\_growth      | workload\_ratio | pto\_spike         |
| AI    | 0.22      | low\_recognition | low\_1on1       | tenure\_transition |

**Career Plan JSON** (`career_plan_E_1042.json`)

```json
{
  "employee_id": "E_1042",
  "current_role": "Data Engineer",
  "target_role": "Senior Data Engineer",
  "skill_gaps": ["Snowflake perf tuning", "dbt testing"],
  "courses": ["Snowflake Performance Deep Dive", "Advanced dbt: Testing & CI"],
  "mentor": "M_309 (Senior DE, Bengaluru)",
  "project_rotation": "FinTech ETL Revamp (4 weeks)"
}
```

> **Note:** Sample outputs and benchmarks are illustrative—update with your own measurements.

---

## 🛡️ Privacy & Security

* **Local-only processing**: No data leaves your infrastructure
* **PII masking**: Automatic anonymization of sensitive fields
* **Role-based access**: Configurable permissions
* **Audit logging**: Track system interactions
* **Encryption**: Optional at-rest encryption

---

## 📈 Performance Benchmarks *(demo numbers)*

| Model              | Dataset Size  | Training Time | Accuracy     | Inference Time |
| ------------------ | ------------- | ------------- | ------------ | -------------- |
| Attrition (LogReg) | 10K employees | 2.3s          | 0.76 AUC     | 15ms           |
| Sentiment (VADER)  | 50K responses | N/A           | 0.82 F1      | 5ms            |
| RAG Pipeline       | 500 documents | 45s indexing  | \~90% hit\@k | <2s            |

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# Performance tests
pytest tests/performance/ --benchmark
```

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m "Add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request


## 🗺️ Roadmap

### Phase 1 (Current)

* ✅ Basic attrition prediction
* ✅ HR policy RAG system
* ✅ Streamlit dashboard
* ✅ Local deployment

### Phase 2

* 🔄 Advanced ML models (XGBoost, Neural Networks)
* 🔄 What-if scenario simulation
* 🔄 HRIS integrations
* 🔄 Mobile-responsive UI

### Phase 3

* 📋 Fine-tuned domain-specific LLMs
* 📋 Real-time alerting
* 📋 Advanced analytics dashboard
* 📋 Multi-tenant support

---

## 🎯 Project Goals

* **Privacy-First AI**: Enterprise-grade AI without data leakage
* **Practical Implementation**: Actionable HR insights with accessible tech
* **Open Innovation**: Contribute to the open-source HR analytics ecosystem
* **Local-First Architecture**: Prove powerful AI can run fully offline

---

## 📚 Research & Papers

* Predictive HR Analytics
* Local Language Model Deployment
* Privacy-Preserving ML
* Explainable AI in HR Context

---

## 👥 Team

* **Ganesh Sundaresan** 
* **Shashank A**
* **Simran Singh**
* **Asma Khanum**

---

<p align="center">
Made with ❤️ by the MindCare AI Team
</p>
