# SWYNEX-Model-or-API-Integration

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![Security](https://img.shields.io/badge/Security-python--dotenv-green.svg)](https://pypi.org/project/python-dotenv/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An internship submission project for **SWYNEX Technologies** titled **"Model or API Integration"**.

This project demonstrates a clean, working prototype integrating a pretrained AI model via **Hugging Face Transformers** with a modern, minimal **Streamlit** user interface.

---

## 📌 Project Overview

The application provides real-time sentiment analysis using a pretrained transformer model:

```
USER ENTERS TEXT
       ↓
HUGGING FACE MODEL (DistilBERT)
       ↓
SENTIMENT RESULT (POSITIVE / NEGATIVE)
       ↓
CONFIDENCE SCORE (XX.XX%)
```

- **Frontend Interface:** Clean, light, modern UI titled **`🤖 AI Text Analyzer`** built with Streamlit.
- **Model Pipeline:** Hugging Face `transformers` pipeline (`sentiment-analysis`) utilizing `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
- **Caching:** Loaded once into memory via `@st.cache_resource` for sub-second subsequent inferences.
- **Security:** Standard `.env.example` and `.gitignore` configuration ensuring zero credentials or secrets in source control.

---

## 🛠️ Technology Stack

- **Language:** Python 3.10+ (Tested on Python 3.12)
- **Web UI:** Streamlit
- **AI / NLP Model:** Hugging Face Transformers (`pipeline`)
- **Deep Learning Backend:** PyTorch (CPU / CUDA compatible)
- **Environment Management:** python-dotenv

---

## 📂 Project Structure

```
swynex/
├── app.py                # Streamlit web application
├── requirements.txt      # Dependency specification
├── .env.example          # Environment variable template
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation and evaluation benchmark
```

---

## 🚀 Setup & Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/eswarvaniraj-hub/SWYNEX-Model-or-API-Integration.git
cd SWYNEX-Model-or-API-Integration
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Copy `.env.example` to `.env` (optional, no secret API key is required to run the local Hugging Face model):
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

### 5. Run the Application
```bash
streamlit run app.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 🧪 Sample Inputs and Expected Outputs

The application includes sample buttons directly in the UI for instant testing:

| Sample Type | Sample Input Text | Expected Sentiment | Expected Confidence |
| :--- | :--- | :---: | :---: |
| **Positive Sample** | *"I really enjoyed using this application!"* | **POSITIVE** | `99%+` |
| **Negative Sample** | *"The application was slow and frustrating."* | **NEGATIVE** | `99%+` |

### Sample Output Example
```json
[
  {
    "label": "POSITIVE",
    "score": 0.9997
  }
]
```
- **Sentiment:** 😊 POSITIVE
- **Confidence:** 99.97%

---

## 🤖 About the Model

- **Model Identifier:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- **Architecture:** DistilBERT (Distilled version of BERT)
- **Task:** Binary Sentiment Classification (SST-2 benchmark)
- **Parameters:** ~67 million parameters, optimized for low latency and high accuracy

---

## 🔒 Security & Repository Hygiene

- **Zero Hardcoded Secrets:** No API keys are required or committed in the codebase.
- **Git Exclusion:** `.gitignore` excludes virtual environments (`venv/`), environment files (`.env`), and cache directories (`__pycache__/`, `.cache/`).

---

## 📄 License
This project is open-source under the MIT License for educational and technical evaluation purposes.
