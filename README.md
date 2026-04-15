# 🛡️ SafeCampus — Anti-Ragging & Cyberbullying Prevention Portal

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Security](https://img.shields.io/badge/Security-Hardened-brightgreen)

> A victim-initiated reporting portal that uses DistilBERT transformer models to detect cyberbullying, assess severity, and auto-generate formal incident reports for college anti-ragging cells.

---

## 💡 What It Does

Students paste a bullying message they received. The system:

1. **Detects** if it's cyberbullying using a DistilBERT toxicity classifier
2. **Classifies context** via zero-shot DistilBART (threat, ragging, suicide risk, etc.)
3. **Scores severity** from 0–100% across 4 danger tiers (Low → Critical)
4. **Explains** which words triggered the detection (LIME)
5. **Auto-generates** a formal incident report in seconds
6. **Alerts** the college counselor if risk is CRITICAL

**No surveillance. No privacy invasion. Student consents at every step.**

---

## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/Aryan-710/AntiRagging.git
cd AntiRagging

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run safecampus_app.py
```

### Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo → select `safecampus_app.py` as the main file
4. Deploy ✓

The app includes `runtime.txt` (Python 3.11), `packages.txt`, and `.streamlit/config.toml` for zero-config deployment.

### Deploy to Heroku / Railway

```bash
# Uses the included Procfile
git push heroku main
```

---

## 🔒 Security Measures

| Measure | Implementation |
|---|---|
| **Input Sanitization** | HTML tags stripped via `bleach`, control characters removed, unicode normalized |
| **XSS Prevention** | All user-derived content escaped with `html.escape()` before rendering |
| **Rate Limiting** | Max 10 analyses per minute per session (DoS protection) |
| **XSRF Protection** | Enabled in `.streamlit/config.toml` |
| **No Data Persistence** | Zero storage — all analysis is ephemeral |
| **Dependency Pinning** | Exact version pins for supply-chain safety |
| **Input Validation** | Server-side length check, URL-only rejection, whitespace-only rejection |

---

## 📊 Model Architecture

| Component | Details |
|---|---|
| **Toxicity Classifier** | `martin-ha/toxic-comment-model` (DistilBERT, ~66M params) |
| **Context Classifier** | `valhalla/distilbart-mnli-12-3` (Zero-shot, multi-label) |
| **Explainability** | LIME word-level explanation |
| **Severity Scoring** | Composite: 50% model probability + contextual flag weights |

### Context Labels Detected

- Insult / Harassment
- Direct Threat
- Academic / Hierarchy Abuse
- Severe Ragging / Hazing
- Suicide / Self-harm Risk

---

## 📁 Project Structure

```
AntiRagging/
├── safecampus_app.py                    # Main Streamlit application
├── requirements.txt                     # Pinned Python dependencies
├── Procfile                             # Heroku/Railway deployment
├── runtime.txt                          # Python version pin
├── packages.txt                         # System dependencies (Streamlit Cloud)
├── .gitignore                           # Excludes caches, secrets, pkl files
├── .streamlit/
│   ├── config.toml                      # Production server config
│   └── secrets.toml.example             # Secrets template
├── models/                              # Legacy TF-IDF models (gitignored)
├── hinglish_cyberbullying_dataset.csv   # Hinglish training data
├── Cyberbullying_SuicidePrevention_Review2.ipynb  # Research notebook
└── README.md
```

---

## 🔬 Research Papers

| Paper | Used For |
|---|---|
| Cyberbullying Detection using NLP — ScienceDirect, 2025 | Baseline approach |
| Hybrid Models for Cyberbullying — MDPI Electronics, 2021 | Model selection validation |
| Suicidal Ideation Detection — MDPI IJERPH, 2022 | Severity scoring basis |
| LIME "Why Should I Trust You?" — KDD 2016 | Explainability module |
| Sentence Transformer Fine-Tuning — Springer, 2024 | Future work (BERT) |

---

## 🗺️ Future Work Roadmap

| Phase | Goals |
|---|---|
| **Phase 1** | Fine-tune DistilBERT, true multi-class classifier (6 bullying types) |
| **Phase 2** | FastAPI backend, browser extension, counselor SMS/email alerts |
| **Phase 3** | Multi-modal (image + text via CLIP), Hinglish support (IndicBERT) |

---

## 📦 Dataset

- [Cyberbullying Classification — Kaggle](https://www.kaggle.com/) — 47,692 labelled tweets across 6 categories
- `hinglish_cyberbullying_dataset.csv` — Custom Hinglish bullying dataset for Indian college context

---

## 🆘 Emergency Helplines (India)

| Service | Contact |
|---|---|
| **Anti-Ragging Helpline** | `1800-180-5522` |
| **iCall** | `9152987821` |
| **Vandrevala Foundation** | `1860-2662-345` |
| **iCall Chat** | [icallhelpline.org](https://icallhelpline.org) |

---

## 📝 License

MIT License — See [LICENSE](LICENSE) for details.
