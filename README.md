# CyberBullying_Suicide_Prevention
🔗 Live Demo → Launch SafeCampus

💡 What It Does
Students paste a bullying message they received. The system:

Detects if it's cyberbullying and what type (age, gender, religion, ethnicity)
Scores severity from 0–100% across 3 danger tiers
Explains which words triggered the detection (LIME)
Auto-generates a formal incident report in seconds
Alerts the college counselor if risk is CRITICAL

No surveillance. No privacy invasion. Student consents at every step.

📊 Model Performance
ModelF1 ScoreAUC-ROCAccuracyLogistic Regression ⭐0.9260.9040.871Random Forest———SVM Linear———

📁 Project Structure
├── safecampus_app.py        # Streamlit portal
├── Cyberbullying_SuicidePrevention_Review2.ipynb
├── requirements.txt
└── models/
    ├── tfidf.pkl
    ├── LogisticRegression.pkl
    ├── tfidf_multi.pkl
    ├── lr_multi.pkl
    └── meta.json

🔬 Research Papers
PaperUsed ForCyberbullying Detection using NLP — ScienceDirect, 2025Baseline approachHybrid Models for Cyberbullying — MDPI Electronics, 2021Model selection validationSuicidal Ideation Detection — MDPI IJERPH, 2022Severity scoring basisLIME "Why Should I Trust You?" — KDD 2016Explainability moduleSentence Transformer Fine-Tuning — Springer, 2024Future work (BERT)

🗺️ Future Work

Short term — DistilBERT fine-tuning, full multi-class classifier
Medium term — FastAPI backend, browser extension, counselor SMS alerts
Long term — Multi-modal (image + text), Hindi/Hinglish support


📦 Dataset
Cyberbullying Classification — Kaggle — 47,692 labelled tweets across 6 categories

🆘 Emergency Helplines (India)
iCall: 9152987821  |  Vandrevala: 1860-2662-345  |  icallhelpline.org
